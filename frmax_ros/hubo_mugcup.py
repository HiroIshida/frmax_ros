import time
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import rospkg
import rospy
import trimesh
from frmax2.core import DGSamplerConfig
from frmax2.dmp import determine_dmp_metric
from geometry_msgs.msg import PoseStamped
from movement_primitives.dmp import DMP
from nav_msgs.msg import Path as RosPath

try:
    from skmp.constraint import (
        AbstractEqConst,
        BoxConst,
        CollFreeConst,
        ConfigPointConst,
        PoseConstraint,
    )
    from skmp.kinematics import (
        ArticulatedCollisionKinematicsMap,
        ArticulatedEndEffectorKinematicsMap,
    )
    from skmp.robot.pr2 import PR2Config
    from skmp.robot.utils import get_robot_state, set_robot_state
    from skmp.satisfy import (
        SatisfactionConfig,
        satisfy_by_optimization,
        satisfy_by_optimization_with_budget,
    )
    from skmp.solver.interface import Problem
    from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
except ImportError:

    class ArticulatedCollisionKinematicsMap:
        ...

    class ArticulatedEndEffectorKinematicsMap:
        ...


from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import MeshLink
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from tinyfk import BaseType, RotationType

from frmax_ros._hubo_mugcup_recovery import RecoveryMixIn
from frmax_ros.rollout import (
    AutomaticTrainerBase,
    PlanningScene,
    RolloutAbortedException,
    RolloutExecutorBase,
    speak,
)
from frmax_ros.utils import CoordinateTransform


class PathPlanner:
    pr2: PR2
    scene: PlanningScene

    def __init__(self, pr2: PR2, scene: PlanningScene, visualize: bool = True):
        self.pr2 = pr2
        self.scene = scene

    @lru_cache(maxsize=None)
    def get_box_const(self, arm: Literal["larm", "rarm"], base_type: BaseType) -> BoxConst:
        plan_conf = PR2Config(control_arm=arm, base_type=base_type)
        box_const = plan_conf.get_box_const()
        box_const.lb -= 1e-4
        box_const.ub += 1e-4
        return box_const

    @lru_cache(maxsize=None)
    def get_collision_kin(
        self, arm: Literal["larm", "rarm"], base_type: BaseType
    ) -> ArticulatedCollisionKinematicsMap:
        plan_conf = PR2Config(control_arm=arm, base_type=base_type)
        return plan_conf.get_collision_kin()

    @lru_cache(maxsize=None)
    def get_ef_kin(
        self, arm: Literal["larm", "rarm"], base_type: BaseType
    ) -> ArticulatedEndEffectorKinematicsMap:
        plan_conf = PR2Config(control_arm=arm, base_type=base_type)
        return plan_conf.get_endeffector_kin(
            rot_type=RotationType.XYZW
        )  # 2023/12/30: found that XYZW was not propagated. Fixed  at skmp's 6bd8515 but not sure it works after this commit.

    def _setup_constraints(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
        *,
        co_object: Optional[Coordinates] = None,
        consider_table: bool = True,
        consider_dummy: bool = True,
        consider_object: bool = True,
        base_type: BaseType = BaseType.FIXED,
    ) -> Tuple[AbstractEqConst, Optional[CollFreeConst], BoxConst]:
        if consider_object:
            assert co_object is not None

        plan_conf = PR2Config(control_arm=arm, base_type=base_type)
        sdfs = []
        if consider_table:
            sdfs.append(self.scene.table.sdf)
        if consider_dummy:
            rospy.loginfo("dummy object is considered")
            sdfs.append(self.scene.dummy_obstacle.sdf)
        if consider_object:
            sdfs.append(self.scene.target_object.sdf)
        if len(sdfs) > 0:
            usdf = UnionSDF(sdfs)
            colkin = self.get_collision_kin(arm, base_type)
            ineq_const = CollFreeConst(colkin, usdf, self.pr2, only_closest_feature=True)
        else:
            ineq_const = None

        efkin = self.get_ef_kin(arm, base_type)
        if isinstance(target, Coordinates):
            eq_const = PoseConstraint.from_skrobot_coords([target], efkin, self.pr2)
        else:
            eq_const = ConfigPointConst(target)

        box_const = self.get_box_const(arm, base_type)
        return eq_const, ineq_const, box_const

    def is_feasible_target(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
        co_object: Optional[Coordinates] = None,
    ) -> Optional[bool]:
        base_type = BaseType.PLANER

        eq_const, ineq_const, box_const = self._setup_constraints(
            target,
            arm,
            co_object=co_object,
            consider_table=False,
            consider_dummy=False,
            consider_object=True,
            base_type=base_type,
        )
        box_const.lb -= 3.0
        box_const.ub += 3.0

        satis_con = SatisfactionConfig(acceptable_error=1e-5, disp=False, n_max_eval=50)
        joint_names = PR2Config(control_arm=arm).get_control_joint_names()
        q_init = get_robot_state(self.pr2, joint_names, base_type=base_type)
        ret = satisfy_by_optimization_with_budget(
            eq_const, box_const, None, q_init, config=satis_con, n_trial_budget=100
        )
        assert ret.success, "as base type is planer, the IK should be always feasible"
        if ineq_const is not None:
            return ineq_const.is_valid(ret.q)

    def solve_ik(
        self,
        co: Coordinates,
        arm: Literal["larm", "rarm"],
        q_seed: np.ndarray,
        co_object: Optional[Coordinates] = None,
        consider_table: bool = True,
        consider_dummy: bool = True,
        consider_object: bool = True,
        use_skrobot: bool = False,
    ) -> Optional[np.ndarray]:
        eq_const, ineq_const, box_const = self._setup_constraints(
            co,
            arm,
            co_object=co_object,
            consider_table=consider_table,
            consider_dummy=consider_dummy,
            consider_object=consider_object,
        )
        if use_skrobot:
            arm_robot = self.pr2.rarm if arm == "rarm" else self.pr2.larm
            move_target = self.pr2.rarm_end_coords if arm == "rarm" else self.pr2.larm_end_coords
            q = arm_robot.inverse_kinematics(co, move_target=move_target, seed=q_seed)
            if isinstance(q, bool) and q == False:
                rospy.loginfo("callision agnonistic IK failed")
                return None
            assert isinstance(q, np.ndarray)
            if ineq_const is not None and not ineq_const.is_valid(q):
                rospy.loginfo("solved Ik but in collision")
                return None
            return q
        else:
            satis_con = SatisfactionConfig(acceptable_error=1e-5, disp=False, n_max_eval=50)
            ret = satisfy_by_optimization(eq_const, box_const, ineq_const, q_seed, config=satis_con)
            if ret.success:
                rospy.loginfo("IK solved")
                return ret.q
            else:
                rospy.loginfo("IK failed")
                return None

    def plan_path(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
        *,
        q_start: Optional[np.ndarray] = None,
        co_object: Optional[Coordinates] = None,
        consider_table: bool = True,
        consider_dummy: bool = True,
        consider_object: bool = True,
    ) -> Optional[Trajectory]:

        eq_const, ineq_const, box_const = self._setup_constraints(
            target,
            arm,
            co_object=co_object,
            consider_table=consider_table,
            consider_dummy=consider_dummy,
            consider_object=consider_object,
        )
        joint_names = PR2Config(control_arm=arm).get_control_joint_names()
        if q_start is None:
            q_start = get_robot_state(self.pr2, joint_names)
        problem = Problem(q_start, box_const, eq_const, ineq_const, None)

        ompl_config = OMPLSolverConfig(n_max_call=5000, simplify=True, n_max_satisfaction_trial=20)
        ompl_solver = OMPLSolver.init(ompl_config)
        ompl_solver.setup(problem)

        ret = ompl_solver.solve()
        return ret.traj


class GraspingPlanerTrajectory:
    seq_tf_ef_to_nominal: List[CoordinateTransform]
    pregrasp_gripper_pos: ClassVar[float] = 0.03

    # @classmethod
    # def get_goal_position_scaling(cls) -> np.ndarray:
    #     xytheta_scaling = np.array([0.01, 0.01, np.deg2rad(5.0)])
    #     goal_position_scaling = xytheta_scaling
    #     return goal_position_scaling

    # @classmethod
    # def get_force_scaling(cls) -> np.ndarray:
    #     xytheta_scaling = np.array([0.03, 0.03, np.deg2rad(5.0)])
    #     force_scalineg = xytheta_scaling * 200
    #     return force_scalineg

    def __init__(self, param: np.ndarray, dt: float = 0.1, *, im_using_this_in_demo: bool = True):
        if dt != 0.1:
            # double check
            assert not im_using_this_in_demo
        assert param.shape == (3 * 6,)
        n_split = 100
        start = np.array([-0.06, -0.045, 0.0])
        goal = np.array([-0.0, -0.045, 0.0])
        diff_step = (goal - start) / (n_split - 1)
        traj_default = np.array([start + diff_step * i for i in range(n_split)])
        n_weights_per_dim = 6
        dmp = DMP(3, execution_time=1.0, n_weights_per_dim=n_weights_per_dim, dt=dt)
        dmp.imitate(np.linspace(0, 1, n_split), traj_default.copy())
        dmp.configure(start_y=traj_default[0])

        dmp.forcing_term.weights_ += param.reshape(-1, n_weights_per_dim)
        _, planer_traj = dmp.open_loop()

        height = 0.065
        tf_seq = []
        for pose in planer_traj:
            trans = np.array([pose[0], pose[1], height])
            co = Coordinates(trans, rot=[pose[2], 0, 0.5 * np.pi])
            tf_ef_to_nominal = CoordinateTransform.from_skrobot_coords(co, "ef", "nominal")
            tf_seq.append(tf_ef_to_nominal)
        self.seq_tf_ef_to_nominal = tf_seq

    def instantiate(
        self,
        tf_object_to_base: CoordinateTransform,
        object_recog_error: np.ndarray,
    ) -> List[CoordinateTransform]:
        assert tf_object_to_base.src == "object"
        assert tf_object_to_base.dest == "base"
        x_error, y_error, yaw_error = object_recog_error
        tf_nominal_to_object = CoordinateTransform.from_skrobot_coords(
            Coordinates(pos=[x_error, y_error, 0], rot=[yaw_error, 0.0, 0.0]), "nominal", "object"
        )
        tf_nominal_to_base = tf_nominal_to_object * tf_object_to_base
        tf_ef_to_base_list = []
        for tf_ef_to_nominal in self.seq_tf_ef_to_nominal:
            tf_ef_to_base = tf_ef_to_nominal * tf_nominal_to_base
            tf_ef_to_base_list.append(tf_ef_to_base)
        return tf_ef_to_base_list

    def get_path_msg(
        self,
        tf_object_to_base: CoordinateTransform,
        object_recog_error: np.ndarray,
    ) -> RosPath:
        tf_ef_to_base_list = self.instantiate(tf_object_to_base, object_recog_error)
        path_msg = RosPath()
        path_msg.header.frame_id = "base_footprint"
        path_msg.header.stamp = rospy.Time.now()
        for tf_ef_to_base in tf_ef_to_base_list:
            ros_pose = tf_ef_to_base.to_ros_pose()
            ros_pose_stamped = PoseStamped()
            ros_pose_stamped.header = path_msg.header
            ros_pose_stamped.pose = ros_pose
            path_msg.poses.append(ros_pose_stamped)
        return path_msg


class MugcupGraspRolloutExecutor(RecoveryMixIn, RolloutExecutorBase):
    path_planner: PathPlanner
    tf_object_to_april: CoordinateTransform
    pregrasp_gripper_pos: ClassVar[float] = 0.03

    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = Path(rospack.get_path("frmax_ros"))
        mug_model_path = pkg_path / "model" / "hubolab_mug.stl"
        mesh = trimesh.load_mesh(mug_model_path)
        mugcup = MeshLink(mesh, with_sdf=True)
        super().__init__(mugcup)

        self.path_planner = PathPlanner(self.pr2, self.scene)

    def initialize_robot(self):
        self.pr2.reset_manip_pose()
        self.pr2.r_shoulder_pan_joint.joint_angle(-1.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_pan_joint.joint_angle(1.1)
        self.pr2.l_shoulder_lift_joint.joint_angle(+0.0)
        self.pr2.l_elbow_flex_joint.joint_angle(-1.2)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)
        self.pr2.head_tilt_joint.joint_angle(+1.2)
        self.ri.angle_vector(self.pr2.angle_vector(), time_scale=5.0)
        self.ri.move_gripper("larm", self.pregrasp_gripper_pos)
        self.ri.move_gripper("rarm", self.pregrasp_gripper_pos)
        self.ri.wait_interpolation()

    def get_tf_object_to_base(self, reset: bool = True) -> CoordinateTransform:
        if reset:
            self.pose_provider.reset()
        tf_april_to_base = self.pose_provider.get_tf_object_to_base()
        tf_object_to_april = CoordinateTransform(
            np.array([0.0, -0.0, -0.098]), np.eye(3), "object", "april"
        )
        return tf_object_to_april * tf_april_to_base

    def get_auto_annotation(self) -> Optional[bool]:
        gripper_pos = self.ri.gripper_states["larm"].process_value
        if gripper_pos > 0.004:
            return True
        elif gripper_pos < 0.002:
            return False
        else:
            return None

    def get_policy_dof(self) -> int:
        return 18

    def rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        assert param.shape == (self.get_policy_dof(),)
        planer_traj = GraspingPlanerTrajectory(param)
        try:
            tf_object_to_base = self.get_tf_object_to_base()
        except TimeoutError:
            reason = "failed to get object pose"
            raise RolloutAbortedException(reason, True)
        self.scene.update(tf_object_to_base.to_skrobot_coords())

        x_pos, y_pos = tf_object_to_base.trans[:2]
        if x_pos > 0.53 or y_pos > 0.2:
            reason = f"invalid object position ({x_pos}, {y_pos})"
            raise RolloutAbortedException(reason, False)

        tf_ef_to_base_seq = planer_traj.instantiate(tf_object_to_base, error)
        path_msg = planer_traj.get_path_msg(tf_object_to_base, error)
        self.pub_grasp_path.publish(path_msg)
        rospy.loginfo("Published grasp path")

        joint_names = PR2Config(control_arm="larm").get_control_joint_names()

        # if start position is in collision, consider execution fails
        co_init = tf_ef_to_base_seq[0].to_skrobot_coords()
        co_object = tf_object_to_base.to_skrobot_coords()
        if not self.path_planner.is_feasible_target(co_init, "larm", co_object=co_object):
            rospy.loginfo("the object will be in collision")
            return False

        # plan reaching
        def plan_whole(q_init) -> Optional[Trajectory]:
            q_traj_reaching = self.path_planner.plan_path(
                co_init, "larm", q_start=q_init, co_object=co_object
            )
            if q_traj_reaching is None:
                rospy.logwarn("(inside) planning failed")
                return None

            # plan grasping using uncalibrated pose
            # this to check if the trajectory is not in collision
            # at least without calibration
            # NOTE that this is not used for actual execution
            q_now = q_traj_reaching[-1]
            for tf_ef_to_base in tf_ef_to_base_seq:
                q_now = self.path_planner.solve_ik(
                    tf_ef_to_base.to_skrobot_coords(),
                    "larm",
                    q_now,
                    consider_table=True,
                    consider_object=False,
                    consider_dummy=False,
                )
                if q_now is None:
                    rospy.logwarn("(inside) IK failed")
                    return None
            return q_traj_reaching

        q_traj_reaching = None
        q_init = get_robot_state(self.pr2, joint_names)
        for _ in range(5):
            ret = plan_whole(q_init)
            if ret is not None:
                q_traj_reaching = ret
                break
        if q_traj_reaching is None:
            raise RolloutAbortedException("planning failed", False)

        # now execute reaching and grasping
        times_reaching = [0.1] * 17 + [0.6] * 2 + [1.0]
        q_traj_reaching = q_traj_reaching.resample(20).numpy()
        self.send_command_to_real_robot(q_traj_reaching, times_reaching, "larm")

        # calibrate and check pose again
        rospy.loginfo("calibrating")
        self.ri.move_gripper("larm", 0.0)
        self.offset_prover.reset()
        self.pose_provider.reset(acceptable_xy_std=0.005, acceptable_theta_std=np.deg2rad(5.0))
        time.sleep(3.0)
        try:
            offset = self.offset_prover.get_offset()
        except TimeoutError:
            self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
            reason = "failed to get calibration"
            raise RolloutAbortedException(reason, False)
        try:
            tf_object_to_base_again = self.get_tf_object_to_base(reset=False)
            # if the object is not in the same position, this means the robot collided
            # with the object. consider execution fails
            distance = np.linalg.norm(
                tf_object_to_base_again.trans[:2] - tf_object_to_base.trans[:2]
            )
            yaw_new = rpy_angle(tf_object_to_base_again.rot)[0][0]
            yaw_original = rpy_angle(tf_object_to_base.rot)[0][0]
            rospy.loginfo(f"object moved from {distance}m / {abs(yaw_new - yaw_original)}rad")
            if distance > 0.01 or abs(yaw_new - yaw_original) > np.deg2rad(5.0):
                self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
                speak("object moved! This means the robot may have collided with the object")
                raise RolloutAbortedException("object moved", False)

        except TimeoutError:
            self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
            reason = "failed to get object pose"
            raise RolloutAbortedException(reason, True)

        # move to pregrasp
        self.ri.move_gripper("larm", planer_traj.pregrasp_gripper_pos)

        # plan grasping using the calibrated pose
        q_list = [q_traj_reaching[-1]]
        for tf_ef_to_base in tf_ef_to_base_seq:
            co = tf_ef_to_base.to_skrobot_coords()
            co.translate(-offset, wrt="world")
            q_now = self.path_planner.solve_ik(
                co,
                "larm",
                q_list[-1],
                consider_table=False,
                consider_object=False,
                consider_dummy=False,
                use_skrobot=True,
            )
            if q_now is None:
                # go back to home pose
                self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
                self.ri.move_gripper("larm", 0.0)
                reason = "IK failed with calibration"
                raise RolloutAbortedException(reason, False)
            q_list.append(q_now)

        q_traj_grasping = np.array(q_list)
        set_robot_state(self.pr2, joint_names, q_traj_grasping[-1])

        times_grasping = [0.3] * len(q_traj_grasping)
        self.send_command_to_real_robot(q_traj_grasping, times_grasping, "larm")
        self.ri.move_gripper("larm", 0.0, effort=100)

        # shake back and force
        av_now = self.pr2.angle_vector()
        self.pr2.larm.move_end_pos([-0.06, 0.0, 0.0])
        self.pr2.larm.move_end_pos([0.0, 0.0, 0.025], wrt="world")  # slide to z
        self.ri.angle_vector(self.pr2.angle_vector(), time_scale=1.0, time=0.5)
        self.ri.wait_interpolation()
        self.ri.angle_vector(av_now, time_scale=1.0, time=0.5)
        self.ri.wait_interpolation()
        time.sleep(0.5)
        annot = self.get_auto_annotation()
        assert annot is not None
        self.ri.move_gripper("larm", 0.08)

        # back to initial pose
        self.send_command_to_real_robot(q_traj_grasping[::-1], [0.5] * len(q_traj_grasping), "larm")
        self.ri.move_gripper("larm", 0.0)
        self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
        return annot


class MugcupGraspTrainer(AutomaticTrainerBase):
    @classmethod
    def init(cls) -> "MugcupGraspTrainer":
        ls_param = np.ones(21)
        ls_err = np.array([0.005, 0.005, np.deg2rad(5.0)])
        config = DGSamplerConfig(
            n_mc_param_search=30,
            c_svm=10000,
            integration_method="mc",
            n_mc_integral=300,
            r_exploration=1.0,
            learning_rate=1.0,
        )
        return super().init(ls_param, ls_err, config)  # type: ignore

    @staticmethod
    def get_rollout_executor() -> RolloutExecutorBase:
        return MugcupGraspRolloutExecutor()

    @staticmethod
    def sample_situation() -> np.ndarray:
        b_min = np.array([-0.03, -0.03, -np.pi * 0.2])
        b_max = np.array([0.03, 0.03, np.pi * 0.2])
        return np.random.uniform(b_min, b_max)

    @staticmethod
    def is_valid_param(param: np.ndarray) -> bool:
        scale = GraspingPlanerTrajectory.get_goal_position_scaling()
        param_goal = param[-3:]
        x, y, yaw = param_goal * scale
        if abs(x) > 0.04 or abs(yaw) > np.deg2rad(45.0):
            return False
        return True

    @staticmethod
    def get_project_name() -> str:
        return "mugcup"


if __name__ == "__main__":
    test_with_nominal = True
    if test_with_nominal:
        param_metric = determine_dmp_metric(6, 0.25 * np.array([0.03, 0.03, 0.3]))
        e = MugcupGraspRolloutExecutor()
        e.recover()
        for _ in range(1000):
            z = np.random.randn(18)
            param = param_metric.M @ z
            e.rollout(param, np.zeros(3))
        assert False
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action="store_true", help="resume")
        parser.add_argument("--refine", action="store_true", help="refine")
        parser.add_argument("--episode", type=int, help="episode to load")
        args = parser.parse_args()
        np.random.seed(0)
        if args.refine:
            try:
                trainer = MugcupGraspTrainer.load_refined()
            except FileNotFoundError:
                trainer = MugcupGraspTrainer.load()
            for _ in range(50):
                trainer.step_refinement()
        else:
            if args.resume:
                trainer = MugcupGraspTrainer.load(args.episode)
            else:
                trainer = MugcupGraspTrainer.init()
            for _ in range(300):
                trainer.step()
