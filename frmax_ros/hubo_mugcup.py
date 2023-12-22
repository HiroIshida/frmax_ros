import time
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import rospkg
import rospy
import trimesh
from frmax2.core import DGSamplerConfig
from geometry_msgs.msg import PoseStamped
from movement_primitives.dmp import DMP
from nav_msgs.msg import Path as RosPath
from skmp.constraint import (
    AbstractEqConst,
    BoxConst,
    CollFreeConst,
    ConfigPointConst,
    PoseConstraint,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import MeshLink
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from tinyfk import BaseType, RotationType
from utils import CoordinateTransform

from frmax_ros.rollout import (
    AutomaticTrainerBase,
    PlanningScene,
    RolloutAbortedException,
    RolloutExecutorBase,
)
from frmax_ros.utils import CoordinateTransform


class PathPlanner:
    pr2: PR2
    scene: PlanningScene

    def __init__(self, pr2: PR2, scene: PlanningScene, visualize: bool = True):
        self.pr2 = pr2
        self.scene = scene

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
    ) -> Tuple[AbstractEqConst, CollFreeConst, BoxConst]:
        if consider_object:
            assert co_object is not None

        sdfs = []
        if consider_table:
            sdfs.append(self.scene.table.sdf)
        if consider_dummy:
            rospy.loginfo("dummy object is considered")
            sdfs.append(self.scene.dummy_obstacle.sdf)
        if consider_object:
            sdfs.append(self.scene.target_object.sdf)
        usdf = UnionSDF(sdfs)

        plan_conf = PR2Config(control_arm=arm, base_type=base_type)
        colkin = plan_conf.get_collision_kin()
        ineq_const = CollFreeConst(colkin, usdf, self.pr2, only_closest_feature=True)

        plan_conf.get_control_joint_names()

        efkin = plan_conf.get_endeffector_kin(rot_type=RotationType.XYZW)

        if isinstance(target, Coordinates):
            eq_const = PoseConstraint.from_skrobot_coords([target], efkin, self.pr2)
        else:
            eq_const = ConfigPointConst(target)

        box_const = plan_conf.get_box_const()
        box_const.lb -= 1e-4
        box_const.ub += 1e-4

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
        return ineq_const.is_valid(ret.q)

    def solve_ik_skrobot(
        self,
        co: Coordinates,
        arm: Literal["larm", "rarm"],
        q_seed: np.ndarray,
        co_object: Optional[Coordinates] = None,
        consider_table: bool = True,
        consider_dummy: bool = True,
        consider_object: bool = True,
    ) -> Optional[np.ndarray]:
        eq_const, ineq_const, box_const = self._setup_constraints(
            co,
            arm,
            co_object=co_object,
            consider_table=consider_table,
            consider_dummy=consider_dummy,
            consider_object=consider_object,
        )
        arm_robot = self.pr2.rarm if arm == "rarm" else self.pr2.larm
        move_target = self.pr2.rarm_end_coords if arm == "rarm" else self.pr2.larm_end_coords
        ret = arm_robot.inverse_kinematics(co, move_target=move_target, seed=q_seed)
        if isinstance(ret, bool) and ret == False:
            rospy.loginfo("callision agnonistic IK failed")
            return None
        assert isinstance(ret, np.ndarray)
        print(ineq_const.evaluate_single(ret, with_jacobian=False))
        if not ineq_const.is_valid(ret):
            rospy.loginfo("solved Ik but in collision")
            return None
        return ret

    def plan_path(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
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
        q_start = get_robot_state(self.pr2, joint_names)
        problem = Problem(q_start, box_const, eq_const, ineq_const, None)

        ompl_config = OMPLSolverConfig(n_max_call=5000, simplify=True, n_max_satisfaction_trial=20)
        ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
        ompl_solver.setup(problem)

        ret = ompl_solver.solve()
        return ret.traj


class GraspingPlanerTrajectory:
    seq_tf_ef_to_nominal: List[CoordinateTransform]
    is_valid: bool
    pregrasp_gripper_pos: ClassVar[float] = 0.03

    def __init__(self, param: np.ndarray):
        assert param.shape == (3 * 6 + 3,)
        n_split = 100
        start = np.array([-0.06, -0.045, 0.0])
        goal = np.array([-0.005, -0.045, 0.0])
        diff_step = (goal - start) / (n_split - 1)
        traj_default = np.array([start + diff_step * i for i in range(n_split)])
        n_weights_per_dim = 6
        dmp = DMP(3, execution_time=1.0, n_weights_per_dim=n_weights_per_dim, dt=0.1)
        dmp.imitate(np.linspace(0, 1, n_split), traj_default.copy())
        dmp.configure(start_y=traj_default[0])

        xytheta_scaling = np.array([0.01, 0.01, np.deg2rad(20.0)])
        goal_position_scaling = xytheta_scaling * 1.0

        xytheta_scaling = np.array([0.03, 0.03, np.deg2rad(20.0)])
        force_scalineg = xytheta_scaling * 300
        n_dim = 3
        n_goal_dim = 3
        W = param[:-n_goal_dim].reshape(n_dim, -1)
        dmp.forcing_term.weights_[:, :] += W[:, :] * force_scalineg[:, None]
        goal_param = param[-n_goal_dim:] * goal_position_scaling
        dmp.goal_y += goal_param
        _, planer_traj = dmp.open_loop()

        is_valid = True
        if np.any(np.abs(planer_traj[:, 2]) > np.deg2rad(45.0)):
            is_valid = False
        if np.abs(planer_traj[-1, 0] - planer_traj[0, 0]) > 0.1:
            is_valid = False
        self.is_valid = is_valid

        height = 0.08
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


class MugcupGraspRolloutExecutor(RolloutExecutorBase):
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
        self.tf_object_to_april = CoordinateTransform(np.array([0.013, -0.004, -0.095]), np.eye(3))

    def initialize_robot(self):
        self.pr2.reset_manip_pose()
        self.pr2.r_shoulder_pan_joint.joint_angle(-1.1)
        self.pr2.l_shoulder_pan_joint.joint_angle(1.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)
        self.pr2.head_tilt_joint.joint_angle(+1.2)
        self.ri.angle_vector(self.pr2.angle_vector(), time_scale=5.0)
        self.ri.move_gripper("larm", self.pregrasp_gripper_pos)
        self.ri.move_gripper("rarm", self.pregrasp_gripper_pos)

    def get_tf_object_to_base(self) -> CoordinateTransform:
        self.pose_provider.reset()
        tf_april_to_base = self.pose_provider.get_tf_object_to_base()
        tf_object_to_april = CoordinateTransform(
            np.array([0.013, -0.004, -0.095]), np.eye(3), "object", "april"
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
        return 21

    def rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        assert param.shape == (self.get_policy_dof(),)
        planer_traj = GraspingPlanerTrajectory(param)
        try:
            tf_object_to_base = self.get_tf_object_to_base()
        except TimeoutError:
            reason = "failed to get object pose"
            raise RolloutAbortedException(reason)
        self.scene.update(tf_object_to_base.to_skrobot_coords())

        x_pos, y_pos = tf_object_to_base.trans[:2]
        if x_pos > 0.6 or y_pos > 0.2:
            reason = "invalid object position"
            raise RolloutAbortedException(reason)

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
        q_traj_reaching = self.path_planner.plan_path(co_init, "larm", co_object=co_object)
        if q_traj_reaching is None:
            reason = "Failed to plan reaching"
            raise RolloutAbortedException(reason)

        # now execute reaching and grasping
        times_reaching = [0.3] * 7 + [0.6] * 2 + [1.0]
        q_traj_reaching = q_traj_reaching.resample(10).numpy()
        self.send_command_to_real_robot(q_traj_reaching, times_reaching, "larm")

        # calibrate
        self.offset_prover.reset()
        time.sleep(2.0)
        tf_efcalib_to_ef = self.offset_prover.get_cloudtape_to_tape().inverse()
        tf_efcalib_to_ef.src = "efcalib"
        tf_efcalib_to_ef.dest = "ef"
        tf_efcalib_to_base_seq = [
            tf_efcalib_to_ef * tf_ef_to_base for tf_ef_to_base in tf_ef_to_base_seq
        ]

        # move to pregrasp
        self.ri.move_gripper("larm", planer_traj.pregrasp_gripper_pos)

        # plan grasping using the calibrated pose
        set_robot_state(self.pr2, joint_names, q_traj_reaching[-1])
        q_list = []
        for tf_ef_to_base in tf_efcalib_to_base_seq:
            res = self.pr2.larm.inverse_kinematics(
                tf_ef_to_base.to_skrobot_coords(),
                link_list=self.pr2.larm.link_list,
                move_target=self.pr2.larm_end_coords,
            )
            if isinstance(res, bool) and res == False:
                reason = "IK failed"
                raise RolloutAbortedException(reason)

            q = get_robot_state(self.pr2, joint_names)
            q_list.append(q)
        q_traj_grasping = np.array(q_list)
        set_robot_state(self.pr2, joint_names, q_traj_grasping[-1])

        times_grasping = [0.5] * len(q_traj_grasping)
        self.send_command_to_real_robot(q_traj_grasping, times_grasping, "larm")
        self.ri.move_gripper("larm", 0.0, effort=100)

        # shake back and force
        av_now = self.pr2.angle_vector()
        self.pr2.larm.move_end_pos([-0.03, 0.0, 0.0])
        self.ri.angle_vector(self.pr2.angle_vector(), time_scale=1.0, time=1.0)
        self.ri.wait_interpolation()
        self.ri.angle_vector(av_now, time_scale=1.0, time=1.0)
        self.ri.wait_interpolation()
        time.sleep(0.5)
        annot = self.get_auto_annotation()
        if annot is None:
            reason = "ambiguous annotation"
            raise RolloutAbortedException(reason)
        self.ri.move_gripper("larm", self.pregrasp_gripper_pos)

        # back to initial pose
        self.send_command_to_real_robot(q_traj_grasping[::-1], [0.5] * len(q_traj_grasping), "larm")
        self.send_command_to_real_robot(q_traj_reaching[::-1], times_reaching[::-1], "larm")
        return annot


class MugcupGraspTrainer(AutomaticTrainerBase):
    b_min: np.ndarray
    b_max: np.ndarray

    def __init__(self):
        ls_param = np.ones(21)
        ls_err = np.array([0.005, 0.005, np.deg2rad(5.0)])
        self.b_min = np.array([-0.03, -0.03, -np.pi * 0.2])
        self.b_max = np.array([0.03, 0.03, np.pi * 0.2])
        config = DGSamplerConfig(
            param_ls_reduction_rate=0.999,
            n_mc_param_search=30,
            c_svm=10000,
            integration_method="mc",
            n_mc_integral=1000,
            r_exploration=0.5,
            learning_rate=1.0,
        )
        super().__init__(ls_param, ls_err, config, "mugcup", n_init_sample=10)

    @staticmethod
    def get_rollout_executor() -> RolloutExecutorBase:
        return MugcupGraspRolloutExecutor()

    def sample_situation(self) -> np.ndarray:
        return np.random.uniform(self.b_min, self.b_max)

    @staticmethod
    def is_valid_param(param: np.ndarray) -> bool:
        traj = GraspingPlanerTrajectory(param)
        return traj.is_valid


if __name__ == "__main__":
    # e = MugcupGraspRolloutExecutor()
    # e.rollout(np.zeros(21), np.zeros(3))
    # rospy.spin()

    trainer = MugcupGraspTrainer()
    trainer.next()
