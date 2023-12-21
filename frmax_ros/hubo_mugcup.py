from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import rospkg
import rospy
import trimesh
from geometry_msgs.msg import PoseStamped
from movement_primitives.dmp import DMP
from nav_msgs.msg import Path as RosPath
from rospy import Publisher
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
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.joint import RotationalJoint
from skrobot.model.primitives import Axis, Box, MeshLink
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType, RotationType
from utils import CoordinateTransform

from frmax_ros.node import ObjectPoseProvider
from frmax_ros.utils import CoordinateTransform


class ExecutorBase:  # TODO: move later to task-agonistic module
    pose_provider: ObjectPoseProvider
    pr2: PR2
    ri: PR2ROSRobotInterface
    pub_grasp_path: Publisher

    def __init__(self):
        rospy.init_node("robot_interface", disable_signals=True, anonymous=True)
        self.pr2 = PR2()
        self._confine_infinite_rotation(self.pr2, ["caster"])
        self.ri = PR2ROSRobotInterface(self.pr2)
        self.initialize_robot()
        self.pub_grasp_path = rospy.Publisher("/grasp_path", RosPath, queue_size=1, latch=True)
        self.pose_provider = ObjectPoseProvider()

    @staticmethod
    def _confine_infinite_rotation(pr2: PR2, filter_words: List[str]) -> None:
        # infinite rotation causes problem due to adhoc interpolation by https://github.com/iory/scikit-robot/pull/244
        # thus we confine the range of rotation
        for joint in pr2.joint_list:
            if any([word in joint.name for word in filter_words]):
                continue
            if isinstance(joint, RotationalJoint):
                if np.isinf(joint.min_angle):
                    joint.min_angle = -2 * np.pi
                    rospy.loginfo(f"clamp min angle of {joint.name} from -inf to {joint.min_angle}")
                    print("clamp min angle")
                if np.isinf(joint.max_angle):
                    joint.max_angle = +2 * np.pi
                    rospy.loginfo(f"clamp max angle of {joint.name} from +inf to {joint.max_angle}")

    @abstractmethod
    def initialize_robot(self):
        pass

    @abstractmethod
    def rollout(self, planer_traj: "GraspingPlanerTrajectory", error: np.ndarray) -> Optional[bool]:
        pass

    def execute(self, q_traj: List[np.ndarray], times: List[float], arm: Literal["larm", "rarm"]):
        conf = PR2Config(control_arm=arm)
        joint_names = conf.get_control_joint_names()
        av_list = []
        q_prev = q_traj[0]
        for q in q_traj:
            set_robot_state(self.pr2, joint_names, q)
            av_list.append(self.pr2.angle_vector())
            q_diff = q - q_prev
            is_huge_diff = np.any(np.abs(q_diff) > np.pi)
            if is_huge_diff:
                rospy.logerr("huge diff")
                assert False
            q_prev = q

        rospy.loginfo("follow trajectory")
        self.ri.angle_vector_sequence(av_list, times=times, time_scale=1.0)
        self.ri.wait_interpolation()
        rospy.loginfo("finish sending")


class PathPlanner:
    table: Box
    dummy_obstacle: Box
    target_object: MeshLink
    pr2: PR2

    def __init__(self, pr2: PR2):
        rospack = rospkg.RosPack()
        pkg_path = Path(rospack.get_path("frmax_ros"))
        mug_model_path = pkg_path / "model" / "hubolab_mug.stl"
        mesh = trimesh.load_mesh(mug_model_path)
        mugcup = MeshLink(mesh, with_sdf=True)
        self.target_object = mugcup

        h = 0.72
        table = Box(extents=[0.5, 0.9, h], with_sdf=True)
        table.translate([0.5, 0, 0.5 * h])
        self.table = table

        dummy_obstacle = Box([0.45, 0.6, 0.03], pos=[0.5, 0.0, 1.2], with_sdf=True)
        self.dummy_obstacle = dummy_obstacle
        self.pr2 = pr2

    def _setup_constraints(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
        co_object: Optional[Coordinates] = None,
        consider_table: bool = True,
        consider_dummy: bool = True,
        consider_object: bool = True,
        base_type: BaseType = BaseType.FIXED,
    ) -> Tuple[AbstractEqConst, CollFreeConst, BoxConst]:
        if consider_object:
            assert co_object is not None
            self.target_object.newcoords(co_object)

        sdfs = []
        if consider_table:
            sdfs.append(self.table.sdf)
        if consider_dummy:
            sdfs.append(self.dummy_obstacle.sdf)
        if consider_object:
            sdfs.append(self.target_object.sdf)
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
            co_object,
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

    def plan_path(
        self,
        target: Union[Coordinates, np.ndarray],
        arm: Literal["larm", "rarm"],
        co_object: Optional[Coordinates] = None,
        consider_dummy: bool = True,
        consider_object: bool = True,
        display_planning_scene: bool = False,
    ) -> Optional[Trajectory]:

        eq_const, ineq_const, box_const = self._setup_constraints(
            target, arm, co_object, consider_dummy, consider_object
        )
        joint_names = PR2Config(control_arm=arm).get_control_joint_names()
        q_start = get_robot_state(self.pr2, joint_names)
        problem = Problem(q_start, box_const, eq_const, ineq_const, None)

        ompl_config = OMPLSolverConfig(n_max_call=5000, simplify=True, n_max_satisfaction_trial=20)
        ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
        ompl_solver.setup(problem)

        if display_planning_scene:
            viewer = TrimeshSceneViewer()
            viewer.add(self.table)
            viewer.add(self.dummy_obstacle)
            viewer.add(self.target_object)
            viewer.add(self.pr2)
            if isinstance(target, Coordinates):
                axis = Axis.from_coords(target)
                viewer.add(axis)
            viewer.show()

        ret = ompl_solver.solve()
        return ret.traj


class GraspingPlanerTrajectory:
    seq_tf_ef_to_nominal: List[CoordinateTransform]

    def __init__(self, param: np.ndarray):
        assert param.shape == (3 * 6 + 3,)
        n_split = 100
        start = np.array([-0.06, -0.04, 0.0])
        goal = np.array([-0.0, -0.04, 0.0])
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


class MugcupGraspExecutor(ExecutorBase):
    path_planner: PathPlanner
    tf_object_to_april: CoordinateTransform
    pregrasp_gripper_pos: ClassVar[float] = 0.05

    def __init__(self):
        super().__init__()
        self.path_planner = PathPlanner(self.pr2)
        self.tf_object_to_april = CoordinateTransform(np.array([0.013, -0.004, -0.095]), np.eye(3))

    def initialize_robot(self):
        self.pr2.reset_manip_pose()
        self.pr2.r_shoulder_pan_joint.joint_angle(-1.1)
        self.pr2.l_shoulder_pan_joint.joint_angle(1.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)
        self.pr2.head_tilt_joint.joint_angle(+1.2)
        self.ri.angle_vector(self.pr2.angle_vector())
        self.ri.move_gripper("larm", self.pregrasp_gripper_pos)

    def get_tf_object_to_base(self) -> CoordinateTransform:
        tf_april_to_base = self.pose_provider.get_tf_object_to_base()
        tf_object_to_april = CoordinateTransform(
            np.array([0.013, -0.004, -0.095]), np.eye(3), "object", "april"
        )
        return tf_object_to_april * tf_april_to_base

    def rollout(self, planer_traj: GraspingPlanerTrajectory, error: np.ndarray) -> Optional[bool]:
        tf_object_to_base = self.get_tf_object_to_base()
        tf_ef_to_base_seq = planer_traj.instantiate(tf_object_to_base, error)
        path_msg = planer_traj.get_path_msg(tf_object_to_base, error)
        self.pub_grasp_path.publish(path_msg)
        rospy.loginfo("Published grasp path")

        # if start position is in collision, consider execution fails
        co_init = tf_ef_to_base_seq[0].to_skrobot_coords()
        co_object = tf_object_to_base.to_skrobot_coords()
        if not self.path_planner.is_feasible_target(co_init, "larm", co_object=co_object):
            rospy.loginfo("the object will be in collision")
            return False

        # plan reaching
        q_traj_reaching = self.path_planner.plan_path(co_init, "larm", co_object=co_object)
        if q_traj_reaching is None:
            rospy.loginfo("Failed to plan reaching")
            return None  # None means we cannot evaluate the

        # plan grasp trajectory
        joint_names = PR2Config(control_arm="larm").get_control_joint_names()
        set_robot_state(self.pr2, joint_names, q_traj_reaching[-1])
        q_list = []
        for tf_ef_to_base in tf_ef_to_base_seq:
            res = self.pr2.larm.inverse_kinematics(
                tf_ef_to_base.to_skrobot_coords(),
                link_list=self.pr2.larm.link_list,
                move_target=self.pr2.larm_end_coords,
            )
            if isinstance(res, bool) and res == False:
                rospy.loginfo("Failed to plan IK")
                return None
            q = get_robot_state(self.pr2, joint_names)
            q_list.append(q)
        q_traj_grasping = np.array(q_list)

        # now execute reaching and grasping
        times_reaching = [0.3] * 20
        self.execute(q_traj_reaching.resample(20).numpy(), times_reaching, "larm")
        times_grasping = [0.5] * len(q_traj_grasping)
        self.execute(q_traj_grasping, times_grasping, "larm")

        # back to initial pose
        self.execute(q_traj_grasping[::-1], [1.0] * len(q_traj_grasping), "larm")
        # self.execute(q_traj_reaching[::-1].resample(20).numpy(), times_reaching, "larm")
        return True


if __name__ == "__main__":
    e = MugcupGraspExecutor()
    traj = GraspingPlanerTrajectory(np.zeros(3 * 6 + 3))
    e.rollout(traj, np.zeros(3))
    rospy.spin()
