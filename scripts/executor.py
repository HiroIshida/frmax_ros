import time
from typing import List, Optional

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from rospy import Subscriber
from sensor_msgs.msg import PointCloud2
from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis, Box, PointCloudLink
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer
from utils import CoordinateTransform, chain_transform


class RobotInterfaceWrap(PR2ROSRobotInterface):
    def __init__(self, pr2: PR2):
        super().__init__(pr2)
        # offset_map = {
        #         "l_shoulder_lift_joint": -0.04,
        #         "l_elbow_flex_joint": -0.10,
        #         }
        offset_map = {
            "l_shoulder_lift_joint": 0.0,
            "l_elbow_flex_joint": 0.0,
        }
        self.offset_indices = [pr2.joint_names.index(name) for name in offset_map.keys()]
        self.offset_values = np.array(list(offset_map.values()))

    def angle_vector(self, av: Optional[np.ndarray] = None, **kwargs):
        if av is None:
            return super().angle_vector()
        av = av.copy()
        av[self.offset_indices] += self.offset_values
        super().angle_vector(av, **kwargs)

    def angle_vector_sequence(self, avs, **kwargs):
        avs = [av.copy() for av in avs]
        for av in avs:
            av[self.offset_indices] += self.offset_values
        super().angle_vector_sequence(avs, **kwargs)


class Executor:
    tf_obj_base: Optional[CoordinateTransform]
    pr2: PR2
    ri: PR2ROSRobotInterface
    debug_cloud: Optional[np.ndarray]

    def __init__(self):
        pr2 = PR2()
        pr2.reset_manip_pose()
        self.pr2 = pr2
        self.pr2.r_shoulder_pan_joint.joint_angle(-2.1)
        self.pr2.l_shoulder_pan_joint.joint_angle(2.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)
        self.ri = RobotInterfaceWrap(pr2)
        self.ri.move_gripper("larm", 0.05)
        self.ri.angle_vector(self.pr2.angle_vector())
        self.ri.wait_interpolation()
        time.sleep(2.0)

        self.sub = Subscriber("/object_pose", PoseStamped, self.callback)
        self.sub_cloud = Subscriber("/debug_cloud", PointCloud2, self.callback_cloud)
        self.tf_obj_base = None
        self.debug_cloud = None

    def callback(self, msg: PoseStamped):
        tf = CoordinateTransform.from_ros_pose(msg.pose)
        tf.src = "object"
        tf.dest = "base"
        self.tf_obj_base = tf

    def callback_cloud(self, msg: PointCloud2):
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        self.debug_cloud = points

    def plannable(self) -> bool:
        return self.tf_obj_base is not None

    def plan(
        self,
        planer_pose_traj: List[np.ndarray],
        hypo_error: Optional[np.ndarray] = None,
        rot: float = -np.pi * 0.5,
    ):
        assert self.plannable()
        assert self.tf_obj_base is not None
        if hypo_error is None:
            hypo_error = np.zeros(3)
        x_error, y_error, yaw_error = hypo_error

        def to_transform(x, y, yaw, rot) -> CoordinateTransform:
            tf_obj_hypo = CoordinateTransform.from_skrobot_coords(
                Coordinates([x_error, y_error, 0.0], [yaw_error, 0, 0.0]), src="object", dest="hypo"
            )
            tf_reach_hypo = CoordinateTransform.from_skrobot_coords(
                Coordinates([x, y, 0.0], [yaw, 0, rot]), src="reach", dest="hypo"
            )
            tf_reach_obj = chain_transform(
                tf_reach_hypo,  # reach -> hypo
                tf_obj_hypo.inverse(),  # hypo -> object
            )
            tf_reach_base = chain_transform(
                tf_reach_obj,  # reach -> object
                self.tf_obj_base,  # object -> base
            )
            return tf_reach_base

        coords_list = []
        for relative_pose in planer_pose_traj:
            tf_reach_base = to_transform(*relative_pose, rot)
            co_reach = tf_reach_base.to_skrobot_coords()
            coords_list.append(co_reach)

        # setup common stuff
        pr2_plan_conf = PR2Config(control_arm="larm")
        joint_names = pr2_plan_conf._get_control_joint_names()

        colkin = pr2_plan_conf.get_collision_kin()
        box = Box([0.88, 1.0, 0.1], pos=[0.6, 0.0, 0.66], with_sdf=True)
        colfree_const = CollFreeConst(colkin, box.sdf, self.pr2)

        box_const = pr2_plan_conf.get_box_const()
        q_init = get_robot_state(self.pr2, joint_names)

        efkin = pr2_plan_conf.get_endeffector_kin()
        co_reach_init = coords_list[0]

        def whole_plan() -> Optional[List[np.ndarray]]:
            # solve full plan to initial pose
            pose_const = PoseConstraint.from_skrobot_coords([co_reach_init], efkin, self.pr2)
            problem = Problem(q_init, box_const, pose_const, colfree_const, None)
            ompl_config = OMPLSolverConfig(n_max_call=2000, simplify=True)
            ompl_solver = OMPLSolver.init(ompl_config)
            ompl_solver.setup(problem)
            res = ompl_solver.solve()
            if res.traj is None:
                return None

            # solve ik for each pose
            q_list = list(res.traj.resample(8).numpy())
            for co_reach in coords_list[1:]:
                set_robot_state(self.pr2, joint_names, q_list[-1])
                pose_const = PoseConstraint.from_skrobot_coords([co_reach], efkin, self.pr2)
                pose_const.reflect_skrobot_model(self.pr2)
                satis_con = SatisfactionConfig(acceptable_error=1e-5, disp=False)
                ret = satisfy_by_optimization(
                    pose_const, box_const, None, q_list[-1], config=satis_con
                )
                if not ret.success:
                    return None
                # check collision free after ik (dont explicitly consider in ik)
                colfree_const.reflect_skrobot_model(self.pr2)
                if not colfree_const.is_valid(ret.q):
                    return None
                q_list.append(ret.q)
            return q_list

        for _ in range(5):
            q_list = whole_plan()
            if q_list is None:
                rospy.logwarn("failed to plan, retrying...")
                continue
            else:
                rospy.loginfo("solved!")
                break
        assert q_list is not None

        if True:
            print(len(q_list))
            viewer = TrimeshSceneViewer()
            if self.debug_cloud is not None:
                cloud = PointCloudLink(self.debug_cloud)
                viewer.add(cloud)
            axis = Axis.from_coords(co_reach)
            viewer.add(self.pr2)
            viewer.add(box)
            viewer.add(axis)
            viewer.show()
            for q in q_list:
                set_robot_state(self.pr2, joint_names, q)
                viewer.redraw()
                time.sleep(1.0)
            time.sleep(1000)

        # create full angle vector sequence
        avs = []
        for q in res.traj.resample(8):
            print(q)
            set_robot_state(self.pr2, joint_names, q)
            avs.append(self.pr2.angle_vector())

        times = [0.3 for _ in range(6)] + [0.6 for _ in range(2)]
        self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
        self.ri.wait_interpolation()

        self.pr2.larm.move_end_pos([0.08, 0.0, 0.0])
        self.ri.angle_vector(self.pr2.angle_vector(), time_scale=5.0, time=3.0)
        self.ri.wait_interpolation()
        self.ri.move_gripper("larm", 0.0)


if __name__ == "__main__":
    rospy.init_node("executor")

    executor = Executor()

    # wait until the object pose is received
    while not executor.plannable():
        time.sleep(0.1)
    rospy.loginfo("Object pose is received")
    traj = np.array(
        [
            [-0.06, -0.05, 0.0],
            [-0.05, -0.05, 0.0],
            [-0.04, -0.05, 0.0],
            [-0.03, -0.05, 0.0],
            [-0.02, -0.05, 0.0],
            [-0.01, -0.05, 0.0],
            [-0.00, -0.05, 0.0],
        ]
    )
    executor.plan(traj)
