import os
import sys
import time
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import rospy
from frmax2.core import (
    BlackBoxSampler,
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
)
from geometry_msgs.msg import PoseStamped
from movement_primitives.dmp import DMP
from rospy import Subscriber
from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis, Box, Cylinder
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from utils import CoordinateTransform, chain_transform


@contextmanager
def suppress_output(stream_name):
    if stream_name not in ["stdout", "stderr"]:
        raise ValueError("Invalid stream name. Use 'stdout' or 'stderr'.")

    # Select the appropriate stream
    stream = sys.stdout if stream_name == "stdout" else sys.stderr
    fd = stream.fileno()

    def _redirect_stream(to):
        stream.close()  # Close and flush the stream
        os.dup2(to.fileno(), fd)  # Redirect fd to the 'to' file
        if stream_name == "stdout":
            sys.stdout = os.fdopen(fd, "w")  # Reopen stdout for Python
        else:
            sys.stderr = os.fdopen(fd, "w")  # Reopen stderr for Python

    # Duplicate the file descriptor
    with os.fdopen(os.dup(fd), "w") as old_stream:
        with open(os.devnull, "w") as file:
            _redirect_stream(to=file)
        try:
            yield  # Allow code to run with redirected stream
        finally:
            _redirect_stream(to=old_stream)  # Restore the original stream


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
    is_simulation: bool

    def __init__(self, debug_pose_msg: Optional[PoseStamped]):
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.reset_manip_pose()
        self.pr2 = pr2
        self.pr2.r_shoulder_pan_joint.joint_angle(-2.1)
        self.pr2.l_shoulder_pan_joint.joint_angle(2.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)

        self.is_simulation = debug_pose_msg is not None
        if self.is_simulation:
            tf = CoordinateTransform.from_ros_pose(debug_pose_msg.pose)
            tf.src = "object"
            tf.dest = "base"
            self.tf_obj_base = tf
        else:
            self.ri = RobotInterfaceWrap(pr2)
            self.ri.move_gripper("larm", 0.05)
            self.ri.angle_vector(self.pr2.angle_vector())
            self.ri.wait_interpolation()
            time.sleep(2.0)

            self.sub = Subscriber("/object_pose", PoseStamped, self.callback)
            self.tf_obj_base = None

    def callback(self, msg: PoseStamped):
        tf = CoordinateTransform.from_ros_pose(msg.pose)
        tf.src = "object"
        tf.dest = "base"
        self.tf_obj_base = tf

    def plannable(self) -> bool:
        return self.tf_obj_base is not None

    def plan(
        self,
        planer_pose_traj: List[np.ndarray],
        hypo_error: Optional[np.ndarray] = None,
        rot: float = -np.pi * 0.5,
    ) -> Optional[bool]:
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
        table = Box([0.88, 1.0, 0.1], pos=[0.6, 0.0, 0.66], with_sdf=True)
        magcup = Cylinder(0.05, 0.12, with_sdf=True)
        magcup.visual_mesh.visual.face_colors = [255, 0, 0, 150]  # type: ignore
        magcup.newcoords(self.tf_obj_base.to_skrobot_coords())
        magcup.translate([0, 0, -0.03])
        sdf_all = UnionSDF([table.sdf, magcup.sdf])
        colfree_const_all = CollFreeConst(colkin, sdf_all, self.pr2)
        colfree_const_table = CollFreeConst(colkin, table.sdf, self.pr2)

        box_const = pr2_plan_conf.get_box_const()
        q_init = get_robot_state(self.pr2, joint_names)

        efkin = pr2_plan_conf.get_endeffector_kin()
        co_reach_init = coords_list[0]

        def whole_plan() -> Optional[List[np.ndarray]]:
            # solve full plan to initial pose
            pose_const = PoseConstraint.from_skrobot_coords([co_reach_init], efkin, self.pr2)
            problem = Problem(q_init, box_const, pose_const, colfree_const_all, None)
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
                colfree_const_table.reflect_skrobot_model(self.pr2)
                if not colfree_const_table.is_valid(ret.q):
                    return None
                q_list.append(ret.q)
            return q_list

        for _ in range(10):
            q_list = whole_plan()
            if q_list is None:
                rospy.logwarn("failed to plan, retrying...")
                continue
            else:
                rospy.loginfo("solved!")
                break
        if q_list is None:
            rospy.logwarn("failed to plan")
            return None

        if True:
            viewer = TrimeshSceneViewer()
            axis = Axis.from_coords(co_reach)
            viewer.add(self.pr2)
            viewer.add(table)
            viewer.add(magcup)
            viewer.add(axis)
            viewer.show()
            for q in q_list:
                set_robot_state(self.pr2, joint_names, q)
                viewer.redraw()
                time.sleep(1.0)
            time.sleep(1000)

        if not self.is_simulation:
            # create full angle vector sequence
            avs = []
            for q in q_list:
                set_robot_state(self.pr2, joint_names, q)
                avs.append(self.pr2.angle_vector())

            times = (
                [0.3 for _ in range(6)]
                + [0.6 for _ in range(2)]
                + [0.5 for _ in range(len(planer_pose_traj) - 1)]
            )
            assert len(times) == len(avs)
            self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
            self.ri.wait_interpolation()
            self.ri.move_gripper("larm", 0.0)

            def wait_for_label() -> bool:
                while True:
                    user_input = input("Add label: Enter 'y' for True or 'n' for False: ")
                    if user_input.lower() == "y":
                        return True
                    elif user_input.lower() == "n":
                        return False

            label = wait_for_label()
            self.ri.move_gripper("larm", 0.05)
            self.ri.angle_vector_sequence(avs[::-1], times=[0.3] * len(avs), time_scale=1.0)
            self.ri.wait_interpolation()
            return label


def create_trajectory(param: np.ndarray, dt: float = 0.1) -> List[np.ndarray]:
    assert param.shape == (3 * 6 + 3,)
    n_split = 100
    start = np.array([-0.06, -0.045, 0.0])
    goal = np.array([-0.0, -0.045, 0.0])
    diff_step = (goal - start) / (n_split - 1)
    traj_default = np.array([start + diff_step * i for i in range(n_split)])
    n_weights_per_dim = 6
    dmp = DMP(3, execution_time=1.0, n_weights_per_dim=n_weights_per_dim, dt=dt)
    dmp.imitate(np.linspace(0, 1, n_split), traj_default.copy())
    dmp.configure(start_y=traj_default[0])

    # set param
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
    return planer_traj


if __name__ == "__main__":
    debug_pose_msg = PoseStamped()
    debug_pose_msg.pose.position.x = 0.47641722876585824
    debug_pose_msg.pose.position.y = -0.054688228244401484
    debug_pose_msg.pose.position.z = 0.8
    debug_pose_msg.pose.orientation.x = 0.0
    debug_pose_msg.pose.orientation.y = 0.0
    debug_pose_msg.pose.orientation.z = -0.6325926153678488
    debug_pose_msg.pose.orientation.w = 0.7744847209481055

    executor = Executor(debug_pose_msg)

    debug: bool = False
    if debug:
        param = np.random.randn(21) * 0.0
        planer_traj = create_trajectory(param)

        # wait until the object pose is received
        while not executor.plannable():
            time.sleep(0.1)
        rospy.loginfo("Object pose is received")
        executor.plan(planer_traj)
    else:

        def sample_situation() -> np.ndarray:
            x = np.random.uniform(-0.03, 0.03)
            y = np.random.uniform(-0.03, 0.03)
            yaw = np.random.uniform(-np.pi * 0.2, np.pi * 0.2)
            return np.array([x, y, yaw])

        param_init = np.zeros(21)
        traj = create_trajectory(param_init)
        # create initial dataset
        n_init_sample = 10
        X, Y = []
        for _ in range(n_init_sample):
            error = sample_situation()
            is_success = executor.plan(traj, hypo_error=error)
            X.append(np.hstack([param_init, error]))
            Y.append(is_success)

        X = np.array(X)
        Y = np.array(Y)
        ls_param = np.ones(21)
        ls_err = np.array([0.01, 0.01, np.deg2rad(5.0)])
        metric = CompositeMetric.from_ls_list([ls_param, ls_err])

        config = DGSamplerConfig(
            param_ls_reduction_rate=0.999,
            n_mc_param_search=30,
            c_svm=10000,
            integration_method="mc",
            n_mc_integral=1000,
            r_exploration=0.5,
            learning_rate=1.0,
        )
        sampler: BlackBoxSampler = DistributionGuidedSampler(
            X, Y, metric, param_init, config, situation_sampler=sample_situation
        )
        for _ in range(1000):
            sampler.update_center()
            x = sampler.ask()
            assert x is not None
            param, error = x[:-3], x[-3:]
            while True:
                y = executor.plan(traj, hypo_error=error)
                if y is not None:
                    break
            sampler.tell(x, y)
