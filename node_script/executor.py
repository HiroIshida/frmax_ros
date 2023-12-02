import argparse
import copy
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import dill
import numpy as np
import rospkg
import rospy
from frmax2.core import (
    BlackBoxSampler,
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
)
from geometry_msgs.msg import PoseStamped
from movement_primitives.dmp import DMP
from nav_msgs.msg import Path as RosPath
from rospy import Publisher, Subscriber
from skmp.constraint import BoxConst, CollFreeConst, ConfigPointConst, PoseConstraint
from skmp.kinematics import ArticulatedEndEffectorKinematicsMap
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig, satisfy_by_optimization
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis, Box, Cylinder
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from sound_play.libsoundplay import SoundClient
from std_msgs.msg import Header
from tinyfk import RotationType
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


class SoundClientWrap(SoundClient):
    def __init__(self, always_local: bool = False):
        super().__init__()
        self.always_local = always_local

    def say(self, message: str, blocking: bool = False, local: bool = False):
        if local or self.always_local:
            subprocess.call('echo "{}" | festival --tts'.format(message), shell=True)
        rospy.logdebug("sound client: {}".format(message))
        super().say(message, volume=0.2, blocking=blocking)


class RobotInterfaceWrap(PR2ROSRobotInterface):
    """A safer version of PR2ROSRobotInterface"""

    pr2: PR2

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
        self.pr2 = pr2

    def angle_vector(self, av: Optional[np.ndarray] = None, **kwargs):
        rospy.logdebug("angle_vector: {}".format(av))
        if av is None:
            return super().angle_vector()
        if np.any(np.isinf(av)) or np.any(np.isnan(av)):
            raise ValueError("angle vector contains inf or nan")

        # reflect
        self.pr2.angle_vector(av)

        av = av.copy()
        av[self.offset_indices] += self.offset_values
        super().angle_vector(av, **kwargs)

    def angle_vector_sequence(self, avs, **kwargs):
        # reflect
        self.pr2.angle_vector(avs[-1])
        avs = [av.copy() for av in avs]
        rospy.logdebug("avs[0]: {}".format(avs[0]))
        rospy.logdebug("avs[-1]: {}".format(avs[-1]))
        for av in avs:
            av[self.offset_indices] += self.offset_values
            if np.any(np.isinf(av)) or np.any(np.isnan(av)):
                raise ValueError("angle vector contains inf or nan")
        super().angle_vector_sequence(avs, **kwargs)


class PlanningCongig:
    joint_names: List[str]
    colfree_const_all: CollFreeConst
    colfree_const_table: CollFreeConst
    colfree_const_magcup: CollFreeConst
    box_const: BoxConst
    efkin: ArticulatedEndEffectorKinematicsMap
    table: Box
    magcup: Cylinder

    def __init__(
        self,
        tf_obj_base: CoordinateTransform,
        pr2: PR2,
        consider_dummy_obstacle: bool = True,
        arm="larm",
    ):
        pr2_plan_conf = PR2Config(control_arm=arm)
        joint_names = pr2_plan_conf._get_control_joint_names()

        colkin = pr2_plan_conf.get_collision_kin()
        table = Box([0.88, 1.0, 0.1], pos=[0.6, 0.0, 0.66], with_sdf=True)
        dummy_obstacle = Box([0.45, 0.6, 0.03], pos=[0.5, 0.0, 1.2], with_sdf=True)
        dummy_obstacle.visual_mesh.visual.face_colors = [255, 255, 255, 150]  # type: ignore
        if consider_dummy_obstacle:
            table_sdfs = [table.sdf, dummy_obstacle.sdf]
        else:
            table_sdfs = [table.sdf]

        magcup = Cylinder(0.0525, 0.12, with_sdf=True)
        magcup.visual_mesh.visual.face_colors = [255, 0, 0, 150]  # type: ignore
        magcup.newcoords(tf_obj_base.to_skrobot_coords())
        magcup.translate([0, 0, -0.03])

        colfree_const_all = CollFreeConst(colkin, UnionSDF(table_sdfs + [magcup.sdf]), pr2)
        colfree_const_table = CollFreeConst(colkin, UnionSDF(table_sdfs), pr2)
        colfree_const_magcup = CollFreeConst(colkin, magcup.sdf, pr2)

        box_const = pr2_plan_conf.get_box_const()
        efkin = pr2_plan_conf.get_endeffector_kin(rot_type=RotationType.XYZW)

        self.joint_names = joint_names
        self.colfree_const_all = colfree_const_all
        self.colfree_const_table = colfree_const_table
        self.colfree_const_magcup = colfree_const_magcup
        self.box_const = box_const
        self.efkin = efkin
        self.table = table
        self.magcup = magcup


class Executor:
    tf_obj_base: Optional[CoordinateTransform]
    raw_msg: Optional[PoseStamped]
    pr2: PR2
    ri: PR2ROSRobotInterface
    is_simulation: bool
    q_home: np.ndarray
    av_home: np.ndarray
    auto_annotation: bool
    sound_client: SoundClientWrap

    def __init__(self, debug_pose_msg: Optional[PoseStamped], auto_annotation: bool = True):
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.reset_manip_pose()
        self.pr2 = pr2
        self.pr2.r_shoulder_pan_joint.joint_angle(-1.1)
        self.pr2.l_shoulder_pan_joint.joint_angle(1.1)
        self.pr2.r_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_shoulder_lift_joint.joint_angle(-0.5)
        self.pr2.l_wrist_roll_joint.joint_angle(0.0)
        self.pr2.head_tilt_joint.joint_angle(+1.2)
        pr2_plan_conf = PR2Config(control_arm="larm")
        joint_names = pr2_plan_conf._get_control_joint_names()
        self.q_home = get_robot_state(self.pr2, joint_names)
        self.auto_annotation = auto_annotation
        self.av_home = self.pr2.angle_vector()

        self.is_simulation = debug_pose_msg is not None
        print("is_simulation: {}".format(self.is_simulation))
        if self.is_simulation:
            tf = CoordinateTransform.from_ros_pose(debug_pose_msg.pose)
            tf.src = "object"
            tf.dest = "base"
            self.tf_obj_base = tf
            self.raw_msg = debug_pose_msg
        else:
            self.sound_client = SoundClientWrap(always_local=False)
            self.ri = RobotInterfaceWrap(pr2)
            self.ri.move_gripper("larm", 0.05)
            self.ri.move_gripper("rarm", 0.03)
            self.ri.angle_vector(self.pr2.angle_vector())
            self.ri.wait_interpolation()
            time.sleep(2.0)
            rospy.logdebug("q_home: {}".format(self.q_home))
            rospy.logdebug("av_home: {}".format(self.av_home))

            self.pub = Publisher("/debug_trajectory", RosPath, queue_size=1, latch=True)
            self.sub = Subscriber("/object_pose_filtered", PoseStamped, self.callback)
            self.tf_obj_base = None
            self.sound_client.say("ready to start", local=True)

    def callback(self, msg: PoseStamped):
        self.raw_msg = copy.deepcopy(msg)
        msg.pose.position.z = 0.8  # for cup
        tf = CoordinateTransform.from_ros_pose(msg.pose)
        tf.src = "object"
        tf.dest = "base"
        self.tf_obj_base = tf

    def msg_available(self) -> bool:
        return self.tf_obj_base is not None

    def reset(self) -> None:
        if self.is_simulation:
            return
        self.tf_obj_base = None

    def wait_for_label(self) -> Optional[bool]:
        def get_humann_annotation() -> Optional[bool]:
            while True:
                user_input = input("Add label: Enter 'y' for True or 'n' for False, r for retry")
                if user_input.lower() == "y":
                    return True
                elif user_input.lower() == "n":
                    return False
                elif user_input.lower() == "r":
                    return None

        if self.auto_annotation:
            gripper_pos = self.ri.gripper_states["larm"].process_value
            if gripper_pos > 0.004:
                return True
            elif gripper_pos < 0.002:
                return False
            else:
                self.sound_client.say("uncertain. please label manually", local=True)
                # self.sound_client.say("aaaaaaaaaaaaaaaaaaaaaaaaa", local=True)
                get_humann_annotation()
        else:
            return get_humann_annotation()

    @lru_cache(maxsize=1)
    def recovery_ik_lib(self) -> List[np.ndarray]:
        pr2 = PR2()
        arm = pr2.rarm
        end_coords = pr2.rarm_end_coords
        lib = []
        while len(lib) < 30:
            pr2.reset_manip_pose()
            x = np.random.uniform(0.3, 0.6)
            y = np.random.uniform(-0.5, 0.5)
            z = np.random.uniform(0.7, 0.9)
            co = Coordinates(pos=[x, y, z], rot=[0.0, 1.6, 0.5])

            out = arm.inverse_kinematics(co, link_list=arm.link_list, move_target=end_coords)
            if out is not False:
                lib.append(np.array(out))
        return lib

    def wait_until_ready(self) -> None:
        timeout = 10.0
        ts = time.time()
        while not self.msg_available():
            time.sleep(0.1)
            if time.time() - ts > timeout:
                rospy.logwarn("timeout in wait_until_ready")
                self.sound_client.say(
                    "Could not find object for {} seconds".format(timeout), local=True
                )
                self.sound_client.say("Please input y after fixing the environment", local=True)
                # self.sound_client.say("aaaaaaaaaaaaaaaaaaaaaaaaa", local=True)
                while True:
                    user_input = input("push y after fixing the environment")
                    if user_input.lower() == "y":
                        break
                ts = time.time()

    def recover(
        self,
        xy_desired: np.ndarray,
        yaw_desired: float,
    ) -> bool:
        assert yaw_desired is None or (-np.pi < yaw_desired < 0.0)

        if self.is_simulation:
            return False
        self.reset()

        def create_pregrasp_and_grasp_poses(co_obj):
            co_pregrasp = co_obj.copy_worldcoords()
            co_pregrasp.translate([0.05, 0.0, 0.0])
            co_pregrasp.translate([0.0, 0.0, 0.074])
            co_pregrasp.rotate(+np.pi * 0.5, "y")
            co_pregrasp.rotate(-np.pi * 0.5, "x")

            co_grasp = co_pregrasp.copy_worldcoords()
            co_grasp.translate([0.063, 0.0, 0.0])
            return co_pregrasp, co_grasp

        self.wait_until_ready()
        co_obj = self.tf_obj_base.to_skrobot_coords()
        co_pregrasp, co_grasp = create_pregrasp_and_grasp_poses(co_obj)

        xy_displacement = xy_desired - co_obj.worldpos()[:2]
        yaw_now = rpy_angle(co_obj.worldrot())[0][0]  # default
        yaw_displacement = yaw_desired - yaw_now
        if np.abs(yaw_displacement) < 0.1 and np.linalg.norm(xy_displacement) < 0.1:
            rospy.logdebug("no need to recover")
            return True

        co_obj_desired = co_obj.copy_worldcoords()
        co_obj_desired.translate([xy_displacement[0], xy_displacement[1], 0.0], wrt="world")
        co_obj_desired.rotate(yaw_displacement, "z")
        co_predesired, co_desired = create_pregrasp_and_grasp_poses(co_obj_desired)

        plan_config = PlanningCongig(
            self.tf_obj_base, self.pr2, consider_dummy_obstacle=False, arm="rarm"
        )
        q_init = get_robot_state(self.pr2, plan_config.joint_names)

        class _PlanningFailure(Exception):
            pass

        def solve_ik_with_collision_check(
            q_seed, co_target, colfree_const
        ) -> Tuple[np.ndarray, np.ndarray]:
            self.pr2.rarm.angle_vector(q_seed)
            ret = self.pr2.inverse_kinematics(
                co_target, link_list=self.pr2.rarm.link_list, move_target=self.pr2.rarm_end_coords
            )
            if ret is False:
                raise _PlanningFailure()
            q = self.pr2.rarm.angle_vector()
            av = self.pr2.angle_vector()
            if not colfree_const.is_valid(q):
                raise _PlanningFailure()
            return q, av

        def point_to_point_connection(q1, q2, colfree_const, n_wp=8) -> List[np.ndarray]:
            point_const = ConfigPointConst(q2)
            problem = Problem(q1, plan_config.box_const, point_const, colfree_const, None)
            ompl_config = OMPLSolverConfig(n_max_call=2000, simplify=True)
            ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
            ompl_solver.setup(problem)
            res = ompl_solver.solve()
            if res.traj is None:
                raise _PlanningFailure()
            q_list = list(res.traj.resample(n_wp).numpy())
            return q_list

        def get_avs(q_list: List[np.ndarray]) -> List[np.ndarray]:
            avs = []
            for q in q_list:
                set_robot_state(self.pr2, plan_config.joint_names, q)
                avs.append(self.pr2.angle_vector())
            return avs

        for q_seed_lib in self.recovery_ik_lib():

            try:
                # phase1
                q_pregrasp, av_pregrasp = solve_ik_with_collision_check(
                    q_seed_lib, co_pregrasp, plan_config.colfree_const_all
                )
                q_init_to_q_pregrasp = point_to_point_connection(
                    q_init, q_pregrasp, plan_config.colfree_const_all
                )
                q_grasp, av_grasp = solve_ik_with_collision_check(
                    q_pregrasp, co_grasp, plan_config.colfree_const_table
                )
                q_list_reach = q_init_to_q_pregrasp + [q_grasp]

                # phase2
                q_predesired, av_predesired = solve_ik_with_collision_check(
                    q_pregrasp, co_predesired, plan_config.colfree_const_table
                )
                q_pregrasp_to_q_predesired = point_to_point_connection(
                    q_pregrasp, q_predesired, plan_config.colfree_const_table
                )
                q_desired, av_desired = solve_ik_with_collision_check(
                    q_predesired, co_desired, plan_config.colfree_const_table
                )
                q_list_place = [q_pregrasp] + q_pregrasp_to_q_predesired + [q_desired]

                self.sound_client.say("plan to recovery success. Robot will move.")
                rospy.logdebug(
                    "(recover) successfully planned recovery trajectory. start moving to grasp position"
                )

                # FIXME: move back to home position (it already supposed to be at home position but sometimes not)
                self.ri.angle_vector(self.av_home, time_scale=1.0, time=5.0)

                avs = get_avs(q_list_reach)
                times = [0.6] * (len(avs) - 2) + [1.0, 2.0]
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                self.ri.move_gripper("rarm", 0.0)
                self.ri.wait_interpolation()
                time.sleep(2.0)
                gripper_pos = self.ri.gripper_states["rarm"].process_value
                rospy.logdebug("(recover) gripper_pos: {}".format(gripper_pos))
                if gripper_pos < 0.002:
                    rospy.logwarn("(recover) failed to grasp")
                    self.sound_client.say("failed to grasp. return to home position")
                    self.ri.move_gripper("rarm", 0.03)
                    self.ri.wait_interpolation()
                    q_list_back = [q_pregrasp] + q_init_to_q_pregrasp[::-1]
                    avs = get_avs(q_list_back)
                    times = [0.6] * len(avs)
                    self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                    self.ri.wait_interpolation()
                    return False

                rospy.logdebug("(recover) start placing")
                avs = get_avs(q_list_place)
                times = [0.6] * (len(avs) - 2) + [1.0, 2.0]
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                self.ri.move_gripper("rarm", 0.03)

                rospy.logdebug("(recover) start going back to home position")
                q_list_back = [q_predesired, q_pregrasp] + q_init_to_q_pregrasp[::-1]
                avs = get_avs(q_list_back)
                times = [0.6] * len(avs)
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                rospy.logdebug("recovered procedure finished")
                return True
            except _PlanningFailure:
                continue
        return False

    def robust_execute(
        self,
        planer_pose_traj: List[np.ndarray],
        hypo_error: Optional[np.ndarray] = None,
        rot: float = -np.pi * 0.5,
    ) -> bool:
        time.time()

        xy_desired_nominal = np.array([0.5, 0.0])
        yaw_desired_nominal = -np.pi * 0.5

        xy_desired_trial_list = [xy_desired_nominal]
        yaw_desired_trial_list = [yaw_desired_nominal]
        n_recovery_budget = 5

        for i in range(n_recovery_budget):
            xy_desired_trial = xy_desired_nominal + np.random.uniform(-0.06, 0.06, 2)
            yaw_desired_trial = yaw_desired_nominal + np.random.uniform(-np.pi * 0.2, -np.pi * 0.2)
            xy_desired_trial_list.append(xy_desired_trial)
            yaw_desired_trial_list.append(yaw_desired_trial)

        while True:
            self.reset()
            self.wait_until_ready()

            # check the object is in the right position
            assert self.tf_obj_base is not None
            obj_pos = self.tf_obj_base.to_skrobot_coords().worldpos()

            if obj_pos[0] < 0.55 and abs(obj_pos[1]) < 0.25:
                y = self.execute(planer_pose_traj, hypo_error=hypo_error)
                self.reset()
                if y is not None:
                    return y

            message = "failed to execute. move to recovery"
            rospy.logwarn("message")
            self.sound_client.say(message)

            if len(xy_desired_trial_list) == 0:
                rospy.loginfo("no more trial. ask user to recover manually")
                self.sound_client.say(
                    "No more trial remains. Please recover manually to successful plan", local=True
                )
                rospy.logdebug("error: {}".format(hypo_error))
                while True:
                    user_input = input("push y to retry")
                    if user_input.lower() == "y":
                        break
            else:
                xy_desired = xy_desired_trial_list.pop(0)
                yaw_desired = yaw_desired_trial_list.pop(0)

                recovery_success = self.recover(xy_desired, yaw_desired)
                if recovery_success:
                    rospy.loginfo("recovery success")
                else:
                    rospy.logwarn("recovery failed")

        assert False

    def execute(
        self,
        planer_pose_traj: List[np.ndarray],
        hypo_error: Optional[np.ndarray] = None,
        rot: float = -np.pi * 0.5,
    ) -> Optional[bool]:
        assert self.msg_available()
        assert self.tf_obj_base is not None
        path_debug_args = Path("/tmp/frmax_debug_args.pkl")
        with path_debug_args.open("wb") as f:
            pickle.dump((planer_pose_traj, hypo_error, rot, self.raw_msg), f)

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
            assert self.tf_obj_base is not None
            tf_reach_base = chain_transform(
                tf_reach_obj,  # reach -> object
                self.tf_obj_base,  # object -> base
            )
            return tf_reach_base

        coords_list = []
        for relative_pose in planer_pose_traj:
            tf_reach_base = to_transform(*relative_pose, rot)  # type: ignore
            co_reach = tf_reach_base.to_skrobot_coords()
            coords_list.append(co_reach)

        if not self.is_simulation:
            debug_path = RosPath()
            common_header = Header()
            common_header.stamp = rospy.Time.now()
            common_header.frame_id = "base_footprint"
            for co in coords_list:
                pose_msg = CoordinateTransform.from_skrobot_coords(co).to_ros_pose()
                pose_stamped_msg = PoseStamped(pose=pose_msg)
                pose_stamped_msg.header = common_header
            pose_msg_list = [
                CoordinateTransform.from_skrobot_coords(co).to_ros_pose() for co in coords_list
            ]
            pose_stamped_msg_list = [PoseStamped(pose=msg) for msg in pose_msg_list]
            debug_path.header = common_header
            debug_path.poses = pose_stamped_msg_list
            self.pub.publish(debug_path)

        plan_config = PlanningCongig(self.tf_obj_base, self.pr2)
        joint_names = plan_config.joint_names
        q_init = get_robot_state(self.pr2, joint_names)
        co_reach_init = coords_list[0]
        efkin = plan_config.efkin
        box_const = plan_config.box_const
        colfree_const_all = plan_config.colfree_const_all
        colfree_const_table = plan_config.colfree_const_table
        colfree_const_magcup = plan_config.colfree_const_magcup

        # check if initial pose is collision free
        # NOTE: solve unconstrained IK first. which should be solved...
        # if this cannot be true, then the whole plan is not feasible
        # thus return None. If solved successfully, then check collision
        # and if not collision free, meaning that any configuration
        # satisfying the constraint is not collision free, then return False
        satis_con = SatisfactionConfig(acceptable_error=1e-5, disp=False, n_max_eval=50)
        pose_const = PoseConstraint.from_skrobot_coords([co_reach_init], efkin, self.pr2)
        box_const_dummy = copy.deepcopy(box_const)
        box_const_dummy.lb -= 3.0
        box_const_dummy.ub += 3.0
        ret = satisfy_by_optimization(pose_const, box_const_dummy, None, q_init, config=satis_con)
        if not ret.success:
            rospy.logwarn("failed to plan to initial pose without collision constraint")
            return None
        is_collide = not colfree_const_magcup.is_valid(ret.q)
        if is_collide:
            rospy.loginfo("initial pose is not collision free. consider it as failure")
            return False

        def whole_plan(n_resample) -> Optional[List[np.ndarray]]:
            # solve full plan to initial pose
            pose_const = PoseConstraint.from_skrobot_coords([co_reach_init], efkin, self.pr2)
            problem = Problem(q_init, box_const, pose_const, colfree_const_all, None)
            ompl_config = OMPLSolverConfig(n_max_call=2000, simplify=True)
            ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
            ompl_solver.setup(problem)
            res = ompl_solver.solve()
            if res.traj is None:
                rospy.logwarn("failed to plan to initial pose")
                return None

            # solve ik for each pose
            q_list = list(res.traj.resample(n_resample).numpy())
            for co_reach in coords_list[1:]:
                set_robot_state(self.pr2, joint_names, q_list[-1])
                pose_const = PoseConstraint.from_skrobot_coords([co_reach], efkin, self.pr2)
                pose_const.reflect_skrobot_model(self.pr2)
                satis_con = SatisfactionConfig(acceptable_error=1e-5, disp=False, n_max_eval=50)
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

        n_resample = 8
        q_list = None
        for _ in range(5):
            q_list = whole_plan(n_resample)
            if q_list is None:
                rospy.logdebug("failed to plan, retrying...")
                continue
            else:
                break
        if q_list is None:
            rospy.logwarn("failed to plan")
            return None
        rospy.logdebug("successfully planned reaching + grasping trajectory")

        if self.is_simulation:
            viewer = TrimeshSceneViewer()
            co_object = self.tf_obj_base.to_skrobot_coords()
            axis_object = Axis.from_coords(co_object)
            axis = Axis.from_coords(co_reach)
            viewer.add(self.pr2)
            viewer.add(table)
            viewer.add(dummy_obstacle)
            viewer.add(magcup)
            viewer.add(axis)
            viewer.add(axis_object)
            viewer.show()
            for q in q_list:
                set_robot_state(self.pr2, joint_names, q)
                viewer.redraw()
                time.sleep(1.0)
            time.sleep(1000)

        if self.is_simulation:
            label = self.wait_for_label()
            return label
        else:
            # create full angle vector sequence
            avs = []
            for q in q_list:
                set_robot_state(self.pr2, joint_names, q)
                avs.append(self.pr2.angle_vector())

            times_reach = [0.4 for _ in range(n_resample)]
            times_grasp = [0.5 for _ in range(len(planer_pose_traj) - 1)]
            avs_reach, avs_grasp = avs[:n_resample], avs[n_resample:]
            rospy.logdebug("start reacing")
            self.sound_client.say("reaching phase")
            self.ri.angle_vector_sequence(avs_reach, times=times_reach, time_scale=1.0)
            self.ri.wait_interpolation()

            rospy.logdebug("start grasping phase")
            self.sound_client.say("grasping phase")
            time.sleep(1.5)
            self.ri.angle_vector_sequence(avs_grasp, times=times_grasp, time_scale=1.0)
            self.ri.wait_interpolation()

            self.ri.move_gripper("larm", 0.0)
            time.sleep(2.0)
            self.pr2.larm.move_end_pos(
                [-0.02, 0.0, 0.0], wrt="local"
            )  # move back bit to ensure robustness
            self.ri.angle_vector(self.pr2.angle_vector(), time_scale=1.0, time=1.0)
            self.ri.wait_interpolation()
            label = self.wait_for_label()
            self.ri.move_gripper("larm", 0.05)
            self.ri.angle_vector_sequence(
                avs_reach[::-1], times=[0.4] * len(avs_reach), time_scale=1.0
            )
            self.ri.wait_interpolation()
            self.pr2.angle_vector(self.ri.potentio_vector())

            # check if actually played back
            q_now = get_robot_state(self.pr2, joint_names)
            diff = q_now - self.q_home
            if np.any(diff > 0.2):  # TODO: remove ?
                rospy.logwarn("failed to play back. plan again...")
                configuration_const = ConfigPointConst(self.q_home)
                problem = Problem(q_now, box_const, configuration_const, colfree_const_table, None)
                ompl_config = OMPLSolverConfig(n_max_call=2000, simplify=True)
                ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
                ompl_solver.setup(problem)
                res = ompl_solver.solve()
                if res.traj is None:
                    rospy.logerr("failed to plan to home pose (should not happen)")
                    assert False
                assert res.traj is not None
                q_list = list(res.traj.resample(n_resample).numpy())
                for q in q_list:
                    set_robot_state(self.pr2, joint_names, q)
                    self.ri.angle_vector(self.pr2.angle_vector())
                    self.ri.wait_interpolation()
                rospy.loginfo("at home position")

            if label:
                self.sound_client.say("success")
            else:
                self.sound_client.say("failure")
            return label


def create_trajectory(param: np.ndarray, dt: float = 0.1) -> np.ndarray:
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


def is_valid_param(param: np.ndarray) -> bool:
    traj = create_trajectory(param)
    if np.any(np.abs(traj[:, 2]) > np.deg2rad(45.0)):
        return False
    if np.abs(traj[-1, 0] - traj[0, 0]) > 0.1:
        return False
    return True


@dataclass
class UniformSituationSampler:
    b_min: np.ndarray
    b_max: np.ndarray

    def __call__(self) -> np.ndarray:
        s = np.random.uniform(self.b_min, self.b_max)
        return s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce", action="store_true", help="reprodice the debug arg")
    parser.add_argument("--test", action="store_true", help="init-test")
    parser.add_argument("--init", action="store_true", help="init")
    parser.add_argument("--episode", type=int, default=-1, help="episode number to load")

    args = parser.parse_args()

    if args.reproduce:
        cache_file_path = Path("/tmp/frmax_debug_args.pkl")
        if cache_file_path.exists():
            with Path("/tmp/frmax_debug_args.pkl").open("rb") as f:
                planer_traj, hypo_error, rot, raw_msg = pickle.load(f)

                print("planer_traj: {}".format(planer_traj))
                print("hypo_error: {}".format(hypo_error))
                print("rot: {}".format(rot))
                print("raw_msg: {}".format(raw_msg))
        else:
            param = np.random.randn(21) * 0.0
            planer_traj = create_trajectory(param)
            hypo_error = np.zeros(3)
            rot = -np.pi * 0.5

            raw_msg = PoseStamped()
            raw_msg.pose.position.x = 0.47641722876585824
            raw_msg.pose.position.y = -0.054688228244401484
            raw_msg.pose.position.z = 0.8
            raw_msg.pose.orientation.x = 0.0
            raw_msg.pose.orientation.y = 0.0
            raw_msg.pose.orientation.z = -0.6325926153678488
            raw_msg.pose.orientation.w = 0.7744847209481055

        executor = Executor(raw_msg)
        rospy.loginfo("Object pose is received")
        out = executor.robust_execute(planer_traj, hypo_error=hypo_error, rot=rot)
        rospy.loginfo("label: {}".format(out))
    else:

        situation_sampler = UniformSituationSampler(
            np.array([-0.03, -0.03, -np.pi * 0.2]), np.array([0.03, 0.03, np.pi * 0.2])
        )

        # create initial dataset
        executor = Executor(None, auto_annotation=True)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("frmax_ros")
        data_path = Path(pkg_path) / "data"
        assert data_path.exists()

        sampler: Optional[BlackBoxSampler] = None
        if args.episode >= 0:
            i_episode_offset = args.episode + 1
            cache_file_path = data_path / "sampler_cache-{}.pkl".format(args.episode)
            assert cache_file_path.exists()
            if cache_file_path.exists():
                with cache_file_path.open("rb") as f:
                    sampler = dill.load(f)
        else:
            i_episode_offset = 0
            if len(list(data_path.iterdir())) > 0:
                rospy.loginfo("cache file exists")
                # wait for user input to continue
                while True:
                    user_input = input("push y to remove all cache files and proceed")
                    if user_input.lower() == "y":
                        break
                for p in data_path.iterdir():
                    p.unlink()

            param_init = np.zeros(21)
            # param init is assumed to be success with zero error
            n_init_sample = 5
            X, Y = [], []
            X.append(np.hstack([param_init, np.zeros(3)]))
            Y.append(True)

            assert is_valid_param(param_init)
            traj = create_trajectory(param_init)
            executor.robust_execute(traj)  # this is supposed to be success

            for _ in range(n_init_sample):
                error = situation_sampler()
                is_success = executor.robust_execute(traj, hypo_error=error)
                X.append(np.hstack([param_init, error]))
                Y.append(is_success)
            rospy.loginfo("Y: {}".format(Y))
            executor.sound_client.say("all initial samples are collected")

            X = np.array(X)
            Y = np.array(Y)
            ls_param = np.ones(21) * 3
            ls_err = np.array([0.005, 0.005, np.deg2rad(5.0)])
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
            sampler = DistributionGuidedSampler(
                X,
                Y,
                metric,
                param_init,
                config,
                situation_sampler=situation_sampler,
                is_valid_param=is_valid_param,
            )
        assert sampler is not None

        if args.test or args.init:
            if args.init:
                opt_param = np.zeros(21)
            else:
                opt_param = sampler.optimize(200, method="cmaes")
            results = []
            success_count = 0
            est_success_count = 0
            fp_count = 0
            for i in range(50):
                error = situation_sampler()
                rospy.loginfo("error: {}".format(error))
                is_success_real = executor.robust_execute(
                    create_trajectory(opt_param), hypo_error=error
                )
                rospy.loginfo("is_success_real: {}".format(is_success_real))
                x = np.hstack([opt_param, error])

                if is_success_real:
                    success_count += 1

                rospy.loginfo("success_count: {}".format(success_count))
                rospy.loginfo("success rate: {}".format(success_count / (i + 1)))
                if not args.init:
                    is_success_est = sampler.fslset.is_inside(x)
                    rospy.loginfo("is_success_est: {}".format(is_success_est))
                    results.append((error, is_success_real, is_success_est))

                    if is_success_est:
                        est_success_count += 1
                        if not is_success_real:
                            fp_count += 1

                    rospy.loginfo("fp_count: {}".format(fp_count))
                    if est_success_count > 0:
                        rospy.loginfo("fp rate: {}".format(fp_count / est_success_count))

            if args.init:
                file_name = data_path / "init_result.pkl"
            else:
                file_name = data_path / "test_result-{}.pkl".format(args.episode)
            with file_name.open("wb") as f:
                pickle.dump(results, f)
        else:
            for i in range(100):
                i_episode = i + i_episode_offset
                executor.sound_client.say("episode number {}".format(i_episode))
                rospy.loginfo("iteration: {}".format(i_episode))
                time.sleep(0.5)
                x = sampler.ask()
                assert x is not None
                param, error = x[:-3], x[-3:]
                assert is_valid_param(param)
                rospy.loginfo("param: {}".format(param))
                rospy.loginfo("error: {}".format(error))
                traj = create_trajectory(param)
                y = executor.robust_execute(traj, hypo_error=error)
                rospy.loginfo("label: {}".format(y))
                sampler.tell(x, y)
                cache_file_path = data_path / "sampler_cache-{}.pkl".format(i_episode)
                with cache_file_path.open("wb") as f:
                    dill.dump(sampler, f)
