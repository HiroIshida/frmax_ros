import _thread
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import rospy
from evdev import InputDevice, categorize, ecodes
from frmax2.core import CompositeMetric, DGSamplerConfig, DistributionGuidedSampler
from nav_msgs.msg import Path as RosPath
from rospy import Publisher
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.joint import RotationalJoint
from skrobot.model.primitives import Axis, Box, MeshLink, PointCloudLink
from skrobot.models.pr2 import PR2
from skrobot.sdf import GridSDF
from skrobot.viewers import TrimeshSceneViewer

from frmax_ros.node import ObjectPoseProvider, YellowTapeOffsetProvider


def speak(message: str) -> None:
    rospy.loginfo(message)
    subprocess.call('echo "{}" | festival --tts'.format(message), shell=True)


class PlanningScene:
    table: Box
    dummy_obstacle: Box
    target_object: MeshLink
    pr2: PR2
    axis: Axis
    viewer: Optional[TrimeshSceneViewer]

    def __init__(self, pr2: PR2, target_object: MeshLink, visualize: bool = True):
        self.target_object = target_object
        h = 0.72
        table = Box(extents=[0.5, 0.9, h], with_sdf=True)
        table.translate([0.5, 0, 0.5 * h])
        table.visual_mesh.visual.face_colors = [165, 42, 42, 255]
        self.table = table

        dummy_obstacle = Box([0.45, 0.6, 0.03], pos=[0.5, 0.0, 1.2], with_sdf=True)
        dummy_obstacle.visual_mesh.visual.face_colors = [150, 150, 150, 100]
        self.dummy_obstacle = dummy_obstacle
        self.pr2 = pr2
        self.axis = Axis.from_coords(Coordinates())

        if visualize:
            self.viewer = TrimeshSceneViewer()
            colkin = PR2Config().get_collision_kin(whole_body=True)
            colkin.reflect_skrobot_model(pr2)
            self.colvis = CollisionSphereVisualizationManager(colkin, self.viewer, None)
            self.viewer.add(self.axis)
            self.viewer.add(table)
            self.viewer.add(dummy_obstacle)
            self.viewer.add(target_object)
            self.viewer.add(pr2)
            self.viewer.show()

    def add_debug_sdf_pointcloud(self) -> None:
        sdf: GridSDF = self.target_object.sdf
        points, _ = sdf.surface_points(1000)
        pcloud = PointCloudLink(points)
        self.viewer.add(pcloud)

    def update(
        self, co_object: Optional[Coordinates] = None, co_axis: Optional[Coordinates] = None
    ):
        if co_object is not None:
            self.target_object.newcoords(co_object)
        if co_axis is not None:
            self.axis.newcoords(co_axis)
        if self.viewer is not None:
            self.viewer.redraw()
        self.colvis.update(self.pr2)


@dataclass
class RolloutAbortedException(Exception):
    message: str


class RolloutExecutorBase(ABC):  # TODO: move later to task-agonistic module
    pose_provider: ObjectPoseProvider
    offset_prover: YellowTapeOffsetProvider
    pr2: PR2
    ri: PR2ROSRobotInterface
    pub_grasp_path: Publisher
    scene: PlanningScene

    def __init__(self, target_object: MeshLink, use_obinata_keyboard: bool = False):
        # rospy.init_node("robot_interface", disable_signals=True, anonymous=True)
        self.pr2 = PR2()
        self.ri = PR2ROSRobotInterface(self.pr2)
        self._confine_infinite_rotation(self.pr2, ["caster"])
        self.initialize_robot()
        self.pub_grasp_path = rospy.Publisher("/grasp_path", RosPath, queue_size=1, latch=True)
        self.pose_provider = ObjectPoseProvider()
        self.offset_prover = YellowTapeOffsetProvider()
        self.scene = PlanningScene(self.pr2, target_object)
        # run monitor in background
        if use_obinata_keyboard:
            # sudo chmod 666 /dev/input/event22   # don't forget this
            t = threading.Thread(target=self.monitor_keyboard)
            t.start()

    @staticmethod
    def monitor_keyboard():
        ri = PR2ROSRobotInterface(PR2())
        command = "ls -l /dev/input/by-id | grep 'usb-SIGMACHIP_USB_Keyboard-event-kbd'"

        try:
            result = subprocess.check_output(command, shell=True, text=True)
            device_file = result.split()[-1]
        except subprocess.CalledProcessError as e:
            result = "Error occurred: " + str(e)
            device_file = None
        match = re.search(r"event(\d+)", device_file)
        event_number = match.group(1) if match else None
        device_path = f"/dev/input/event{event_number}"
        keyboard = InputDevice(device_path)
        try:
            for event in keyboard.read_loop():
                if event.type == ecodes.EV_KEY:
                    key_event = categorize(event)
                    if key_event.keystate == key_event.key_down:
                        print(f"Key pressed: {key_event.keycode}. stop robot")
                        av_now = ri.angle_vector()
                        ri.angle_vector(av_now, time_scale=5.0, time=1.0)
                        _thread.interrupt_main()  # fuck

        except KeyboardInterrupt:
            print("Stopping keyboard monitoring")

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
                if np.isinf(joint.max_angle):
                    joint.max_angle = +2 * np.pi
                    rospy.loginfo(f"clamp max angle of {joint.name} from +inf to {joint.max_angle}")

    @abstractmethod
    def initialize_robot(self):
        pass

    @abstractmethod
    def rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        pass

    def recover(self):
        # This is the simplest implementation of recovery
        # which ask human to fix the environment
        while True:
            user_input = input("push y after fixing the environment")
            if user_input.lower() == "y":
                break

    def robust_rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        while True:
            try:
                label = self.rollout(param, error)
                return label
            except RolloutAbortedException as e:
                rospy.logwarn(e.message)
                speak(f"recovery required. reason is {e.message}")
                self.recover()

    def send_command_to_real_robot(
        self, q_traj: List[np.ndarray], times: List[float], arm: Literal["larm", "rarm"]
    ):
        conf = PR2Config(control_arm=arm)
        joint_names = conf.get_control_joint_names()
        av_list = []
        for q in q_traj:
            set_robot_state(self.pr2, joint_names, q)
            av_list.append(self.pr2.angle_vector())

        rospy.loginfo("executing angle vector")
        self.ri.angle_vector_sequence(av_list, times=times, time_scale=1.0)
        self.ri.wait_interpolation()
        rospy.loginfo("finish sending")

    @abstractmethod
    def get_auto_annotation(self) -> Optional[bool]:
        # return None if uncertain and need manual annotation
        pass

    @abstractmethod
    def get_policy_dof(self) -> int:
        pass

    def get_manual_annotation(self) -> Optional[bool]:
        while True:
            speak("manual annotation required")
            user_input = input("Add label: Enter 'y' for True or 'n' for False, r for retry")
            if user_input.lower() == "y":
                return True
            elif user_input.lower() == "n":
                return False
            elif user_input.lower() == "r":
                return None

    def get_label(self) -> Optional[bool]:
        annot = self.get_auto_annotation()
        if annot is None:
            return self.get_manual_annotation()


class AutomaticTrainerBase(ABC):
    i_episode_next: int
    rollout_executor: RolloutExecutorBase
    sampler: DistributionGuidedSampler

    def __init__(
        self,
        ls_param: np.ndarray,
        ls_error: np.ndarray,
        sampler_config: DGSamplerConfig,
        n_init_sample: int = 10,
    ):
        speak("start trining")
        self.i_episode_next = 0
        rollout_executor = self.get_rollout_executor()
        dof = rollout_executor.get_policy_dof()
        param_init = np.zeros(dof)
        X = [np.hstack([param_init, np.zeros(ls_error.size)])]
        Y = [True]
        for i in range(n_init_sample):
            speak(f"initial sampling number {i}")
            e = self.sample_situation()
            label = rollout_executor.robust_rollout(param_init, e)
            X.append(np.hstack([param_init, e]))
            Y.append(label)
        X = np.array(X)
        Y = np.array(Y)
        speak("finish initial sampling")

        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        self.rollout_executor = rollout_executor
        self.sampler = DistributionGuidedSampler(
            X,
            Y,
            metric,
            param_init,
            sampler_config,
            situation_sampler=self.sample_situation,
            is_valid_param=self.is_valid_param,
        )

    @staticmethod
    @abstractmethod
    def get_rollout_executor() -> RolloutExecutorBase:
        pass

    @abstractmethod
    def sample_situation(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_valid_param(self, param: np.ndarray) -> bool:
        pass

    def next(self) -> None:
        speak(f"start episode {self.i_episode_next}")
        x = self.sampler.ask()
        assert isinstance(x, np.ndarray)
        speak(f"determined next data point")
        param_dof = self.rollout_executor.get_policy_dof()
        param, error = x[:param_dof], x[param_dof:]
        label = self.rollout_executor.robust_rollout(param, error)
        speak(f"label {label}")
        time.sleep(1.0)
        self.sampler.tell(x, label)
        self.i_episode_next += 1
