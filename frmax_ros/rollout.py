import _thread
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import dill
import numpy as np
import rospkg
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
from sound_play.libsoundplay import SoundClient

from frmax_ros.node import LarmEndEffectorOffsetProvider, ObjectPoseProvider


def speak(message: str) -> None:
    rospy.loginfo(message)
    subprocess.call('echo "{}" | festival --tts'.format(message), shell=True)
    sound_client = SoundClient()
    try:
        sound_client.say(message, volume=0.5)
    except:
        print("sound client error")


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
        table = Box(
            extents=[0.7, 1.1, h], with_sdf=True
        )  # x and y are bit larger than the actual table
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
    april_recognition: bool


class RolloutExecutorBase(ABC):  # TODO: move later to task-agonistic module
    pose_provider: ObjectPoseProvider
    offset_prover: LarmEndEffectorOffsetProvider
    pr2: PR2
    ri: PR2ROSRobotInterface
    pub_grasp_path: Publisher
    scene: PlanningScene

    def __init__(self, target_object: MeshLink, use_obinata_keyboard: bool = True):
        # rospy.init_node("robot_interface", disable_signals=True, anonymous=True)
        self.pr2 = PR2()
        self.ri = PR2ROSRobotInterface(self.pr2)
        self._confine_infinite_rotation(self.pr2, ["caster"])
        self.initialize_robot()
        self.pub_grasp_path = rospy.Publisher("/grasp_path", RosPath, queue_size=1, latch=True)
        self.pose_provider = ObjectPoseProvider()
        self.offset_prover = LarmEndEffectorOffsetProvider
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

    def recover(self) -> bool:
        rospy.logwarn("recovery is not implemented")
        return False

    def robust_rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        while True:
            try:
                return self.rollout(param, error)
            except RolloutAbortedException as e:
                rospy.logwarn(e.message)
                speak(f"automatic recovery start")

                if not e.april_recognition:
                    if self.recover():
                        speak(f"automatic recovery finished")
                    try:
                        return self.rollout(param, error)
                    except RolloutAbortedException:
                        speak(f"rollout failed even after automatic recovery")
                speak(f"manual recovery required. reason is {e.message}")
                while True:
                    user_input = input("push y after fixing the environment")
                    if user_input.lower() == "y":
                        break

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


@dataclass
class AutomaticTrainerBase(ABC):
    i_episode_next: int
    i_refine_episode_next: int
    rollout_executor: RolloutExecutorBase
    sampler: DistributionGuidedSampler

    @classmethod
    def init(
        cls,
        ls_param: np.ndarray,
        ls_error: np.ndarray,
        sampler_config: DGSamplerConfig,
        n_init_sample: int = 10,
    ) -> "AutomaticTrainerBase":

        project_path = cls.get_project_path()

        if len(list(project_path.iterdir())) > 0:
            while True:
                user_input = input("push y to remove all cache files and proceed")
                if user_input.lower() == "y":
                    break
            for p in project_path.iterdir():
                p.unlink()

        speak("start training")
        rollout_executor = cls.get_rollout_executor()
        dof = rollout_executor.get_policy_dof()
        param_init = np.zeros(dof)
        X = [np.hstack([param_init, np.zeros(ls_error.size)])]
        Y = [True]
        for i in range(n_init_sample):
            speak(f"initial sampling number {i}")
            e = cls.sample_situation()
            label = rollout_executor.robust_rollout(param_init, e)
            X.append(np.hstack([param_init, e]))
            Y.append(label)
        X = np.array(X)
        Y = np.array(Y)
        speak("finish initial sampling")

        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        rollout_executor = rollout_executor
        sampler = DistributionGuidedSampler(
            X,
            Y,
            metric,
            param_init,
            sampler_config,
            situation_sampler=cls.sample_situation,
            is_valid_param=cls.is_valid_param,
            use_prefacto_branched_ask=False,
        )
        return cls(0, 0, rollout_executor, sampler)

    @staticmethod
    @abstractmethod
    def get_project_name() -> str:
        pass

    @classmethod
    def get_project_path(cls) -> Path:
        project_name = cls.get_project_name()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("frmax_ros")
        data_path = Path(pkg_path) / "data"
        assert data_path.exists()
        project_path = data_path / project_name
        project_path.mkdir(exist_ok=True)
        return project_path

    @staticmethod
    @abstractmethod
    def get_rollout_executor() -> RolloutExecutorBase:
        pass

    @staticmethod
    @abstractmethod
    def sample_situation() -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def is_valid_param(param: np.ndarray) -> bool:
        pass

    def save(self):
        assert self.i_refine_episode_next == 0
        project_path = self.get_project_path()
        data_path = project_path / f"sampler-{self.i_episode_next}.pkl"
        with open(data_path, "wb") as f:
            dill.dump(self.sampler, f)
        rospy.loginfo(f"save sampler to {data_path}")

    def save_refined(self):
        assert self.i_refine_episode_next > 0
        project_path = self.get_project_path()
        data_path = project_path / f"refined-{self.i_episode_next}-{self.i_refine_episode_next}.pkl"
        with open(data_path, "wb") as f:
            dill.dump(self.sampler, f)
        rospy.loginfo(f"save sampler to {data_path}")

    @classmethod
    def load_sampler(
        cls, i_episode_next: Optional[int] = None
    ) -> Tuple[DistributionGuidedSampler, int]:
        project_path = cls.get_project_path()

        if i_episode_next is not None:
            max_episode_path = project_path / f"sampler-{i_episode_next}.pkl"
            assert max_episode_path.exists(), f"no cache file found at {max_episode_path}"
            with open(max_episode_path, "rb") as f:
                sampler = dill.load(f)
            return sampler, i_episode_next
        else:
            max_episode = -1
            max_episode_path = None
            for p in project_path.iterdir():
                if p.name.startswith("sampler-"):
                    episode = int(p.name.split("-")[-1].split(".")[0])
                    if episode > max_episode:
                        max_episode = episode
                        max_episode_path = p
            assert max_episode_path is not None, "no cache file found"
            rospy.loginfo(f"load sampler from {max_episode_path}")
            with open(max_episode_path, "rb") as f:
                sampler = dill.load(f)
            return sampler, max_episode

    @classmethod
    def load_refined_sampler(
        cls, i_refine_next: Optional[int] = None
    ) -> Tuple[DistributionGuidedSampler, int, int]:
        project_path = cls.get_project_path()
        d = {}
        for p in project_path.iterdir():
            pattern = r"^refined-(\d+)-(\d+)\.pkl$"
            match = re.match(pattern, p.name)
            if match is not None:
                i_episode_next = int(match.group(1))
                if i_episode_next not in d:
                    d[i_episode_next] = []
                i_refine_episode_next = int(match.group(2))
                d[i_episode_next].append(i_refine_episode_next)

        if len(d.keys()) == 0:
            raise FileNotFoundError("no cache file found")
        elif len(d.keys()) > 1:
            raise ValueError("multiple cache files found")

        i_episode_next = list(d.keys())[0]
        if i_refine_next is None:
            i_refine_episode_next = max(d[i_episode_next])
        else:
            i_refine_episode_next = i_refine_next
        latest_episode_path = project_path / f"refined-{i_episode_next}-{i_refine_episode_next}.pkl"
        assert latest_episode_path.exists(), f"no cache file found at {latest_episode_path}"
        with open(latest_episode_path, "rb") as f:
            sampler = dill.load(f)
        return sampler, i_episode_next, i_refine_episode_next

    @classmethod
    def load(cls, i_episode_next: Optional[int] = None):
        sampler, i_episode_next = cls.load_sampler(i_episode_next)
        rollout_executor = cls.get_rollout_executor()
        return cls(i_episode_next, 0, rollout_executor, sampler)

    @classmethod
    def load_refined(cls):
        sampler, i_episode_next, i_refine_episode_next = cls.load_refined_sampler()
        rollout_executor = cls.get_rollout_executor()
        return cls(i_episode_next, i_refine_episode_next, rollout_executor, sampler)

    def step(self) -> None:
        speak(f"start episode {self.i_episode_next}")

        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()

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
        self.save()

        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

    def step_refinement(self) -> None:
        speak(f"start refinement {self.i_refine_episode_next}")

        if self.sampler.count_additional == 0:
            param_opt = self.sampler.optimize(200, method="cmaes")
        else:
            param_opt = self.sampler.get_optimal_after_additional()
        x = self.sampler.ask_additional(param_opt)
        param_dof = self.rollout_executor.get_policy_dof()
        param, error = x[:param_dof], x[param_dof:]
        label = self.rollout_executor.robust_rollout(param, error)
        speak(f"label {label}")
        time.sleep(1.0)
        self.sampler.tell(x, label)
        self.i_refine_episode_next += 1
        self.save_refined()
