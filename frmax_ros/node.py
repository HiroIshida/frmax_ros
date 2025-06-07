import time
from typing import List, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from geometry_msgs.msg import PoseStamped
from rospy import Publisher, Subscriber
from sensor_msgs.msg import PointCloud2
from skrobot.coordinates.math import (
    quaternion2rpy,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)
from typing_extensions import Optional

from frmax_ros.utils import CoordinateTransform


class AverageQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def enqueue(self, value_with_stamp: Tuple[np.ndarray, rospy.Time]):
        value, stamp = value_with_stamp
        assert np.all(np.isfinite(value))
        self.queue.append(value_with_stamp)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    @property
    def values(self) -> List[np.ndarray]:
        return [vws[0] for vws in self.queue]

    @property
    def stamps(self) -> List[float]:
        return [vws[1].to_sec() for vws in self.queue]

    def get_average(self) -> np.ndarray:
        if not self.queue:
            return None
        return np.mean(self.values, axis=0)

    def get_std(self) -> np.ndarray:
        if not self.queue:
            return None
        return np.std(self.values, axis=0)

    def duration(self) -> float:
        if not self.queue:
            return np.inf
        t_oldest = self.queue[0][1].to_sec()
        t_latest = self.queue[-1][1].to_sec()
        return t_latest - t_oldest


class ObjectPoseProvider:
    listener: tf.TransformListener
    queue: AverageQueue
    _tf_april_to_base: Optional[CoordinateTransform]
    _z_height: Optional[float]
    _acceptable_xy_std: float
    _acceptable_theta_std: float

    def __init__(self, calibrate_z: bool = True, april_fram_name: str = "apriltag_id0"):
        self.listener = tf.TransformListener()
        self.queue = AverageQueue(max_size=10)
        self._tf_april_to_base = None
        self._z_height = None
        rospy.Timer(rospy.Duration(0.1), self.update_queue)
        self.pub_april_pose = Publisher("april_pose", PoseStamped, queue_size=1, latch=True)
        if calibrate_z:
            self.sub_points = Subscriber(
                "/april_z_calibration/ExtractIndices/output", PointCloud2, self.callback_points
            )
        self._acceptable_xy_std = 0.005
        self._acceptable_theta_std = 0.03
        self._april_fram_name = april_fram_name
        self._calibrate_z = calibrate_z

    def callback_points(self, msg: PointCloud2) -> None:
        if self._z_height is not None:
            return
        arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
        self._z_height = np.sort(arr[:, 2])[-40]

    def reset(self, acceptable_xy_std: float = 0.005, acceptable_theta_std: float = 0.03) -> None:
        self.queue = AverageQueue(max_size=10)
        self._tf_april_to_base = None
        self._z_height = None
        self._acceptable_xy_std = acceptable_xy_std
        self._acceptable_theta_std = acceptable_theta_std

    def get_tf_object_to_base(self, timeout: float = 10.0) -> CoordinateTransform:
        if self._tf_april_to_base is not None:
            return self._tf_april_to_base
        ts = time.time()
        while self._tf_april_to_base is None:
            if time.time() - ts > timeout:
                raise TimeoutError
            rospy.sleep(0.1)
        assert self._tf_april_to_base is not None
        return self._tf_april_to_base

    @staticmethod
    def xyztheta_to_pose(xyztheta: np.ndarray, target_frame: str) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xyztheta[0]
        pose.pose.position.y = xyztheta[1]
        pose.pose.position.z = xyztheta[2]
        rot = wxyz2xyzw(rpy2quaternion([xyztheta[3], 0, 0]))
        pose.pose.orientation.x = rot[0]
        pose.pose.orientation.y = rot[1]
        pose.pose.orientation.z = rot[2]
        pose.pose.orientation.w = rot[3]
        return pose

    def update_queue(self, event) -> None:
        if self._calibrate_z and self._z_height is None:
            return
        if self._tf_april_to_base is not None:
            return
        trans: Optional[np.ndarray] = None
        rot: Optional[np.ndarray] = None
        target_frame = "base_footprint"
        source_frame = self._april_fram_name
        try:
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("TF Exception")
            return

        assert trans is not None and rot is not None

        # we care only yaw and know that othere angles are 0
        ypr = quaternion2rpy(xyzw2wxyz(rot))[0]
        xyztheta = np.hstack([trans, ypr[0]])
        if self._calibrate_z:
            xyztheta[2] = self._z_height

        self.queue.enqueue((xyztheta, rospy.Time.now()))

        self.xyztheta_to_pose(xyztheta, target_frame)

        if self.queue.duration() > 2.0:
            rospy.loginfo("TF is too old")
            return

        if len(self.queue) < self.queue.max_size:
            rospy.loginfo("Waiting for TF. Current queue size: {}".format(len(self.queue)))
            return

        std = self.queue.get_std()
        xy_std = std[:2]
        theta_std = std[3]
        if np.any(xy_std > self._acceptable_xy_std) or theta_std > self._acceptable_theta_std:
            rospy.loginfo("TF is too noisy: {}".format(std))
            return

        xyztheta_filtered = self.queue.get_average()
        pose_filtered = self.xyztheta_to_pose(xyztheta_filtered, target_frame)
        tf_april_to_base = CoordinateTransform.from_ros_pose(pose_filtered.pose, "april", "base")
        self._tf_april_to_base = tf_april_to_base

        april_pose = tf_april_to_base.to_ros_pose()
        object_pose_stamped = PoseStamped()
        object_pose_stamped.header.frame_id = "base_footprint"
        object_pose_stamped.header.stamp = rospy.Time.now()
        object_pose_stamped.pose = april_pose
        self.pub_april_pose.publish(object_pose_stamped)
        rospy.loginfo("Published april: {}".format(april_pose))


class LarmEndEffectorOffsetProvider:
    def __init__(self):
        self.object_poes_provider = ObjectPoseProvider(
            calibrate_z=False, april_fram_name="apriltag_id1"
        )
        self.listener = tf.TransformListener()
        self.offset = None

    def reset(self) -> None:
        self.object_poes_provider.reset()

    def get_offset(self, timeout: float = 5.0) -> np.ndarray:
        # offset wrt base frame
        tf_april_to_base = self.object_poes_provider.get_tf_object_to_base(timeout=timeout)
        target_frame = "base_footprint"
        source_frame = "apriltag_fk"
        try:
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("TF Exception LarmEndEffectorOffsetProvider")
            return
        assert trans is not None and rot is not None
        offset = trans - tf_april_to_base.trans
        return offset


if __name__ == "__main__":
    rospy.init_node("tf_listener")
    provider2 = LarmEndEffectorOffsetProvider()
    provider2.reset()
    offset = provider2.get_offset()
    print(offset)
