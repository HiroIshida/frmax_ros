import time
from typing import List, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros
import tf2_sensor_msgs
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

from frmax_ros.utils import CoordinateTransform, chain_transform


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
    tf_object_to_april: CoordinateTransform
    _tf_object_to_base: Optional[CoordinateTransform]

    def __init__(self, tf_object_to_april: CoordinateTransform):
        self.listener = tf.TransformListener()
        self.queue = AverageQueue(max_size=10)
        self._tf_object_to_base = None
        self.tf_object_to_april = tf_object_to_april
        rospy.Timer(rospy.Duration(0.1), self.update_queue)
        self.pub_object_pose = Publisher("object_pose", PoseStamped, queue_size=1, latch=True)

    def reset(self):
        self.queue = AverageQueue(max_size=10)
        self._tf_object_to_base = None

    def get_tf_object_to_base(self, timeout: float = 10.0) -> CoordinateTransform:
        if self._tf_object_to_base is not None:
            return self._tf_object_to_base
        ts = time.time()
        while self.get_tf_object_to_base is None:
            if time.time() - ts > timeout:
                raise TimeoutError
            rospy.sleep(0.1)
        assert self._tf_object_to_base is not None
        return self._tf_object_to_base

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
        if self._tf_object_to_base is not None:
            return
        trans: Optional[np.ndarray] = None
        rot: Optional[np.ndarray] = None
        target_frame = "base_footprint"
        source_frame = "april"
        try:
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("TF Exception")
            return

        assert trans is not None and rot is not None

        # we care only yaw and know that othere angles are 0
        ypr = quaternion2rpy(xyzw2wxyz(rot))[0]
        xyztheta = np.hstack([trans, ypr[0]])
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
        if np.any(xy_std > 0.005) or theta_std > 0.03:
            rospy.loginfo("TF is too noisy: {}".format(std))
            return

        xyztheta_filtered = self.queue.get_average()
        pose_filtered = self.xyztheta_to_pose(xyztheta_filtered, target_frame)
        tf_april_to_base = CoordinateTransform.from_ros_pose(
            pose_filtered.pose, "april", "base_footprint"
        )
        tf_object_to_base = chain_transform(self.tf_object_to_april, tf_april_to_base)
        self._tf_object_to_base = tf_object_to_base

        object_pose = tf_object_to_base.to_ros_pose()
        object_pose_stamped = PoseStamped()
        object_pose_stamped.header.frame_id = "base_footprint"
        object_pose_stamped.header.stamp = rospy.Time.now()
        object_pose_stamped.pose = object_pose
        self.pub_object_pose.publish(object_pose_stamped)
        rospy.loginfo("Published object_pose: {}".format(object_pose))


class PointCloudProvider:
    buffer: tf2_ros.Buffer
    _pcloud: Optional[np.ndarray]

    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buffer)
        self.sub_pcloud = Subscriber(
            "/kinect_head/depth_registered/half/throttled/points", PointCloud2, self.callback
        )
        self._pcloud = None

    def callback(self, msg: PointCloud2) -> None:
        if self._pcloud is not None:
            return
        rospy.loginfo("PointCloud is received")
        target_frame = "base_footprint"
        source_frame = msg.header.frame_id
        try:
            transform = self.buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.loginfo("TF Exception: {}".format(e))
            return
        transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
        gen = pc2.read_points(transformed_cloud, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        self._pcloud = points
        rospy.loginfo("PointCloud is received and stored")


if __name__ == "__main__":
    rospy.init_node("tf_listener")
    tf_object_april = CoordinateTransform(np.zeros(3), np.eye(3), "object", "april")
    provider1 = ObjectPoseProvider(tf_object_april)
    provider2 = PointCloudProvider()
    rospy.spin()
