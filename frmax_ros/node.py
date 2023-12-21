import time
from typing import List, Optional, Tuple

import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros
import tf2_sensor_msgs
from geometry_msgs.msg import PoseStamped
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array
from rospy import Publisher, Subscriber
from sensor_msgs.msg import JointState, PointCloud2
from sklearn.cluster import DBSCAN
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2rpy,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.link import Link
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer
from trimesh import PointCloud
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
    april_z_offset: float

    def __init__(self, april_z_offset: float = 0.016):
        self.listener = tf.TransformListener()
        self.queue = AverageQueue(max_size=10)
        self._tf_april_to_base = None
        self.april_z_offset = april_z_offset
        rospy.Timer(rospy.Duration(0.1), self.update_queue)
        self.pub_april_pose = Publisher("april_pose", PoseStamped, queue_size=1, latch=True)

    def reset(self):
        self.queue = AverageQueue(max_size=10)
        self._tf_april_to_base = None

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
        if self._tf_april_to_base is not None:
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
        xyztheta[2] += self.april_z_offset

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
        tf_april_to_base = CoordinateTransform.from_ros_pose(pose_filtered.pose, "april", "base")
        self._tf_april_to_base = tf_april_to_base

        april_pose = tf_april_to_base.to_ros_pose()
        object_pose_stamped = PoseStamped()
        object_pose_stamped.header.frame_id = "base_footprint"
        object_pose_stamped.header.stamp = rospy.Time.now()
        object_pose_stamped.pose = april_pose
        self.pub_april_pose.publish(object_pose_stamped)
        rospy.loginfo("Published april: {}".format(april_pose))


class PointCloudProvider:
    buffer: tf2_ros.Buffer
    _pcloud: Optional[np.ndarray]

    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buffer)
        self.sub_pcloud = Subscriber(
            "/kinect_head/depth_registered/throttled/points", PointCloud2, self.callback
        )
        self._pcloud = None

    def get_point_cloud(self, timeout: float = 10.0) -> np.ndarray:
        if self._pcloud is not None:
            return self._pcloud
        ts = time.time()
        while self._pcloud is None:
            if time.time() - ts > timeout:
                raise TimeoutError
            rospy.sleep(1.0)
        assert self._pcloud is not None
        return self._pcloud

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


class YellowTapeOffsetProvider:
    pr2: PR2
    ri: PR2ROSRobotInterface
    gripper_link: Link
    viewer: Optional[TrimeshSceneViewer]
    visualize: bool
    method: str
    _tf_cloudtape_to_tape: Optional[CoordinateTransform]

    def __init__(self, visualize: bool = False, method: str = "naive"):
        self.pr2 = PR2()
        sub_pcloud = message_filters.Subscriber("/yellow_tape/hsi_filter/output", PointCloud2)
        sub_joint_states = message_filters.Subscriber("/joint_states", JointState)
        message_filter = message_filters.ApproximateTimeSynchronizer(
            [sub_pcloud, sub_joint_states], 100, 0.3
        )
        message_filter.registerCallback(self.callback_pointcloud_colored)
        self.gripper_link = self.pr2.l_gripper_l_finger_link
        self.viewer = None
        self.visualize = visualize
        self.method = method
        self._tf_cloudtape_to_tape = None

    def get_cloudtape_to_tape(self, timeout: float = 5.0) -> CoordinateTransform:
        if self._tf_cloudtape_to_tape is not None:
            return self._tf_cloudtape_to_tape

        ts = time.time()
        while self._tf_cloudtape_to_tape is None:
            if time.time() - ts > timeout:
                raise TimeoutError("Timeout waiting for tf_cloudtape_to_tape")
            rospy.sleep(0.05)
            rospy.loginfo("Waiting for tf_cloudtape_to_tape")
        return self._tf_cloudtape_to_tape

    def reset(self) -> None:
        self._tf_cloudtape_to_tape = None
        rospy.loginfo("tf_cloudtape_to_tape is reset")

    def callback_pointcloud_colored(
        self, pcloud_msg: PointCloud2, joint_state_msg: JointState
    ) -> None:
        if self._tf_cloudtape_to_tape is not None:
            return
        table = {name: angle for name, angle in zip(joint_state_msg.name, joint_state_msg.position)}
        [self.pr2.__dict__[name].joint_angle(angle) for name, angle in table.items()]
        co_actual = self.gripper_link.copy_worldcoords()
        co_actual.translate([0.02, 0.045, 0.0])
        co_actual.rotate(-np.pi * 0.06, "z")

        time.time()
        arr = pointcloud2_to_array(pcloud_msg).flatten()
        xyz = get_xyz_points(arr, remove_nans=True)  # nan filter by myself
        if len(xyz) == 0:
            rospy.logwarn("No yellow point cloud found")
            return

        # clustering
        dbscan = DBSCAN(eps=0.005, min_samples=3)
        clusters = dbscan.fit_predict(xyz)
        n_label = np.max(clusters) + 1
        cluster_sizes = [np.sum(clusters == i) for i in range(n_label)]
        largest_cluster_idx = np.argmax(cluster_sizes)
        points_clustered = xyz[clusters == largest_cluster_idx]

        # if multiple clusters are found, the largest one corresponds to the tape
        # should have highest z value
        clusters = [xyz[clusters == i] for i in range(n_label)]
        z_means = [np.mean(cluster) for cluster in clusters]
        highest_cluster_idx = np.argmax(z_means)
        if highest_cluster_idx != largest_cluster_idx:
            rospy.logerr("The highest cluster is not the largest one")
            return

        mean = np.mean(points_clustered, axis=0)
        co_from_cloud = Coordinates(pos=mean, rot=co_actual.worldrot())

        tf_cloudtape_to_base = CoordinateTransform.from_skrobot_coords(
            co_from_cloud, "cloud_tape", "base"
        )
        tf_tape_to_base = CoordinateTransform.from_skrobot_coords(co_actual, "tape", "base")
        self._tf_cloudtape_to_tape = tf_cloudtape_to_base * tf_tape_to_base.inverse()
        rospy.loginfo(f"tf_cloudtape_to_tape: {self._tf_cloudtape_to_tape}")

        if self.viewer is None and self.visualize:
            self.viewer = TrimeshSceneViewer()
            self.viewer.add(self.pr2)
            link = Link()
            axis_from_cloud = Axis.from_coords(co_from_cloud, axis_radius=0.002)
            axis = Axis.from_coords(co_actual, axis_radius=0.002)
            self.viewer.add(axis)
            self.viewer.add(axis_from_cloud)
            link._visual_mesh = PointCloud(points_clustered)
            self.plink = link
            self.viewer.add(link)
            self.viewer.show()


if __name__ == "__main__":
    rospy.init_node("tf_listener")
    tf_object_april = CoordinateTransform(np.zeros(3), np.eye(3), "object", "april")
    provider1 = ObjectPoseProvider()
    provider2 = PointCloudProvider()
    provider3 = YellowTapeOffsetProvider(visualize=True)
    rospy.spin()
