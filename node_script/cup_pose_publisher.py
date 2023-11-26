#!/usr/bin/env python

from typing import ClassVar, List, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sklearn.decomposition import PCA
from skrobot.coordinates import Coordinates
from utils import CoordinateTransform


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

    def observation_time_center(self) -> float:
        if len(self.queue) < 0:
            return np.inf
        t_oldest = self.queue[0][1].to_sec()
        t_latest = self.queue[-1][1].to_sec()
        duration = t_latest - t_oldest
        return duration / len(self.queue)

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

    def is_valid(self) -> bool:
        if len(self.queue) < self.max_size:
            return False
        t_oldest = self.queue[0][1].to_sec()
        t_latest = self.queue[-1][1].to_sec()
        duration = t_latest - t_oldest
        if duration < self.max_size * 1.0:
            return True

    def is_steady(self, threashold: float = 0.005) -> bool:
        if not self.queue:
            return False
        std = self.get_std()
        return np.all(std < threashold)


class LaserScanToPointCloud:
    pose_average_queue: AverageQueue
    cup_height: ClassVar[float] = 0.80

    def __init__(self):
        self.cloud_subscriber = rospy.Subscriber(
            "/perception/hsi_filter/output", PointCloud2, self.callback_cloud
        )
        self.object_pose_publisher = rospy.Publisher("/object_pose", PoseStamped, queue_size=10)
        self.object_pose_raw_publisher = rospy.Publisher(
            "/object_pose_raw", PoseStamped, queue_size=10
        )
        self.pose_average_queue = AverageQueue(20)

    def xyzyaw_to_pose_stamped(self, xyzyaw: np.ndarray) -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        x, y, z, yaw = xyzyaw
        co = Coordinates(np.hstack([x, y, z]))
        co.rotate(yaw - np.pi * 0.5, "z")
        pose = CoordinateTransform.from_skrobot_coords(co).to_ros_pose()
        pose_stamped.pose = pose
        return pose_stamped

    def callback_cloud(self, msg: PointCloud2):
        assert msg.header.frame_id == "base_footprint"
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        points_xy = points[:, :2]
        if points_xy.shape[0] < 20:
            rospy.logwarn("too few points")
            return
        center = np.mean(points, axis=0)
        pca = PCA(n_components=1)
        pca.fit(points_xy)
        direction_vector = pca.components_[0]
        if direction_vector[0] < 0:
            direction_vector = -direction_vector
        yaw_angle = np.arctan2(direction_vector[1], direction_vector[0])
        xyzyaw = np.hstack([center, yaw_angle])
        rospy.loginfo("pose raw: {}".format(xyzyaw))
        pose_stamped_raw = self.xyzyaw_to_pose_stamped(xyzyaw)
        pose_stamped_raw.header.frame_id = "base_footprint"
        self.object_pose_raw_publisher.publish(pose_stamped_raw)

        self.pose_average_queue.enqueue((xyzyaw, msg.header.stamp))

        bad_status_list = []
        pose_std = self.pose_average_queue.get_std()
        rospy.loginfo("pose std: {}".format(pose_std))
        if pose_std[0] > 0.005 or pose_std[1] > 0.005 or pose_std[2] > 0.005 or pose_std[2] > 0.02:
            bad_status_list.append("pose not steady: {}".format(pose_std))
        if len(bad_status_list) > 0:
            rospy.logwarn("Bad status: {}".format(", ".join(bad_status_list)))
            return
        pose_stamped = self.xyzyaw_to_pose_stamped(xyzyaw)
        pose_stamped.header.frame_id = "base_footprint"
        self.object_pose_publisher.publish(pose_stamped)
        rospy.loginfo("publish object pose kinect: {}".format(xyzyaw))


if __name__ == "__main__":
    rospy.init_node("laser_scan_to_point_cloud")
    l2pc = LaserScanToPointCloud()
    rospy.spin()
