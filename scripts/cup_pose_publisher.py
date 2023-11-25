#!/usr/bin/env python

from typing import List, Optional, Tuple

import laser_geometry.laser_geometry as lg
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud2
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
    pcloud_mst_list: List[PointCloud2]
    n_collect: int
    transform_pre: Optional[tf2_ros.TransformStamped]
    quat_history: List[np.ndarray]
    diff_history: List[float]
    pcloud_processed: Optional[np.ndarray]
    center_average_queue: AverageQueue
    yaw_average_queue: AverageQueue

    def __init__(self, n_collect: int = 100):
        self.pcloud_msg_list = []
        self.n_collect = n_collect

        self.laser_subscriber = rospy.Subscriber("/tilt_scan", LaserScan, self.callback_scan)

        self.cloud_subscriber = rospy.Subscriber(
            "/perception/hsi_filter/output", PointCloud2, self.callback_cloud
        )

        self.object_pose_publisher = rospy.Publisher("/object_pose", PoseStamped, queue_size=1)
        self.debug_cloud_publisher = rospy.Publisher("/debug_cloud", PointCloud2, queue_size=1)

        rospy.Timer(rospy.Duration(0.5), self.publish_object_pose)
        rospy.Timer(rospy.Duration(1.0), self.publish_debug_cloud)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_pre = None
        self.quat_history = []
        self.diff_history = []
        self.pcloud_processed = None
        self.center_average_queue = AverageQueue(5)
        self.yaw_average_queue = AverageQueue(5)

    def callback_scan(self, msg: LaserScan):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_footprint", msg.header.frame_id, rospy.Time(0), rospy.Duration(0.01)
            )
            quat_now = np.array([transform.transform.rotation.y, transform.transform.rotation.w])
            if len(self.quat_history) == 0:
                diff_quat = np.inf
            else:
                quat_pre = self.quat_history[-1]
                diff_quat = float(np.linalg.norm(quat_now - quat_pre))
                self.diff_history.append(diff_quat)
            self.quat_history.append(quat_now)
            if len(self.diff_history) < 10:
                return
            mean_diff_recent = np.mean(self.diff_history[-10:])  # type: ignore
            if mean_diff_recent > 0.015:
                rospy.logerr("diff quat too large, skip")
                return

            if isinstance(msg, LaserScan):
                p = lg.LaserProjection()
                msg = p.projectLaser(msg)
            assert isinstance(msg, PointCloud2)
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
            self.pcloud_msg_list.append(transformed_cloud)
            if len(self.pcloud_msg_list) > self.n_collect:
                self.pcloud_msg_list.pop(0)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("TF Error: %s" % str(e))

    def callback_cloud(self, msg: PointCloud2):
        assert msg.header.frame_id == "base_footprint"
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        points_xy = points[:, :2]
        if points_xy.shape[0] < 20:
            rospy.logwarn("too few points")
            return
        pca = PCA(n_components=1)
        pca.fit(points_xy)
        direction_vector = pca.components_[0]
        if direction_vector[0] < 0:
            direction_vector = -direction_vector
        yaw_angle = np.arctan2(direction_vector[1], direction_vector[0])
        self.yaw_average_queue.enqueue((yaw_angle, msg.header.stamp))

    def _cache_filtered_pointcloud(self):
        points_list = []
        for pointcloud_msg in self.pcloud_msg_list:
            gen = pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            points = np.array(list(gen))
            if points.shape[0] > 0:
                points_list.append(points)
        pcloud = np.vstack(points_list)
        pcloud = pcloud[pcloud[:, 0] < 1.0]
        pcloud = pcloud[pcloud[:, 0] > 0.3]
        pcloud = pcloud[pcloud[:, 1] > -0.6]
        pcloud = pcloud[pcloud[:, 1] < +0.6]
        pcloud = pcloud[pcloud[:, 2] > 0.6]
        pcloud = pcloud[pcloud[:, 2] < 1.0]
        self.pcloud_processed = pcloud

    def publish_debug_cloud(self, event) -> None:
        if self.pcloud_processed is None:
            return
        # create point cloud message
        header = self.pcloud_msg_list[0].header
        header.stamp = rospy.Time.now()
        header.frame_id = "base_footprint"
        pcloud_msg = pc2.create_cloud_xyz32(header, self.pcloud_processed)
        self.debug_cloud_publisher.publish(pcloud_msg)

    def publish_object_pose(self, event) -> None:
        if len(self.pcloud_msg_list) < self.n_collect:
            rospy.logwarn("Not enough pointclouds")
            return
        self._cache_filtered_pointcloud()

        pcloud = self.pcloud_processed
        pcloud = pcloud[pcloud[:, 2] > 0.75]
        pcloud = pcloud[pcloud[:, 2] < 0.95]

        assert pcloud is not None
        z_mean = np.mean(pcloud[:, 2])
        z_std = np.std(pcloud[:, 2])
        pcloud = pcloud[pcloud[:, 2] > z_mean + 0.5 * z_std]
        pcloud_xy = pcloud[:, :2]
        center_guess = np.mean(pcloud_xy, axis=0)

        self.center_average_queue.enqueue((center_guess, rospy.Time.now()))
        print(self.center_average_queue.stamps)
        if self.center_average_queue.is_steady() and self.center_average_queue.is_valid():
            if self.yaw_average_queue.is_steady(0.01) and self.yaw_average_queue.is_valid():
                center_mean = self.center_average_queue.get_average()

                co = Coordinates(np.hstack([center_mean, 0.80]))
                yaw_mean = self.yaw_average_queue.get_average()
                co.rotate(yaw_mean - np.pi * 0.5, "z")
                pose = CoordinateTransform.from_skrobot_coords(co).to_ros_pose()
                object_pose_msg = PoseStamped()
                object_pose_msg.header.frame_id = "base_footprint"
                object_pose_msg.header.stamp = rospy.Time.now()
                object_pose_msg.pose = pose
                self.object_pose_publisher.publish(object_pose_msg)
                rospy.loginfo("publish object pose: {}".format(pose))
                return
        rospy.logwarn("Not steady")
        return


if __name__ == "__main__":
    rospy.init_node("laser_scan_to_point_cloud")
    l2pc = LaserScanToPointCloud(n_collect=200)
    rospy.spin()
