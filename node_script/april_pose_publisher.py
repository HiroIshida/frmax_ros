#!/usr/bin/env python
from typing import List, Tuple

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from skrobot.coordinates.math import (
    quaternion2rpy,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)


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


class AprilPosePublisher:
    queue: AverageQueue

    def __init__(self):
        self.pub_filtered = rospy.Publisher("object_pose_filtered", PoseStamped, queue_size=10)
        self.pub = rospy.Publisher("object_pose", PoseStamped, queue_size=10)
        self.listener = tf.TransformListener()
        self.queue = AverageQueue(max_size=10)

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

    def publish_pose(self):
        try:
            target_frame = "base_footprint"
            source_frame = "object"
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

            # we care only yaw and know that othere angles are 0
            ypr = quaternion2rpy(xyzw2wxyz(rot))[0]
            xyztheta = np.hstack([trans, ypr[0]])
            self.queue.enqueue((xyztheta, rospy.Time.now()))

            pose = self.xyztheta_to_pose(xyztheta, target_frame)
            self.pub.publish(pose)

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
            self.pub_filtered.publish(pose_filtered)
            rospy.loginfo("Published pose_filtered: {}".format(pose_filtered))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("TF Exception")


def main():
    rospy.init_node("pose_publisher_node")
    app = AprilPosePublisher()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        app.publish_pose()
        rate.sleep()


if __name__ == "__main__":
    main()
