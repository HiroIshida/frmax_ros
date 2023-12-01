#!/usr/bin/env python
from typing import List

import numpy as np
import rospy
from executor import create_trajectory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rospy import Publisher, Subscriber
from skrobot.coordinates import Coordinates
from utils import CoordinateTransform, chain_transform


class TrajectoryPublisher:
    plaener_traj: List[np.ndarray]
    sub: Subscriber
    pub: Publisher

    def __init__(self):
        planer_traj = create_trajectory(np.zeros(21))
        self.planer_traj = planer_traj
        self.sub = Subscriber(
            "/mugcup_pose", PoseStamped, self.callback
        )  # TODO: change topic name to more general
        self.pub = Publisher("/grasping_trajectory", RosPath, queue_size=1, latch=True)

    def callback(self, object_pose: PoseStamped):
        tf_obj_base = CoordinateTransform.from_ros_pose(
            object_pose.pose, "object", "base_footprint"
        )

        def to_transform(x, y, yaw, tf_obj_base) -> CoordinateTransform:
            # TODO/FIXME: almost same as `to_transform` in `executor.py`

            rot: float = -np.pi * 0.5  # TODO: fuck this

            # because in execution we don't know
            x_error = 0.0
            y_error = 0.0
            yaw_error = 0.0

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
                tf_obj_base,  # object -> base
            )
            return tf_reach_base

        ros_path = RosPath()
        ros_path.header.stamp = object_pose.header.stamp
        ros_path.header.frame_id = "base_footprint"
        ros_path.poses = []
        for planer_pose in self.planer_traj:
            tf_reach_base = to_transform(
                planer_pose[0], planer_pose[1], planer_pose[2], tf_obj_base
            )
            pose = tf_reach_base.to_ros_pose()
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = object_pose.header.stamp
            pose_stamped.header.frame_id = "base_footprint"
            pose_stamped.pose = pose
            ros_path.poses.append(pose_stamped)
        self.pub.publish(ros_path)
        rospy.loginfo("published trajectory")


if __name__ == "__main__":
    rospy.init_node("trajectory_publisher")
    TrajectoryPublisher()
    rospy.spin()
