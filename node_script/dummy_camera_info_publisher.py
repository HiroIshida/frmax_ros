#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo, Image


class DummyInfoPublisher:
    def __init__(self):
        # remote indicates the images are uncompressed
        self.sub_info = rospy.Subscriber(
            "/kinect_head/rgb/camera_info", CameraInfo, self.callback_info
        )
        self.sub_img = rospy.Subscriber(
            "/remote/kinect_head/rgb/image_rect_color", Image, self.callback_image
        )
        self.pub_info = rospy.Publisher(
            "/remote/kinect_head/rgb/camera_info", CameraInfo, queue_size=100
        )
        self.info = None

    def callback_info(self, msg):
        self.info = msg
        self.sub_info.unregister()  # one shot

    def callback_image(self, msg: Image):
        if self.info is None:
            return
        self.info.header = msg.header
        self.pub_info.publish(self.info)
        rospy.loginfo("published camera info")


if __name__ == "__main__":
    rospy.init_node("dummy_camera_info_publisher")
    dummy_info_publisher = DummyInfoPublisher()
    rospy.spin()
