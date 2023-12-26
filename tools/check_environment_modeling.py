import time
from pathlib import Path

import numpy as np
import rospkg
import rospy
import trimesh
from skrobot.model.primitives import Axis, Box, MeshLink, PointCloudLink
from skrobot.viewers import TrimeshSceneViewer

from frmax_ros.node import ObjectPoseProvider, PointCloudProvider
from frmax_ros.utils import CoordinateTransform

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("frmax_ros"))
mug_model_path = pkg_path / "model" / "hubolab_mug.stl"
mesh = trimesh.load_mesh(mug_model_path)
mesh_link = MeshLink(mesh, with_sdf=True)

rospy.init_node("check_environment_modeling")
object_pose_provider = ObjectPoseProvider()
point_cloud_provider = PointCloudProvider()
tf_april_to_base = object_pose_provider.get_tf_object_to_base()
pcloud = point_cloud_provider.get_point_cloud()

tf_object_to_april = CoordinateTransform(np.array([0.013, -0.004, -0.095]), np.eye(3))
tf_object_to_base = tf_object_to_april * tf_april_to_base

mesh_link.newcoords(tf_object_to_base.to_skrobot_coords())

h = 0.72
table = Box(extents=[0.5, 0.75, h], with_sdf=True)
table.translate([0.5, 0, 0.5 * h])

pcloud = PointCloudLink(pcloud)
v = TrimeshSceneViewer()
axis = Axis.from_coords(tf_object_to_base.to_skrobot_coords())
v.add(pcloud)
v.add(mesh_link)
v.add(table)
v.add(axis)
v.show()
import time

time.sleep(1000)
