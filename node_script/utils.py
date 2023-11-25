import functools
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    matrix2quaternion,
    quaternion2matrix,
    wxyz2xyzw,
    xyzw2wxyz,
)

try:
    from skrobot.interfaces.ros.base import ROSRobotInterfaceBase
except ModuleNotFoundError:
    ROSRobotInterfaceBase = None


def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


RosTransform = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


@dataclass
class CoordinateTransform:
    trans: np.ndarray
    rot: np.ndarray
    src: Optional[str] = None
    dest: Optional[str] = None

    def __call__(self, vec_src: np.ndarray) -> np.ndarray:
        if vec_src.ndim == 1:
            return self.rot.dot(vec_src) + self.trans
        elif vec_src.ndim == 2:
            return self.rot.dot(vec_src.T).T + self.trans
        else:
            assert False

    def inverse(self) -> "CoordinateTransform":
        rot_new = self.rot.T
        trans_new = -rot_new.dot(self.trans)
        return CoordinateTransform(trans_new, rot_new, self.dest, self.src)

    @classmethod
    def from_ros_transform(
        cls, ros_tf: RosTransform, src: Optional[str] = None, dest: Optional[str] = None
    ):
        trans_tuple, quat_xyzw_tuple = ros_tf
        trans = np.array(trans_tuple)
        rot = quaternion2matrix(xyzw2wxyz(quat_xyzw_tuple))
        return cls(trans, rot, src, dest)

    def to_ros_transform(
        self, src: Optional[str] = None, dest: Optional[str] = None
    ) -> RosTransform:
        quat = wxyz2xyzw(matrix2quaternion(self.rot))
        trans = tuple(self.trans.tolist())
        quat = tuple(quat.tolist())
        return trans, quat  # type: ignore

    @classmethod
    def from_ros_pose(cls, pose: Pose, src: Optional[str] = None, dest: Optional[str] = None):
        position = pose.position
        quat = pose.orientation
        trans = np.array([position.x, position.y, position.z])
        rot = quaternion2matrix([quat.w, quat.x, quat.y, quat.z])
        return cls(trans, rot, src, dest)

    def to_ros_pose(self) -> Pose:
        quat = wxyz2xyzw(matrix2quaternion(self.rot))
        pose = Pose(Point(*self.trans), Quaternion(*quat))
        return pose

    @classmethod
    def from_skrobot_coords(
        cls, coords: Coordinates, src: Optional[str] = None, dest: Optional[str] = None
    ):
        return cls(coords.worldpos(), coords.worldrot(), src, dest)

    def to_skrobot_coords(self) -> Coordinates:
        return Coordinates(self.trans, self.rot)


def chain_transform(
    tf_a2b: CoordinateTransform, tf_b2c: CoordinateTransform
) -> CoordinateTransform:
    if tf_a2b.dest is not None and tf_b2c.src is not None:
        assert tf_a2b.dest == tf_b2c.src, "{} does not match {}".format(tf_a2b.dest, tf_b2c.src)

    trans_a2c = tf_b2c.trans + tf_b2c.rot.dot(tf_a2b.trans)
    rot_a2c = tf_b2c.rot.dot(tf_a2b.rot)

    src_new = tf_a2b.src
    dest_new = tf_b2c.dest
    return CoordinateTransform(trans_a2c, rot_a2c, src_new, dest_new)
