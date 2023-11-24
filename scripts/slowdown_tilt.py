#!/usr/bin/env python

# this file fetched from https://github.com/PR2/pr2_calibration and slightly modified

PKG = "pr2_mechanism_controllers"

import roslib

roslib.load_manifest(PKG)


import rospy
from pr2_msgs.msg import LaserTrajCmd
from pr2_msgs.srv import *
from std_msgs import *

if __name__ == "__main__":
    cmd = LaserTrajCmd()
    controller = "laser_tilt_controller"
    cmd.header = rospy.Header(None, None, None)
    cmd.profile = "blended_linear"
    # cmd.pos      = [1.0, .26, -.26, -.7,   -.7,   -.26,   .26,   1.0, 1.0]
    d = 0.025
    # cmd.time     = [0.0, 0.4,  1.0, 1.1, 1.1+d,  1.2+d, 1.8+d, 2.2+d, 2.2+2*d]

    dur = 1.0
    pos_start = 0.7
    pos_end = 1.4
    cmd.position = [pos_start, pos_end, pos_start]
    cmd.time_from_start = [0.0, dur, 2 * dur]
    cmd.time_from_start = [rospy.Duration.from_sec(x) for x in cmd.time_from_start]
    cmd.max_velocity = 10
    cmd.max_acceleration = 30

    print("Sending Command to %s: " % controller)
    print("  Profile Type: %s" % cmd.profile)
    print("  Pos: %s " % ",".join(["%.3f" % x for x in cmd.position]))
    print("  Time: %s" % ",".join(["%.3f" % x.to_sec() for x in cmd.time_from_start]))
    print("  MaxRate: %f" % cmd.max_velocity)
    print("  MaxAccel: %f" % cmd.max_acceleration)

    rospy.wait_for_service(controller + "/set_traj_cmd")
    s = rospy.ServiceProxy(controller + "/set_traj_cmd", SetLaserTrajCmd)
    resp = s.call(SetLaserTrajCmdRequest(cmd))

    print("Command sent!")
    print("  Resposne: %f" % resp.start_time.to_sec())
