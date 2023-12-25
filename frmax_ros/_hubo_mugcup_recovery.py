import time
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import rospy
from skmp.constraint import BoxConst, CollFreeConst, ConfigPointConst
from skmp.kinematics import ArticulatedEndEffectorKinematicsMap
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.interface import Problem
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model.primitives import Box, Cylinder
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from tinyfk import RotationType

from frmax_ros.utils import CoordinateTransform


class PlanningCongig:
    joint_names: List[str]
    colfree_const_all: CollFreeConst
    colfree_const_table: CollFreeConst
    colfree_const_magcup: CollFreeConst
    box_const: BoxConst
    efkin: ArticulatedEndEffectorKinematicsMap
    table: Box
    back_side_panel: Box
    magcup: Cylinder

    def __init__(
        self,
        tf_obj_base: CoordinateTransform,
        pr2: PR2,
        consider_dummy_obstacle: bool = True,
        arm="larm",
        use_kanazawa: bool = False,
    ):
        pr2_plan_conf = PR2Config(control_arm=arm)
        joint_names = pr2_plan_conf._get_control_joint_names()

        colkin = pr2_plan_conf.get_collision_kin()
        table = Box([0.88, 0.75, 0.04], pos=[0.6, 0.0, 0.67], with_sdf=True)
        back_side_panel = Box([0.04, 0.75, 1.0], pos=[0.68, 0.0, 0.5], with_sdf=True)
        dummy_obstacle = Box([0.45, 0.6, 0.03], pos=[0.5, 0.0, 1.2], with_sdf=True)
        dummy_obstacle.visual_mesh.visual.face_colors = [255, 255, 255, 150]  # type: ignore
        if consider_dummy_obstacle:
            table_sdfs = [table.sdf, dummy_obstacle.sdf]
        else:
            table_sdfs = [table.sdf]

        if use_kanazawa:
            magcup = Cylinder(0.0525, 0.12, with_sdf=True)
        else:
            magcup = Cylinder(0.042, 0.095, with_sdf=True)
        magcup.visual_mesh.visual.face_colors = [255, 0, 0, 150]  # type: ignore
        magcup.newcoords(tf_obj_base.to_skrobot_coords())
        if use_kanazawa:
            magcup.translate([0, 0, -0.03])
        else:
            magcup.translate([0, 0, -0.03])

        colfree_const_all = CollFreeConst(colkin, UnionSDF(table_sdfs + [magcup.sdf]), pr2)
        colfree_const_table = CollFreeConst(colkin, UnionSDF(table_sdfs), pr2)
        colfree_const_magcup = CollFreeConst(colkin, magcup.sdf, pr2)

        box_const = pr2_plan_conf.get_box_const()
        box_const.lb -= 1e-3
        box_const.ub += 1e-3
        efkin = pr2_plan_conf.get_endeffector_kin(rot_type=RotationType.XYZW)

        self.joint_names = joint_names
        self.colfree_const_all = colfree_const_all
        self.colfree_const_table = colfree_const_table
        self.colfree_const_magcup = colfree_const_magcup
        self.box_const = box_const
        self.efkin = efkin
        self.table = table
        self.back_side_panel = back_side_panel
        self.magcup = magcup


class RecoveryMixIn:
    @lru_cache(maxsize=1)
    def recovery_ik_lib(self) -> List[np.ndarray]:
        rospy.loginfo("generating recovery ik lib")
        pr2 = PR2()
        arm = pr2.rarm
        end_coords = pr2.rarm_end_coords
        lib = []
        while len(lib) < 30:
            pr2.reset_manip_pose()
            x = np.random.uniform(0.3, 0.6)
            y = np.random.uniform(-0.5, 0.5)
            z = np.random.uniform(0.7, 0.9)
            co = Coordinates(pos=[x, y, z], rot=[0.0, 1.6, 0.5])

            out = arm.inverse_kinematics(co, link_list=arm.link_list, move_target=end_coords)
            if out is not False:
                lib.append(np.array(out))
        rospy.loginfo("finish generating recovery ik lib")
        return lib

    def recover(self) -> bool:
        self.initialize_robot()

        xy_desired = np.array([0.45, +0.15])
        yaw_desired = -np.pi * 0.65

        assert yaw_desired is None or (-np.pi < yaw_desired < 0.0)

        def create_pregrasp_and_grasp_poses(co_obj):
            co_pregrasp = co_obj.copy_worldcoords()
            co_pregrasp.translate([0.05, 0.0, 0.0])
            co_pregrasp.translate([0.0, 0.0, 0.13])
            co_pregrasp.rotate(+np.pi * 0.5, "y")
            co_pregrasp.rotate(-np.pi * 0.5, "x")

            co_grasp = co_pregrasp.copy_worldcoords()
            co_grasp.translate([0.08, 0.0, 0.0])
            return co_pregrasp, co_grasp

        try:
            tf_object_to_base = self.get_tf_object_to_base()
        except TimeoutError:
            return False

        co_obj = tf_object_to_base.to_skrobot_coords()
        co_pregrasp, co_grasp = create_pregrasp_and_grasp_poses(co_obj)

        xy_displacement = xy_desired - co_obj.worldpos()[:2]
        yaw_now = rpy_angle(co_obj.worldrot())[0][0]  # default
        yaw_displacement = yaw_desired - yaw_now
        # if np.abs(yaw_displacement) < 0.1 and np.linalg.norm(xy_displacement) < 0.1:
        #     rospy.logdebug("no need to recover")
        #     return True

        co_obj_desired = co_obj.copy_worldcoords()
        co_obj_desired.translate([xy_displacement[0], xy_displacement[1], 0.0], wrt="world")
        co_obj_desired.rotate(yaw_displacement, "z")
        co_predesired, co_desired = create_pregrasp_and_grasp_poses(co_obj_desired)

        plan_config = PlanningCongig(
            tf_object_to_base, self.pr2, consider_dummy_obstacle=False, arm="rarm"
        )
        q_init = get_robot_state(self.pr2, plan_config.joint_names)

        class _PlanningFailure(Exception):
            pass

        def solve_ik_with_collision_check(
            q_seed, co_target, colfree_const
        ) -> Tuple[np.ndarray, np.ndarray]:
            self.pr2.rarm.angle_vector(q_seed)
            ret = self.pr2.inverse_kinematics(
                co_target, link_list=self.pr2.rarm.link_list, move_target=self.pr2.rarm_end_coords
            )
            if ret is False:
                raise _PlanningFailure()
            q = self.pr2.rarm.angle_vector()
            av = self.pr2.angle_vector()
            if not colfree_const.is_valid(q):
                raise _PlanningFailure()
            return q, av

        def point_to_point_connection(q1, q2, colfree_const, n_wp=8) -> List[np.ndarray]:
            point_const = ConfigPointConst(q2)
            problem = Problem(q1, plan_config.box_const, point_const, colfree_const, None)
            ompl_config = OMPLSolverConfig(n_max_call=2000, simplify=True)
            ompl_solver = OMPLSolver.init(ompl_config).as_parallel_solver()
            ompl_solver.setup(problem)
            res = ompl_solver.solve()
            if res.traj is None:
                raise _PlanningFailure()
            q_list = list(res.traj.resample(n_wp).numpy())
            return q_list

        def get_avs(q_list: List[np.ndarray]) -> List[np.ndarray]:
            avs = []
            for q in q_list:
                set_robot_state(self.pr2, plan_config.joint_names, q)
                avs.append(self.pr2.angle_vector())
            return avs

        for q_seed_lib in self.recovery_ik_lib():

            try:
                # phase1
                q_pregrasp, av_pregrasp = solve_ik_with_collision_check(
                    q_seed_lib, co_pregrasp, plan_config.colfree_const_all
                )
                q_init_to_q_pregrasp = point_to_point_connection(
                    q_init, q_pregrasp, plan_config.colfree_const_all
                )
                q_grasp, av_grasp = solve_ik_with_collision_check(
                    q_pregrasp, co_grasp, plan_config.colfree_const_table
                )
                q_list_reach = q_init_to_q_pregrasp + [q_grasp]

                # phase2
                q_predesired, av_predesired = solve_ik_with_collision_check(
                    q_pregrasp, co_predesired, plan_config.colfree_const_table
                )
                q_pregrasp_to_q_predesired = point_to_point_connection(
                    q_pregrasp, q_predesired, plan_config.colfree_const_table
                )
                q_desired, av_desired = solve_ik_with_collision_check(
                    q_predesired, co_desired, plan_config.colfree_const_table
                )
                q_list_place = [q_pregrasp] + q_pregrasp_to_q_predesired + [q_desired]

                rospy.logdebug(
                    "(recover) successfully planned recovery trajectory. start moving to grasp position"
                )

                # FIXME: move back to home position (it already supposed to be at home position but sometimes not)
                # self.ri.angle_vector(self.av_home, time_scale=1.0, time=5.0)

                avs = get_avs(q_list_reach)
                times = [0.3] * (len(avs) - 2) + [1.0, 2.0]
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                self.ri.move_gripper("rarm", 0.0, effort=100)
                self.ri.wait_interpolation()
                time.sleep(2.0)
                gripper_pos = self.ri.gripper_states["rarm"].process_value
                rospy.logdebug("(recover) gripper_pos: {}".format(gripper_pos))
                if gripper_pos < 0.002:
                    rospy.logwarn("(recover) failed to grasp")
                    self.ri.move_gripper("rarm", 0.03)
                    self.ri.wait_interpolation()
                    q_list_back = [q_pregrasp] + q_init_to_q_pregrasp[::-1]
                    avs = get_avs(q_list_back)
                    times = [0.6] * len(avs)
                    self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                    self.ri.wait_interpolation()
                    return False

                rospy.logdebug("(recover) start placing")
                avs = get_avs(q_list_place)
                times = [0.6] * (len(avs) - 2) + [1.0, 2.0]
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                self.ri.move_gripper("rarm", 0.03)

                rospy.logdebug("(recover) start going back to home position")
                q_list_back = [q_predesired, q_pregrasp] + q_init_to_q_pregrasp[::-1]
                avs = get_avs(q_list_back)
                times = [0.3] * len(avs)
                self.ri.angle_vector_sequence(avs, times=times, time_scale=1.0)
                self.ri.wait_interpolation()
                rospy.logdebug("recovered procedure finished")
                return True
            except _PlanningFailure:
                continue
        return False
