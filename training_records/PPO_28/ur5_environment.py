#!/usr/bin/python3
from push_gym.environments.base_environment import BasePybulletEnv
import pybullet as p
import numpy as np
import gym
import os
import math
import time
from push_gym.utils.custom_utils import euler_from_quat, quat_from_euler
UR5_URDF_PATH = "../robots/assets/ur5_with_gripper/ur5.urdf"
UR5_WORKSPACE_URDF_PATH = '../robots/assets/ur5_with_gripper/workspace.urdf'
PLANE_URDF_PATH = '../robots/assets/plane/plane.urdf'

class ArmSim(BasePybulletEnv):
    def __init__(self,render=False, shared_memory=False, hz=100, use_egl=False): # prev 240Hz
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        control_frequency = 100 #15.00
        self.step_size_fraction = 100
        print("self.step_size_fraction: " + str(self.step_size_fraction))
        self.per_step_iterations = int(self.step_size_fraction / control_frequency)

        #initialize arm base without gripper
        self._robot_tool_center = None
        self.robot_end_effector_link_index = None
        self.joint_combinations = None
        self._robot_joint_indices = None
        self.robot_arm_id = None
        self._gripper_id = None

        # Environment variables
        self._max_episode_steps = 100
        self.current_steps = 0

        #load table
        self._p.loadURDF(os.path.join(os.path.dirname(__file__), PLANE_URDF_PATH),[0, 0, -0.001])
        self.table = self._p.loadURDF(os.path.join(os.path.dirname(__file__), UR5_WORKSPACE_URDF_PATH),[0, 0, 0])
        self._p.changeVisualShape(self.table, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client_id)
        #load arm with griper
        #self._load_robot_with_gripper()
        # load arm with rod
        self._load_robot_with_rod()
        self.initial_pose = self.get_ik_joints([0.0, -0.198, 0.021], [np.pi, 0, 0], self._robot_tool_center)[:6]
        self._reset_arm(self.initial_pose)
        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.last_tool_tip_pose = self.tool_tip_pose
        print("Finish ArmSim init")

    def _load_robot_with_rod(self):
        """ Load simple rod from urdf, change color and set link ids """
        self.with_rod = True
        self.with_gripper = False
        robot_start_pose = [0, 0, 0]
        robot_start_orientation = self._p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE# | p.URDF_USE_SELF_COLLISION
        self.robot_arm_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), "../robots/assets/ur5_with_rod/ur5.urdf"),
                                             robot_start_pose, robot_start_orientation, useFixedBase=True, flags=flags)

        robot_joint_info = [self._p.getJointInfo(self.robot_arm_id, i) for i in
                            range(self._p.getNumJoints(self.robot_arm_id))]
        self._all_indices = [x[0] for x in robot_joint_info if x[2] == self._p.JOINT_REVOLUTE]
        all_names = [x[1] for x in robot_joint_info if x[2] == self._p.JOINT_REVOLUTE]
        self._robot_joint_indices = self._all_indices[:6]
        self._robot_joint_names = all_names[:6]

        for i in robot_joint_info:
            if "tool_tip" in str(i[1]):
                self._robot_tool_center = i[0]
            if "ee_fixed_joint" in str(i[1]):
                self.robot_end_effector_link_index = i[0]
            if "rod_fixed_joint" in str(i[1]):
                self.rod_index = i[0]

        self._p.changeVisualShape(self.robot_arm_id, self.rod_index, rgbaColor=[0, 0, 0, 1],
                                  physicsClientId=self.client_id)
        #for i in range(6):
        #    self._p.changeDynamics(self.robot_arm_id, i, lateralFriction=100.0, spinningFriction=100.0,
        #                           rollingFriction=0.1, restitution=0.00001, jointLimitForce=150)
        self._set_initial_arm_pose()
        self.step_simulation(self.per_step_iterations)

        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

    def _load_robot_with_gripper(self):
        """ Load robotiq 2f85 gripper from urdf and set link ids """
        self.with_gripper = True
        self.with_rod = False
        robot_start_orientation = self._p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE #| p.URDF_USE_SELF_COLLISION
        self.robot_arm_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), UR5_URDF_PATH), [0, 0, 0],
                                             robot_start_orientation, useFixedBase=True, flags=flags)

        robot_joint_info = [self._p.getJointInfo(self.robot_arm_id, i) for i in
                            range(self._p.getNumJoints(self.robot_arm_id))]
        print("robot_joint_info : " + str(robot_joint_info))
        self._all_indices = [x[0] for x in robot_joint_info if x[2] == self._p.JOINT_REVOLUTE]
        all_names = [x[1] for x in robot_joint_info if x[2] == self._p.JOINT_REVOLUTE]
        # store joint names and ids
        self._robot_joint_indices, self._gripper_joint_indices = self._all_indices[:6], self._all_indices[6:]
        self._robot_joint_names,  self._gripper_joint_names = all_names[:6], all_names[6:]
        #change dymaics for better grip
        for i in self._gripper_joint_indices:
            self._p.changeDynamics(self.robot_arm_id, i, lateralFriction=1.0, spinningFriction=1.0,
                                   rollingFriction=0.0001, restitution=0.00001, jointLimitForce=100)
        # store ee ids of arm and gripper
        for i in robot_joint_info:
            if "tool_tip" in str(i[1]):
                self._robot_tool_center = i[0]
            if "ee_fixed_joint" in str(i[1]):
                self.robot_end_effector_link_index = i[0]
        self.step_simulation(self.per_step_iterations)
        self._set_initial_arm_pose()
        self._set_initial_gripper_pose(self.robot_arm_id, self._gripper_joint_indices[0])
        self.step_simulation(self.per_step_iterations)

        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

    def _load_robot_single(self, with_arm):
        robot_start_pose = [0, 0, 0]
        robot_start_orientation = self._p.getQuaternionFromEuler([0, 0, 0])
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.robot_arm_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__),"../robots/assets/ur5/ur5.urdf"),
                                             robot_start_pose, robot_start_orientation, useFixedBase=True, flags=flags)

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [self._p.getJointInfo(self.robot_arm_id, i) for i in range(self._p.getNumJoints(self.robot_arm_id))]
        self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == self._p.JOINT_REVOLUTE]
        for i in robot_joint_info:
            if "tool_tip" in str(i[1]):
                self._robot_tool_center = i[0]
            if "ee_fixed_joint" in str(i[1]):
                self.robot_end_effector_link_index = i[0]
        # set robot to initial pose
        self._set_initial_arm_pose(with_arm)

        # load gripper and put it at ee
        self._robot_tool_offset = [0, 0, -0.05]
        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for i, joint_id in enumerate(self._robot_joint_indices[:7]):
            jointInfo = p.getJointInfo(self.robot_arm_id, joint_id, physicsClientId=self.client_id)

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_pose[i]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)
        return lower_limits, upper_limits, joint_ranges, rest_poses

    def _set_initial_arm_pose(self):
        self.initial_pose = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        for j in self._robot_joint_indices:
            self._p.resetJointState(self.robot_arm_id, j, self.initial_pose[j - 2])

    def _set_initial_gripper_pose(self, id, link):
        # set gripper to close
        angle = 0.715 - math.asin((0 - 0.010) / 0.1143)
        self._p.resetJointState(id, link, angle)


    def _reset_arm(self, joint_pose=None):
        #set arm pose if specified
        if joint_pose:
            for j in self._robot_joint_indices:
                self._p.resetJointState(self.robot_arm_id, j, joint_pose[j - 2])
            self._p.setJointMotorControlArray(self.robot_arm_id, self._robot_joint_indices,
                                              self._p.POSITION_CONTROL, joint_pose)
        # close Gripper
        if self.with_gripper:
            angle = 0.715 - math.asin((0 - 0.010) / 0.1143)
            self._p.setJointMotorControl2(self.robot_arm_id, self._gripper_joint_indices[0],
                                          self._p.POSITION_CONTROL, targetPosition=angle)

    def get_ik_joints(self, position, orientation, link):
        joints = self._p.calculateInverseKinematics(self.robot_arm_id, link,
                                                    position, self._p.getQuaternionFromEuler(orientation),
                                                    maxNumIterations=100, residualThreshold=1e-5)
        #joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return list( np.float32(joints))[:6]

    def change_gripper_opening(self, gripper_opening_length):
        if gripper_opening_length == -1: return
        angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        self._p.setJointMotorControl2(self._gripper_id, 1, self._p.POSITION_CONTROL,
                                      targetPosition=angle)

    def _move_to_position(self, action):
        action_lim = 3 if len(action) > 3 else 2
        state = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.last_gripper_image_points = self.utils.get_pos_in_image(state[0])
        pos_action = np.multiply(action[0:action_lim], self.arm_normalization_value)
        yaw_action = np.multiply(action[action_lim], self.yaw_normalization_value)

        goalEndEffectorPos = np.asarray(state[0]) + [pos_action[0], pos_action[1], 0]
        goalEndEffectorPos[2] = self.initial_tool_z
        self.current_gripper_image_points = self.utils.get_pos_in_image(goalEndEffectorPos)
        goalEndEffectorOri = np.array(euler_from_quat(state[1])) + np.array([0, 0, yaw_action])

        target_joint_states = np.array(self.get_ik_joints(goalEndEffectorPos,
                                                          goalEndEffectorOri,
                                                          self._robot_tool_center))
        return self._move_joints(target_joint_states)
    
    def get_jacobian(self, joint_positions, joint_velocities, tool_centre):
        joint_acceleration = [0,0,0,0,0,0]
        ee_index = self.robot_end_effector_link_index
        tool_centre = np.array([0,0,0])
        jacobian  = self._p.calculateJacobian(self.robot_arm_id, 
                                              ee_index, # index of the link to get the jacobian for
                                              tool_centre, #tool_centre,
                                               joint_positions.tolist(),
                                               joint_velocities.tolist(),
                                              joint_acceleration
                                              )
        jacobian_mat = np.array(jacobian)
        jacobian_mat = jacobian_mat.flatten()
        jacobian_output = np.array([[jacobian_mat[0], jacobian_mat[1], jacobian_mat[2], jacobian_mat[3], jacobian_mat[4], jacobian_mat[5]],
                                   [jacobian_mat[6], jacobian_mat[7], jacobian_mat[8], jacobian_mat[9], jacobian_mat[10], jacobian_mat[11]],
                                   [jacobian_mat[12], jacobian_mat[13], jacobian_mat[14], jacobian_mat[15], jacobian_mat[16], jacobian_mat[17]],
                                   [jacobian_mat[18], jacobian_mat[19], jacobian_mat[20], jacobian_mat[21], jacobian_mat[22], jacobian_mat[23]],
                                   [jacobian_mat[24], jacobian_mat[25], jacobian_mat[26], jacobian_mat[27], jacobian_mat[28], jacobian_mat[29]],
                                   [jacobian_mat[30], jacobian_mat[31], jacobian_mat[32], jacobian_mat[33], jacobian_mat[34], jacobian_mat[35]],           
                                   ])
        return jacobian_output
    
    def get_gravity_compensation(self, joint_positions, joint_velocities):
        desired_acc = [0,0,0,0,0,0]
        gravity_compensation = self._p.calculateInverseDynamics(self.robot_arm_id,
                                                                joint_positions,
                                                                joint_velocities, 
                                                                desired_acc)
        return gravity_compensation
    

    
    def update_joint_states(self):
        joint_positions = np.zeros(6)
        joint_velocities = np.zeros(6)
        for i in self._robot_joint_indices:
            joint_state = self._p.getJointState(self.robot_arm_id, i)
            joint_positions[i - self._robot_joint_indices[0]] = joint_state[0]
            joint_velocities[i - self._robot_joint_indices[0]] = joint_state[1]
        return joint_positions, joint_velocities

    def _move_joints(self, target_joint_states, till_timeout=False, max_timeout=5, speed=0.03):
        """
            Move robot tool (end-effector) to a specified pose
            @param action: Target position and orientation of the end-effector link
            action = [x,y,z, roll, pitch, yaw]
        """
        # Move Arm
        self._p.setJointMotorControlArray(
            self.robot_arm_id, self._robot_joint_indices,
            self._p.POSITION_CONTROL, target_joint_states,
            positionGains= speed * np.ones(len(self._robot_joint_indices)),
            #forces=[100, 100, 100, 100, 100, 100],
            #targetVelocities=np.zeros(len(self._robot_joint_indices)),
            #velocityGains=np.ones(len(self._robot_joint_indices))
        )

        if till_timeout:
            timeout_t0 = time.time()
            while True:
                # Keep moving until joints reach the target configuration
                current_joint_state = [
                    self._p.getJointState(self.robot_arm_id, i)[0]
                    for i in self._robot_joint_indices
                ]
                if all([
                    np.abs(
                        current_joint_state[i] - target_joint_states[i]) < 1e-3
                    for i in range(len(self._robot_joint_indices))
                ]):
                    break
                if time.time() - timeout_t0 > max_timeout:
                    self._reset_arm()
                    self.incorrect_image_count += 1
                    return False
                self.step_simulation(1)
            return True
        return True

    def _set_debug_parameter(self):
        #self.gripper_button_control = self._p.addUserDebugParameter("gripper_open", 1., 0., 1.)
        self.obj_reset_button_control = self._p.addUserDebugParameter("object_reset", 1., 0., 0.)
        self.gripper_opening_length_control = self._p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.085)
        self.position_control_group = []
        self.position_control_group.append(self._p.addUserDebugParameter('x', -3.14, 3.14, 0.11)) # 'x', -3.14, 3.14, -0.11
        self.position_control_group.append(self._p.addUserDebugParameter('y', -3.14, 3.14, -0.49)) #'y', -3.14, 3.14, 0.49
        self.position_control_group.append(self._p.addUserDebugParameter('z', 0.9, 1.3, 1.29))
        self.position_control_group.append(self._p.addUserDebugParameter('roll', -3.14, 3.14, 0))
        self.position_control_group.append(self._p.addUserDebugParameter('pitch', -3.14, 3.14, 3.14))#'pitch', -3.14, 3.14, 1.57
        self.position_control_group.append(self._p.addUserDebugParameter('yaw', -3.14, 3.14, 0))
