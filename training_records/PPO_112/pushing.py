#!/usr/bin/python3
from push_gym.environments.ur5_environment import ArmSim
from push_gym.utils.utils import Utils
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import gym
import time
from push_gym.utils.PDController import PDController
import tensorflow as tf

from push_gym.utils.custom_utils import  get_point, get_config, reset_sim
import threading


class Pushing(ArmSim):
    def __init__(self,render=False, shared_memory=False, hz=100, use_egl=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        # World variables
        self.world_id = None
        self._workspace_bounds = np.array([[-0.35, 0.35], [-0.2, -0.75], [0.021, 0.021]])
        self._collision_bounds = np.array([[-0.60, 0.60], [-1., 0], [0., 1.]])
        self.utils = Utils(self, self._p)

        self.max_vel = -999999999
        self.max_reward = -999999999

        self.target_distance_reach = 1 # 1cm

        # init training logging data
        self.mean_total_reward_cumul = 0
        self.mean_disp_reward_cumul = 0
        self.mean_dist_reward_cumul = 0

        self.mean_ee_object_speed_cumul = 0
        self.mean_ee_object_dist_cumul = 0

        self.mean_object_target_speed_cumul = 0
        self.mean_object_target_dist_cumul = 0

        self.mean_normal_total_force_cumul = 0
        self.mean_normal_residual_force_cumul = 0
        self.mean_normal_impedance_force_cumul = 0


        self.previous_debris_target_quad_dist = 0
        self.previous_time = 0
        self.delta_time = 0
        #Generate pushable object
        self.pushing_object_id = -1
        self.pushing_object_id = self.utils.generate_obj()
        self.current_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        self.last_obj_conf = self.current_obj_conf
        #Generate first random goal
        self.goal_obj_conf = self.utils.generate_random_goal(self.pushing_object_id)
        self.utils.debug_gui_target(np.asarray(self.goal_obj_conf[0]))
        #self.utils.debug_gui_target(np.asarray(self.goal_obj_conf[0]))
        #self.get_relative_coodinates_in_obj_frame()
        self.step_simulation(self.per_step_iterations)
        #save initial world state
        self.world_id = self._p.saveState()
        self.set_important_variables()


    def setup_config(self, cfg):
        self.cfg = cfg
        # setting up PD controller
        self.motion_controller = PDController(self._p)

        linear_proportional_gain = np.array(self.cfg["RL"]["controller"]["linear_proportional_gain"])
        angular_proportional_gain = np.array(self.cfg["RL"]["controller"]["angular_proportional_gain"])
        linear_derivative_gain = np.array(self.cfg["RL"]["controller"]["linear_derivative_gain"])
        angular_derivative_gain = np.array([self.cfg["RL"]["controller"]["angular_derivative_gain"]])
        self.motion_controller.setProportionalGain(linear_proportional_gain, angular_proportional_gain)
        self.motion_controller.setDerivativeGain(linear_derivative_gain, angular_derivative_gain)
    
        self.initial_arm_position = self.cfg["general"]["arm_position"]
        self._max_episode_steps = self.cfg["RL"]["env"]["max_episode_length"]
        #define action and observation space for RL
        self._set_action_space()
        self._set_observation_space()

    def set_important_variables(self):
        self.thread_running = False
        #RL Variables
        self.has_contact = 0
        self.n_stack = 1
        self.current_run = 0
        self.current_steps = 0

        self.target_max_dist = 2
        #Debug Variables
        self.Debug = False
        self.single_step_training_debug = False
        self.path_debug_lines = []
        self.local_window_debug_lines = []
        self.plt_fig = None
        self.plt_obj = None


    def _set_observation_space(self):
        # observation space definition
        # tool orientation : 4 (quaternion (wxyz))
        # tool angular velocity: 3 (xyz)
        # tool - debris relative position : 3 (xyz)
        # tool - debris relative velocity : 3 (xyz)
        # tool - target relative position : 3 (xyz)
        # tool - target relative velocity : 3 (xyz)
        # Total : 19
        low_boundaries = np.full(self.cfg["RL"]["env"]["num_observations"], -np.inf)
        high_boundaries = np.full(self.cfg["RL"]["env"]["num_observations"], np.inf)
        self.observation_space = gym.spaces.Box(
                                low = low_boundaries,
                                high = high_boundaries,
                                dtype= np.float32,
                            ) 
        return




    def _set_action_space(self):
        low_boundaries = np.full(self.cfg["RL"]["env"]["num_action"], -1)
        high_boundaries = np.full(self.cfg["RL"]["env"]["num_action"], 1)
        self.action_space = gym.spaces.Box(
                                            low=low_boundaries,
                                            high= high_boundaries,
                                            dtype=np.float32
                                        )
    
    def _get_observation(self):
        # robot state observations
        self.robot_joint_pos, self.robot_joint_vel = super().update_joint_states()
        self.jacobian = super().get_jacobian(self.robot_joint_pos, self.robot_joint_vel, self._robot_tool_center)
        # robot end effector cartesian position and velocities
        ee_data = np.asarray(self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center))
        self.ee_position = np.asarray(ee_data[0])
        self.ee_orientation = np.array(ee_data[1])
        self.ee_velocities = self.get_end_effector_velocity(self.jacobian, self.robot_joint_vel)
        self.ee_lin_velocities = np.array([self.ee_velocities[0], self.ee_velocities[1], self.ee_velocities[2]])
        self.ee_ang_velocities = np.array([self.ee_velocities[3], self.ee_velocities[4], self.ee_velocities[5]])

        # object world state
        self.obj_lin_pos, _= self._p.getBasePositionAndOrientation(self.pushing_object_id)#, self.client_id)
        self.current_obj__lin_vel, _ = self._p.getBaseVelocity(self.pushing_object_id)#, self.client_id)
        self.object_position = np.asarray(self.obj_lin_pos)
        self.object_velocity = np.asarray(self.current_obj__lin_vel)

        # goal state , we just look at the position and the target is not moving
        goal_position = np.array(self.goal_obj_conf[0])

        # updates relative distance and velocities
        self.tool_object_pos, self.tool_object_vel = self.get_relative_state(self.object_position, 
                                                                            self.object_velocity,
                                                                            self.ee_position,
                                                                            self.ee_lin_velocities
                                                                            )

        self.tool_target_pos, self.tool_target_vel = self.get_relative_state(goal_position,# to define
                                                                            np.array([0,0,0]),
                                                                            self.ee_position,
                                                                            self.ee_lin_velocities
                                                                            )
        
        self.debris_target_pos, debris_target_vel = self.get_relative_state(goal_position,# to define
                                                            np.array([0,0,0]),
                                                            self.object_position, 
                                                            self.object_velocity
                                                            )
        
        # data for reward function
        self.debris_target_dist = pow(self.debris_target_pos[0] * 10,2) + pow(self.debris_target_pos[1] * 10,2)
        self.debris_target_dist_delta = self.previous_debris_target_quad_dist - self.debris_target_dist 

        self.current_time = time.time()
        self.delta_time = self.current_time - self.previous_time

        self.debris_target_vel = self.debris_target_dist_delta / self.delta_time

        self.previous_time = self.current_time
        self.previous_debris_target_quad_dist = self.debris_target_dist

        # update input for impedance (PD)
                # setting up observation for PD controller
        self.motion_controller.setPositionsObservation(linear_position_observation= self.ee_position,
                                                angular_position_observation=  self.ee_orientation)

        self.motion_controller.setVelocitiesObservation(linear_velocity_observation = self.ee_lin_velocities, 
                                                    angular_velocity_observation = self.object_velocity)
        
        # data for training log

        self.mean_ee_object_speed_cumul = self.mean_ee_object_speed_cumul +  np.linalg.norm(self.tool_object_vel)
        self.mean_ee_object_dist_cumul = self.mean_ee_object_dist_cumul + np.linalg.norm(self.tool_object_pos)

        self.mean_object_target_speed_cumul = self.mean_object_target_speed_cumul + np.linalg.norm(self.debris_target_vel)
        self.mean_object_target_dist_cumul = self.mean_object_target_dist_cumul + np.linalg.norm(self.debris_target_pos)
        
        obs = np.concatenate(
            [
                self.ee_orientation,
                self.ee_ang_velocities,
                self.tool_object_pos,
                self.tool_object_vel,
                self.tool_target_pos,
                self.tool_target_vel,
                self.robot_joint_pos
            ]
        )
        return obs


 

    def reset(self):
        self.current_steps = 0
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        self.reset_task()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def reset_task(self):
        self.is_reset = True
        # reset sim to initial status
        reset_sim(self.world_id, self._p)
        # ramdomly reset the position of the object
        self.utils.reset_obj(self.pushing_object_id)
        self.step_simulation(self.per_step_iterations)
        self.current_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        self.last_obj_conf = self.current_obj_conf
        # randomly reset the position of the goal
        self.goal_obj_conf = self.utils.generate_random_goal(self.pushing_object_id, 2, max_distance=0.6, min_distance=0.3) #minmax
        self.utils.debug_gui_target(np.asarray(self.goal_obj_conf[0]))
        self._reset_arm()
        # reset object init and goal position and calculate initial distances
        self.utils.reset_obj(self.pushing_object_id)
        self.step_simulation(self.per_step_iterations)
        self.current_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        self.last_obj_conf = self.current_obj_conf
        # randomly reset the position of the arm
         #reset arm pose
        x_trans, y_trans = np.random.choice([-1, 1], 2) * np.random.uniform(0.08, 0.1, 2)
        arm_pose = np.asarray(get_point(self.pushing_object_id, self._p, self.client_id)) - [x_trans, y_trans, 0.01]
        target_joint_states = self.get_ik_joints(arm_pose, [np.pi, 0, 0], self._robot_tool_center)[:6]
        self._reset_arm(target_joint_states)

        self.f = [0,0,0,0,0,0]

        self._p.setJointMotorControlArray(
                self.robot_arm_id, 
                jointIndices = self._robot_joint_indices,
                controlMode = self._p.VELOCITY_CONTROL, 
                targetVelocities = [0,0,0,0,0,0],
                forces = self.f,
                physicsClientId = self.client_id
        )

        # step simulation twice to let objects fall down
        self.step_simulation(self.per_step_iterations)
        self.step_simulation(self.per_step_iterations)

        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.initial_tool_z = self.current_obj_conf[0][2]
        self.last_tool_tip_pose = self.tool_tip_pose

        self.end_of_time = False



    def get_reward(self):
        ''' Calculate reward for current step
        input:
            target_reached: True if object to goal dist is under given threshold
            collision_detected: True if collision is detected
            with_sparse_reward: True if no intermediate rewards are given
            with_shortest: True if Theta star distance schell be used as distance, False if euclidean distance shell be used
        output: reward of current step corresponding to the current situation

        reward structure: 
        reward 1 : reward the object displacement towards the goal (bell shaped function)
        reward 2 : reward the object position towards the goal (bell shaped function)
        '''
        object_velocity_target = self.cfg["RL"]["reward"]["speed_target"] #10 cm/s
        total_reward = 0
        reward_disp_weight = self.cfg["RL"]["reward"]["object_goal_disp_weight"]
        reward_dist_weight = self.cfg["RL"]["reward"]["object_goal_dist_weight"]
        reward_displacement_tolerance = self.cfg["RL"]["reward"]["object_goal_speed_tolerance"] / 2
        reward_distance_tolerance = self.cfg["RL"]["reward"]["object_goal_dist_tolerance"]
        #self.max_vel = max(self.max_vel,self.debris_target_vel )
        debris_target_vel_scaled = self.debris_target_vel
        reward_displacement =  reward_disp_weight * np.exp((-pow(debris_target_vel_scaled - object_velocity_target,2))/(2*pow(reward_displacement_tolerance,2)))
        reward_distance = reward_dist_weight * (np.exp((-pow(self.debris_target_dist,2))/(2*pow(reward_distance_tolerance,2))))

        total_reward = reward_displacement + reward_distance
        self.mean_total_reward_cumul = self.mean_total_reward_cumul + total_reward
        self.mean_disp_reward_cumul = self.mean_disp_reward_cumul + reward_displacement
        self.mean_dist_reward_cumul = self.mean_dist_reward_cumul + reward_distance

        return total_reward
    
    def is_done(self):
        done = False
        # end of episode
        if self.current_steps > (self._max_episode_steps - 1): done = True
        # Object reached the target
        print("self.debris_target_dist: " + str(self.debris_target_dist))
        if self.debris_target_dist < 1: done = True # self.target_distance_reach: done = True
        return done

    def _robot_impedance_control(self):
        from push_gym.utils.transformations import quaternion_from_euler
        # update motion controller command 
        # 1. pd linear control
        lin_pos_command = self.object_position
        lin_pos_command[2] = 0
        lin_vel_command = self.object_velocity
        self.motion_controller.setLinearPositionsCommand(lin_pos_command)
        self.motion_controller.setLinearVelocitiesCommand(lin_vel_command)
        self.forces = self.motion_controller.linearControlRobot()

        # 2. pd angular control
        ang_pos_command = quaternion_from_euler(np.pi,0,(np.pi/2))
        self.motion_controller.setAngularPositionsCommand(ang_pos_command)
        ang_vel_command = np.array([0,0,0])
        self.motion_controller.setAngularVelocitiesCommand(ang_vel_command)
        self.moments = self.motion_controller.angularControlQuaternion()

        compensation_torques = np.asarray(super().get_gravity_compensation(self.robot_joint_pos.tolist(), self.robot_joint_vel.tolist()))


        self.total_forces = self.forces
        self.forces_moments = np.array([self.total_forces[0], self.total_forces[1], self.total_forces[2],
                                        self.moments[0][0], self.moments[0][1], self.moments[0][2]])
        self.joint_torque_commands =  compensation_torques + np.matmul(self.jacobian.transpose(), self.forces_moments)
        
        # apply force moment to robotic manipulator

        max_torque = np.array([150,150,150,150,150,150])
        for i in range(6):
            if self.joint_torque_commands[i] > max_torque[i]:
                self.joint_torque_commands[i] = max_torque[i]
            if self.joint_torque_commands[i] < -max_torque[i]:
                self.joint_torque_commands[i] = -max_torque[i]

        self._p.setJointMotorControlArray(
                        self.robot_arm_id, 
                        jointIndices = self._robot_joint_indices,
                        controlMode = self._p.TORQUE_CONTROL, 
                        forces = self.joint_torque_commands,
                        physicsClientId = self.client_id
        )
        return

    def _apply_action(self, action):
        from push_gym.utils.transformations import quaternion_from_euler
        # RL action breackdown
        x_force = action[0] * self.cfg["RL"]["action"]["x_force_scale"]
        y_force = action[1] * self.cfg["RL"]["action"]["y_force_scale"]
        z_force = action[2] * self.cfg["RL"]["action"]["z_force_scale"]
        yaw = action[3] * np.pi * self.cfg["RL"]["action"]["yaw_scale"]
        # update motion controller command 
        # 1. pd linear control
        lin_pos_command = self.object_position
        lin_pos_command[2] = 0
        lin_vel_command = self.object_velocity
        self.motion_controller.setLinearPositionsCommand(lin_pos_command)
        self.motion_controller.setLinearVelocitiesCommand(lin_vel_command)
        self.forces = self.motion_controller.linearControlRobot()

        # 2. pd angular control
        ang_pos_command = quaternion_from_euler(np.pi,0,(np.pi/2))
        self.motion_controller.setAngularPositionsCommand(ang_pos_command)
        ang_vel_command = np.array([0,0,0])
        self.motion_controller.setAngularVelocitiesCommand(ang_vel_command)
        self.moments = self.motion_controller.angularControlQuaternion()

        compensation_torques = np.asarray(super().get_gravity_compensation(self.robot_joint_pos.tolist(), self.robot_joint_vel.tolist()))

        self.residual_forces = np.array([x_force, y_force, z_force])

        self.total_forces = self.forces + self.residual_forces
        self.forces_moments = np.array([self.total_forces[0], self.total_forces[1], self.total_forces[2],
                                        self.moments[0][0], self.moments[0][1], self.moments[0][2]])
        self.joint_torque_commands =  compensation_torques + np.matmul(self.jacobian.transpose(), self.forces_moments)
        
        # apply force moment to robotic manipulator

        max_torque = np.array([150,150,150,150,150,150])
        for i in range(6):
            if self.joint_torque_commands[i] > max_torque[i]:
                self.joint_torque_commands[i] = max_torque[i]
            if self.joint_torque_commands[i] < -max_torque[i]:
                self.joint_torque_commands[i] = -max_torque[i]

        self.mean_normal_total_force_cumul =  self.mean_normal_total_force_cumul + np.sqrt(pow(self.total_forces[0],2) + pow(self.total_forces[1],2) + pow(self.total_forces[2],2))
        self.mean_normal_residual_force_cumul =  self.mean_normal_residual_force_cumul  + np.sqrt(pow(self.residual_forces[0],2) + pow(self.residual_forces[1],2) + pow(self.residual_forces[2],2))
        self.mean_normal_impedance_force_cumul =  self.mean_normal_impedance_force_cumul + np.sqrt(pow(self.forces[0],2) + pow(self.forces[1],2) + pow(self.forces[2],2))


        self._p.setJointMotorControlArray(
                        self.robot_arm_id, 
                        jointIndices = self._robot_joint_indices,
                        controlMode = self._p.TORQUE_CONTROL, 
                        forces = self.joint_torque_commands,
                        physicsClientId = self.client_id
        )



    def signal_user_input(self):
        self.thread_running = True
        i = input()
        self.Debug = not(self.Debug)
        self.thread_running = False

    def step(self, action):

        self.current_steps += 1
        if not self.thread_running:
            threading.Thread(target=self.signal_user_input).start()

        for i in range(int(self.per_step_iterations)):   
            self._apply_action(action)
            self._p.stepSimulation(physicsClientId=self.client_id)
            #self.step_simulation(self.per_step_iterations)
        
        observations = self._get_observation()
        info = {}
        done = False

        rewards = self.get_reward()

        done = self.is_done()
        return observations, rewards, done, info


    ###########################
    '''Environment functions'''
    ###########################

    def get_end_effector_velocity(self, jacobian, joint_velocities):
        ee_velocity = np.zeros(6)
        ee_velocity = np.matmul(jacobian , joint_velocities)
        return ee_velocity

    def get_relative_state(self, position_a, velocity_a, position_b, velocity_b):
        relative_position = position_a - position_b
        relative_velocity = velocity_a - velocity_b
        return relative_position, relative_velocity
    
    def get_data_training_log(self, n_steps:int): 

        output = np.zeros(10)

        output[0] = self.mean_total_reward_cumul / n_steps
        output[1] = self.mean_disp_reward_cumul / n_steps
        output[2] = self.mean_dist_reward_cumul / n_steps

        output[3] = self.mean_ee_object_speed_cumul / n_steps
        output[4] = self.mean_ee_object_dist_cumul / n_steps 

        output[5] = self.mean_object_target_speed_cumul / n_steps 
        output[6] = self.mean_object_target_dist_cumul / n_steps 

        output[7] = self.mean_normal_total_force_cumul / n_steps 
        output[8] = self.mean_normal_residual_force_cumul / n_steps 
        output[9] = self.mean_normal_impedance_force_cumul / n_steps 

        return output
    
    def reset_logging_data(self):
        self.mean_total_reward_cumul = 0
        self.mean_disp_reward_cumul = 0
        self.mean_dist_reward_cumul = 0

        self.mean_ee_object_speed_cumul = 0
        self.mean_ee_object_dist_cumul = 0

        self.mean_object_target_speed_cumul = 0
        self.mean_object_target_dist_cumul = 0

        self.mean_normal_total_force_cumul = 0
        self.mean_normal_residual_force_cumul = 0
        self.mean_normal_impedance_force_cumul = 0