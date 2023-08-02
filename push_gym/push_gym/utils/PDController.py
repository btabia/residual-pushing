import numpy as np
import math


def get_norm(quaternion): 
    return np.sqrt(pow(quaternion[0],2) + pow(quaternion[1],2)+ pow(quaternion[2],2)+ pow(quaternion[3],2))

def normalise(quaternion):
    norm = get_norm(quaternion)
    output = np.array([quaternion[0]/norm, quaternion[1]/norm, quaternion[2]/norm, quaternion[3]/norm])
    return output

#def quaternion_inverse(quaternion):
#    #inv(q) = conj(q) / (norm(q) * norm(q))
    # set up the conjubgate
#    conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
#    # calculate the norm
#    norm = get_norm(conjugate)

 #   quaternion_inverse = np.array([1.0,0.0,0.0,0.0])
 #   for i in range(4):
 #       quaternion_inverse[i] = conjugate[i] / (norm * norm)

   # return quaternion_inverse

class PDController:
    def __init__(self, _p):
        self._p = _p
        self.linear_P_gain = np.array([1,1,1])
        self.linear_D_gain = np.array([0,0,0])

        self.angular_P_gain = np.array([1,1,1])
        self.angular_D_gain = np.array([0,0,0])
        self.yaw_orientation = 0
        self.moment_limits = np.zeros(3)
        self.pd_force_limit = np.zeros(3)

    def setAngularVelocitiesCommand(self, angular_velocity_command):
        self.angular_velocity_command = angular_velocity_command
    
    def setLinearVelocitiesCommand(self, linear_velocity_command):
        self.linear_velocity_command = linear_velocity_command
    
    def setAngularPositionsCommand(self, angular_position_command):
        self.angular_position_command = angular_position_command
    
    def setLinearPositionsCommand(self, linear_position_command):
        self.linear_position_command = linear_position_command


    def setVelocitiesObservation(self, linear_velocity_observation, angular_velocity_observation):
        self.linear_velocity_observation = linear_velocity_observation
        self.angular_velocity_observation = angular_velocity_observation

    def setPositionsObservation(self, linear_position_observation, angular_position_observation):
        self.linear_position_observation = linear_position_observation
        self.angular_position_observation = angular_position_observation
    
    def setProportionalGain(self, linear_P_gain, angular_P_gain):
        self.linear_P_gain = linear_P_gain
        self.angular_P_gain = angular_P_gain
    
    def setDerivativeGain(self, linear_D_gain, angular_D_gain):
        self.linear_D_gain = linear_D_gain
        self.angular_D_gain = angular_D_gain

    def getYawOrientation(self):
        return self.yaw_orientation
    
    def setForceLimits(self, force_limits):
        self.pd_force_limit = force_limits
    
    def setMomentLimits(self, moment_limits):
        self.moment_limits = moment_limits
    
    def linearControl(self):
        linear_position_error = self.linear_position_command - self.linear_position_observation
        linear_velocity_error = self.linear_velocity_command - self.linear_velocity_observation
        linear_position_output = linear_position_error * self.linear_P_gain
        linear_velocity_output = linear_velocity_error * self.linear_D_gain
        force_output = linear_position_output + linear_velocity_output

        for i in range(3):
            if force_output[i] > self.pd_force_limit[i]: 
                force_output[i] = self.pd_force_limit[i]
            elif force_output[i] < -self.pd_force_limit[i]:
                force_output[i] = -self.pd_force_limit[i]

        return force_output
    
    def linearControlRobot(self):
        linear_position_error = self.linear_position_command - self.linear_position_observation
        linear_velocity_error = self.linear_velocity_command - self.linear_velocity_observation
        linear_position_output = linear_position_error * self.linear_P_gain
        linear_velocity_output =  linear_velocity_error * self.linear_D_gain
        force_output = linear_position_output + linear_velocity_output

        return force_output
    
    def angularControlQuaternion(self):
        from push_gym.utils.transformations import quaternion_matrix,  quaternion_multiply, euler_from_quaternion, quaternion_inverse
        # multiply quaternion
        #calculate the difference quaternion 
        # example from cpp code
        #            // orientation error
        #    // "difference" quaternion
        #    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        #        orientation.coeffs() << -orientation.coeffs();
        #    }
        #    // "difference" quaternion
        #    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
        #    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
        #    // Transform to base frame
        #    error.tail(3) << -transform.linear() * error.tail(3);
        quat_ref = self.angular_position_command
        quat_obs = self.angular_position_observation
        
        #dot_res = np.dot(quat_ref, quat_obs)
        #if dot_res < 0:
        #    quat_obs = -quat_obs

        quat_inv = quaternion_inverse(quat_obs)
        quat_error = quaternion_multiply(quat_ref, quat_inv)
        #rot_mat = quat_to_rot_matrix(quat_obs_norm)
        #error  = np.matmul(-rot_mat, error_frame)
        #angle_error = quat_to_euler_angles(error, degrees = False)
        #angular_position_output = error_frame *  (self.angular_P_gain)
        #for i in range(3):
        #   quat_error[i] = quat_error[i] * np.sign(quat_error[3])
        error_frame = np.array([ quat_error[0], quat_error[1], quat_error[2]])
        #rot_mat = quaternion_matrix(quat_obs) # not necessary in this case
        error  = error_frame #np.matmul(-rot_mat[0:3, 0:3], error_frame)
        angular_position_output = self.angular_P_gain * error
        euler_obs  = np.asarray(euler_from_quaternion(quat_obs))
        euler_obs_deg = euler_obs * (180/np.pi)
        angular_velocity_error = self.angular_velocity_command - self.angular_velocity_observation
        angular_velocity_output = angular_velocity_error * self.angular_D_gain
        moment_output = angular_position_output + angular_velocity_output
        return moment_output

    def angularControl(self):
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        
        angular_position_obs_euler = quat_to_euler_angles(self.angular_position_observation, degrees = False)
        self.yaw_orientation = angular_position_obs_euler[2] * (180 / np.pi)
        angular_position_command_euler = quat_to_euler_angles(self.angular_position_command, degrees = False)
        angular_position_error =  angular_position_command_euler - angular_position_obs_euler
        angular_velocity_error = self.angular_velocity_command - self.angular_velocity_observation
        angular_position_output = angular_position_error * self.angular_P_gain
        angular_velocity_output = angular_velocity_error * self.angular_D_gain
        moment_output = angular_position_output + angular_velocity_output

        for i in range(3):
            if moment_output[i] > self.moment_limits[i]:
                moment_output[i] = self.moment_limits[i]
            elif moment_output[i] < -self.moment_limits[i]:
                moment_output[i] = -self.moment_limits[i]

        return moment_output


