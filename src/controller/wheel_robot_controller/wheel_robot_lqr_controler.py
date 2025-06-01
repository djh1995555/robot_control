from src.controller.base_controller.base_numerical_controler import BaseNumericalContoller
import numpy as np
import math
from scipy.spatial.transform import Rotation
from src.utils.math import *


WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 500.0 # rad/s

class WheelRobotLQRContoller(BaseNumericalContoller):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def reset(self, data):
        self.target_velocity = 1.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_filtered = 0.0

    def get_pitch(self, data) -> float:
        quat = data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)

        return angles[0]

    def get_pitch_dot(self, data) -> float:
        angular = data.joint('robot_body_joint').qvel[-3:]
        # print(angular)
        return angular[0]

    def get_wheel_velocity(self, data) -> float:
        vel_m_0 = data.joint('torso_l_wheel').qvel[0] * WHEEL_RADIUS
        vel_m_1 = data.joint('torso_r_wheel').qvel[0] * WHEEL_RADIUS

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0 * -1 + vel_m_1) / 2.0
    
    def calculate_lqr_velocity(self, data) -> float:
        self.pitch = -self.get_pitch(data)
        self.pitch_dot = self.get_pitch_dot(data)

        self.pitch_dot_filtered = (self.pitch_dot_filtered * .975) + (self.pitch_dot * .025)
        self.velocity_filtered = (self.velocity_filtered * .975) + (self.get_wheel_velocity(data) * .025)

        velocity_linear_error = self.target_velocity - self.velocity_filtered 
        state = np.array([0 - self.pitch, self.pitch_dot_filtered, 0, velocity_linear_error])

        self.K = np.array([[-2.1402165848237837, -0.03501370844016172, 5.9748026764525894e-18, 2.236067977499789]])
        self.action = -(self.K @ state)[0]
        return -self.action / WHEEL_RADIUS
    

    def generate_action(self, data):
        vel = self.calculate_lqr_velocity(data)
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        self.record_logger()
        return -vel + self.yaw, vel + self.yaw
    
    def record_logger(self):
        self.data_logger.add_data('pitch', self.pitch)
        self.data_logger.add_data('pitch_dot', self.pitch_dot)
        self.data_logger.add_data('pitch_dot_filtered', self.pitch_dot_filtered)
        self.data_logger.add_data('velocity_filtered', self.velocity_filtered)
        self.data_logger.add_data('target_velocity', self.target_velocity)
        self.data_logger.add_data('action', self.action)
        self.data_logger.add_data('K0', self.K[0][0])
        self.data_logger.add_data('K1', self.K[0][1])
        self.data_logger.add_data('K2', self.K[0][2])
        self.data_logger.add_data('K3', self.K[0][3])
