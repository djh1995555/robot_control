from src.controller.base_controller.base_numerical_controler import BaseNumericalContoller
import numpy as np
import math
from scipy.spatial.transform import Rotation
from src.utils.math import *

LQR_K = [-2.1402165848237837, -0.03501370844016172, 5.9748026764525894e-18, 2.236067977499789]
WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 500.0 # rad/s

class WheelRobotLQRContoller(BaseNumericalContoller):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def reset(self, data):
        self.velocity_linear_set_point = 0.0
        self.yaw = 0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular = 0.0
        self.velocity_angular_filtered = 0.0

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
        vel_m_0 = data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0 * -1 + vel_m_1) / 2.0
    
    def calculate_lqr_velocity(self, data) -> float:
        pitch = -self.get_pitch(data)
        pitch_dot = self.get_pitch_dot(data)

        # apply a filter to pitch dot, and velocity
        # without these filters the controller seems to lack necessary dampening
        # would like to know why!
        self.pitch_dot_filtered = (self.pitch_dot_filtered * .975) + (pitch_dot * .025)
        self.velocity_angular_filtered = (self.velocity_angular_filtered * .975) + (self.get_wheel_velocity(data) * .025)

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS

        lqr_v = LQR_K[0] * (0 - pitch) + LQR_K[1] * self.pitch_dot_filtered + LQR_K[2] * 0 + LQR_K[3] * velocity_linear_error 
        return -lqr_v / WHEEL_RADIUS
    

    def generate_action(self, data):
        vel = self.calculate_lqr_velocity(data)
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        return -vel + self.yaw, vel + self.yaw
    
