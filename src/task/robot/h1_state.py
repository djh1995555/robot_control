from enum import Enum
import numpy as np
from src.utils.math import *
from scipy.spatial.transform import Rotation

class LegState(Enum):
    LeftStand = 1
    RightStand = 2
    DualStand = 3

class MotionState(Enum):
    Walk = 1
    Walk2Stand = 2
    Stand = 3

class H1State():
    def __init__(self, model_nv):
        self.model_nv = model_nv
        self.R_l_foot_end_to_body_offset = np.identity(3)
        self.R_r_foot_end_to_body_offset = np.identity(3)
        self.motor_pos_cur = [0] * model_nv
        self.motor_pos_des = [0] * model_nv
        self.motor_vel_cur = [0] * model_nv
        self.motor_vel_des = [0] * model_nv
        self.motor_torque_cur = [0] * model_nv
        self.motor_torque_cur_link = [0] * model_nv
        self.motor_torque_des = [0] * model_nv

        self.q = np.zeros(model_nv + 1)  # q =  [global_base_position, global_base_quaternion, joint_positions]
        self.dq = np.zeros(model_nv)     # dq = [global_base_velocity_linear, global_base_velocity_angular, joint_velocities]
        self.ddq = np.zeros(model_nv)
        self.q_des = np.zeros(model_nv)
        self.dq_des = np.zeros(model_nv)
        self.ddq_des = np.zeros(model_nv)

        self.footend_force_l_est = np.zeros(6)
        self.footend_force_r_est = np.zeros(6)
        self.x_des = np.zeros([12, 10])
        self.x_cur = np.zeros(12)
        self.x_cal = np.zeros(12)
        self.dx_cal = np.zeros(12)
        self.footend_force_mpc_cmd = np.zeros(13)    # MPC求解结果
        self.footend_force_cmd = np.zeros(12)        # MPC求解结果中的足端力


        self.base_quat = np.zeros(4)
        self.base_rpy_des = np.zeros(3)
        self.base_pos_des = np.zeros(3)
        self.base_vel_des = np.zeros(3)
        self.base_omega_des = np.zeros(3)

        self.base_rpy = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.base_vel = np.zeros(3)
        self.base_acc = np.zeros(3)
        self.base_omega = np.zeros(3)

        self.base_omega_W = np.zeros(3)

        self.yaw_pre = 0.0

        # 手柄的期望指令
        self.js_eul_des = np.zeros(3)
        self.js_pos_des = np.zeros(3)
        self.js_omega_des = np.zeros(3)
        self.js_vel_des = np.zeros(3)

        self.motion_state = MotionState.Stand

    def update(self):
        self.R_B2W = Rotation.from_euler('zyx', [self.base_rpy[0],self.base_rpy[1],self.base_rpy[2]]).as_matrix()
        self.base_omega_W = self.R_B2W @ self.base_omega.T

        self.q[0] = self.base_pos[0]
        self.q[1] = self.base_pos[1]
        self.q[2] = self.base_pos[2]
        self.q[3] = self.base_quat[0]
        self.q[4] = self.base_quat[1]
        self.q[5] = self.base_quat[2]
        self.q[6] = self.base_quat[3]
        for i in range(self.model_nv - 6):
            self.q[i + 7] = self.motor_pos_cur[i]

        self.dq[0:3] = self.base_vel
        self.dq[3:6] = self.base_omega_W
        for i in range(self.model_nv-6):
            self.dq[i+6] = self.motor_vel_cur[i]
        
