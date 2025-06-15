from src.task.robot.h1_state import *
from scipy import linalg
import numpy as np
class GaitScheduler():
    def __init__(self,cfg):
        self.t_swing = 0.4  # 步态摆动周期
        self.dt = 0.001     # 采样时间
        self.phi = 0.0      # 步态相位
        self.inited = False
        self.first_leg = LegState.LeftStand
        self.leg_state = LegState.DualStand
        self.leg_state_next = self.first_leg
        self.motion_state = MotionState.Stand
        self.enable_next_step = False
        self.touch_down = False
        self.step_num = 0

    def update_state(self, state):
        if(self.motion_state != MotionState.Stand and self.step_num == 0):
            self.leg_state == self.first_leg
        self.state = state

    def step(self):
        self.torques = np.zeros((8,1), dtype=float) # 所有关节的torque，包含6个虚拟关节
        self.torques[6:-1] = self.state.actual_torque
        J_ankle_l = self.state.J_ankle_l
        J_ankle_r = self.state.J_ankle_r
        dJ_ankle_l = self.state.dJ_ankle_l
        dJ_ankle_r = self.state.dJ_ankle_l
        dyn_M_inv = np.linalg.inv(self.state.dyn_M)
        coriolis_torque = self.coriolis_torque
        dq = self.state.dq
        left_foot_force_est = -np.linalg.pinv(J_ankle_l * dyn_M_inv * J_ankle_l.T) * (J_ankle_l * dyn_M_inv * (self.torques - coriolis_torque) + dJ_ankle_l * dq)
        right_foot_force_est = -np.linalg.pinv(J_ankle_r * dyn_M_inv * J_ankle_r.T) * (J_ankle_r * dyn_M_inv * (self.torques - coriolis_torque) + dJ_ankle_r * J_ankle_l * dq)




