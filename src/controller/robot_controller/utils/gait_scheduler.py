import math
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
        coriolis_torque = self.state.coriolis_torque
        dq = self.state.dq
        self.left_foot_force_est = -np.linalg.pinv(J_ankle_l * dyn_M_inv * J_ankle_l.T) * (J_ankle_l * dyn_M_inv * (self.torques - coriolis_torque) + dJ_ankle_l * dq)
        self.right_foot_force_est = -np.linalg.pinv(J_ankle_r * dyn_M_inv * J_ankle_r.T) * (J_ankle_r * dyn_M_inv * (self.torques - coriolis_torque) + dJ_ankle_r * J_ankle_l * dq)

        self.dPhi = 0.0
        if self.motionState == "Walk2Stand":
            self.enableNextStep = False
            self.start_walk = False
            if self.touchDown:
                self.motionState = "Stand"

        if self.motionState == "Stand":
            self.dPhi = 0
            self.phi = 0  # need to refined
            self.isIni = False
            self.enableNextStep = False
            self.stepNumCur = 0
        elif self.motionState == "Walk":
            self.enableNextStep = True
            self.dPhi = 1.0 / self.tSwing * self.dt
        elif self.motionState == "Walk2Stand":
            self.dPhi = 1.0 / self.tSwing * self.dt

        self.phi += self.dPhi
        if self.enableNextStep:
            self.touchDown = False

        if not self.isIni and self.start_walk:
            self.isIni = True
            self.legState = self.firstleg
            if self.legState == "LSt":
                # here define which leg support first
                self.swingStartPos_W = self.fe_r_pos_W.copy()
                self.stanceStartPos_W = self.fe_l_pos_W.copy()
            else:
                self.swingStartPos_W = self.fe_l_pos_W.copy()
                self.stanceStartPos_W = self.fe_r_pos_W.copy()

        if (self.legState == "LSt" and self.right_foot_force_est[2] >= 280 and self.phi >= 0.6):
            if self.enableNextStep:
                self.legState = "RSt"
                self.swingStartPos_W = self.fe_l_pos_W.copy()
                self.stanceStartPos_W = self.fe_r_pos_W.copy()
                self.phi = 0
                self.stepNumCur += 1
        elif (self.legState == "RSt" and self.left_foot_force_est[2] >= 280 and self.phi >= 0.6):
            if self.enableNextStep:
                self.legState = "LSt"
                self.swingStartPos_W = self.fe_r_pos_W.copy()
                self.stanceStartPos_W = self.fe_l_pos_W.copy()
                self.phi = 0
                self.stepNumCur += 1

        if not self.enableNextStep:
            if self.legState == "LSt" and self.right_foot_force_est[2] >= 200:
                self.touchDown = True
                self.stepNumCur += 1
                self.legState = "DSt"
            if self.legState == "RSt" and self.left_foot_force_est[2] >= 200:
                self.touchDown = True
                self.stepNumCur += 1
                self.legState = "DSt"

        if self.phi >= 1:
            self.phi = 1

        if self.legState == "LSt":
            self.posHip_W = self.hip_r_pos_W.copy()
            self.posST_W = self.fe_l_pos_W.copy()
            self.theta0 = -math.pi * 0.5
            if self.motionState == "Walk":
                self.legStateNext = "RSt"
            elif self.motionState == "Walk2Stand":
                self.legStateNext = "DSt"
        elif self.legState == "RSt":
            self.posHip_W = self.hip_l_pos_W.copy()
            self.posST_W = self.fe_r_pos_W.copy()
            self.theta0 = math.pi * 0.5
            if self.motionState == "Walk":
                self.legStateNext = "LSt"
            elif self.motionState == "Walk2Stand":
                self.legStateNext = "DSt"
        else:
            self.posHip_W = self.hip_l_pos_W.copy()
            self.posST_W = self.fe_r_pos_W.copy()
            self.theta0 = math.pi * 0.5
            self.legStateNext = "DSt"


