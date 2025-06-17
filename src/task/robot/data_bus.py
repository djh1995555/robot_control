import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from src.utils.math import *
class MotionState(Enum):
    Stand = auto()
    Walk = auto()
    Walk2Stand = auto()

class LegState(Enum):
    LSt = auto()  # Left leg stance
    RSt = auto()  # Right leg stance
    DSt = auto()  # Double stance

@dataclass
class DataBus:
    def __init__(self, model_nv: int):
        self.model_nv = model_nv  # number of dq
        self.width_hips = 0.229
        
        # Constants for frame mismatch
        self.fe_L_rot_L_off = np.eye(3)  # left foot-end R w.r.t to the body frame in offset posture
        self.fe_R_rot_L_off = np.eye(3)
        
        # Motors, sensors and states feedback
        self.rpy = np.zeros(3)
        self.fL = np.zeros(3)
        self.fR = np.zeros(3)
        self.basePos = np.zeros(3)
        self.baseLinVel = np.zeros(3)  # velocity of the basePos
        self.baseAcc = np.zeros(3)     # baseAcc of the base link
        self.baseAngVel = np.zeros(3)  # angular velocity of the base link
        self.motors_pos_cur = np.zeros(model_nv - 6)
        self.motors_vel_cur = np.zeros(model_nv - 6)
        self.motors_tor_cur = np.zeros(model_nv - 6)
        self.FL_est = np.zeros(6)
        self.FR_est = np.zeros(6)
        self.isdqIni = False
        
        # PVT controls
        self.motors_pos_des = np.zeros(model_nv - 6)
        self.motors_vel_des = np.zeros(model_nv - 6)
        self.motors_tor_des = np.zeros(model_nv - 6)
        self.motors_tor_out = np.zeros(model_nv - 6)
        
        # States and key variables
        self.q = np.zeros(model_nv + 1)
        self.dq = np.zeros(model_nv)
        self.ddq = np.zeros(model_nv)
        self.qOld = np.zeros(model_nv + 1)
        self.J_base = np.zeros((6, model_nv))
        self.J_l = np.zeros((6, model_nv))
        self.J_r = np.zeros((6, model_nv))
        self.J_hd_l = np.zeros((6, model_nv))
        self.J_hd_r = np.zeros((6, model_nv))
        self.J_hip_link = np.zeros((6, model_nv))
        self.dJ_base = np.zeros((6, model_nv))
        self.dJ_l = np.zeros((6, model_nv))
        self.dJ_r = np.zeros((6, model_nv))
        self.dJ_hd_l = np.zeros((6, model_nv))
        self.dJ_hd_r = np.zeros((6, model_nv))
        self.Jcom_W = np.zeros((3, model_nv))  # jacobian of CoM, in world frame
        self.pCoM_W = np.zeros(3)
        self.fe_r_pos_W = np.zeros(3)
        self.fe_l_pos_W = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.base_vel = np.zeros(3)
        self.fe_r_rot_W = np.eye(3)  # in world frame
        self.fe_l_rot_W = np.eye(3)
        self.base_rot = np.eye(3)
        self.fe_r_pos_L = np.zeros(3)  # in Body frame
        self.fe_l_pos_L = np.zeros(3)
        self.fe_r_vel_L = np.zeros(3)  # linear velocity in Body frame
        self.fe_l_vel_L = np.zeros(3)
        self.hip_link_pos = np.zeros(3)
        self.hip_r_pos_L = np.zeros(3)
        self.hip_l_pos_L = np.zeros(3)
        self.hip_r_pos_W = np.zeros(3)
        self.hip_l_pos_W = np.zeros(3)
        self.fe_r_rot_L = np.eye(3)
        self.fe_l_rot_L = np.eye(3)
        self.hip_link_rot = np.eye(3)
        self.fe_r_pos_L_cmd = np.zeros(3)
        self.fe_l_pos_L_cmd = np.zeros(3)
        self.fe_r_rot_L_cmd = np.eye(3)
        self.fe_l_rot_L_cmd = np.eye(3)
        
        self.hd_r_pos_W = np.zeros(3)  # in world frame
        self.hd_l_pos_W = np.zeros(3)
        self.hd_r_rot_W = np.eye(3)
        self.hd_l_rot_W = np.eye(3)
        self.hd_r_pos_L = np.zeros(3)  # in body frame
        self.hd_l_pos_L = np.zeros(3)
        self.hd_r_rot_L = np.eye(3)
        self.hd_l_rot_L = np.eye(3)
        self.qCmd = np.zeros(model_nv + 1)
        self.dqCmd = np.zeros(model_nv)
        self.tauJointCmd = np.zeros(model_nv - 6)
        self.dyn_M = np.zeros((model_nv, model_nv))
        self.dyn_M_inv = np.zeros((model_nv, model_nv))
        self.dyn_C = np.zeros((model_nv, model_nv))
        self.dyn_Ag = np.zeros((6, model_nv))
        self.dyn_dAg = np.zeros((6, model_nv))
        self.dyn_G = np.zeros(model_nv)
        self.dyn_Non = np.zeros(model_nv)
        self.base_omega_L = np.zeros(3)
        self.base_omega_W = np.zeros(3)
        self.base_rpy = np.zeros(3)
        
        self.slop = np.zeros(3)
        self.inertia = np.eye(3)
        
        # State EST
        self.base_pos_est = np.zeros(3)
        self.base_vel_est = np.zeros(3)
        self.eul_est = np.zeros(3)
        self.omegaW_est = np.zeros(3)
        self.fe_l_pos_W_est = np.zeros(3)
        self.fe_r_pos_W_est = np.zeros(3)
        self.delta_acc = np.zeros(3)
        self.freeAcc = np.zeros(3)
        
        self.AX = np.zeros(15)
        self.BU = np.zeros(15)
        self.CX = np.zeros(14)
        self.Y = np.zeros(14)
        self.pbW = np.zeros(6)
        
        # cmd value from the joystick interpreter
        self.js_eul_des = np.zeros(3)
        self.js_pos_des = np.zeros(3)
        self.js_omega_des = np.zeros(3)
        self.js_vel_des = np.zeros(3)
        
        # cmd values for MPC
        self.Xd = np.zeros(12 * 10)
        self.X_cur = np.zeros(12)
        self.X_cal = np.zeros(12)
        self.dX_cal = np.zeros(12)
        self.fe_react_tau_cmd = np.zeros(13 * 3)
        
        self.qp_nWSR_MPC = 0
        self.qp_cpuTime_MPC = 0.0
        self.qpStatus_MPC = 0
        
        # cmd values for WBC
        self.base_rpy_des = np.zeros(3)
        self.base_pos_des = np.zeros(3)
        self.base_vel_des = np.zeros(3)
        self.base_omega_des = np.zeros(3)
        self.des_ddq = np.zeros(model_nv)
        self.des_dq = np.zeros(model_nv)
        self.des_delta_q = np.zeros(model_nv)
        self.des_q = np.zeros(model_nv)
        self.swing_fe_pos_des_W = np.zeros(3)
        self.swing_fe_rpy_des_W = np.zeros(3)
        self.stance_fe_pos_cur_W = np.zeros(3)
        self.stance_fe_rot_cur_W = np.eye(3)
        self.wbc_delta_q_final = np.zeros(model_nv)
        self.wbc_dq_final = np.zeros(model_nv)
        self.wbc_ddq_final = np.zeros(model_nv)
        self.wbc_tauJointRes = np.zeros(model_nv - 6)
        self.wbc_FrRes = np.zeros(12)
        self.Fr_ff = np.zeros(12)
        self.qp_nWSR = 0
        self.qp_cpuTime = 0.0
        self.qp_status = 0
        
        # values for foot-placement
        self.swingStartPos_W = np.zeros(3)
        self.swingDesPosCur_W = np.zeros(3)
        self.swingDesPosCur_L = np.zeros(3)
        self.swingDesPosFinal_W = np.zeros(3)
        self.stanceDesPos_W = np.zeros(3)
        self.posHip_W = np.zeros(3)
        self.posST_W = np.zeros(3)
        self.desV_W = np.zeros(3)  # desired linear velocity
        self.desWz_W = 0.0         # desired angular velocity
        self.theta0 = 0.0          # offset yaw angle of the swing leg, w.r.t body frame
        self.width_hips = 0.0      # distance between the left and right hip
        self.tSwing = 0.0
        self.phi = 0.0
        
        self.leg_contact = [False, False]
        self.thetaZ_des = 0.0
        self.legState = LegState.DSt
        self.legStateNext = LegState.DSt
        self.motionState = MotionState.Stand
        
        # for jump
        self.base_pos_stand = np.zeros(3)
        self.pfeW_stand = np.zeros(6)
        self.pfeW0 = np.zeros(6)
    
    def updateQ(self):
        """Update q according to sensor values"""
        self.base_omega_W = np.array([
            self.baseAngVel[0],
            self.baseAngVel[1],
            self.baseAngVel[2]
        ])
        Rcur = eul2Rot(self.rpy[0], self.rpy[1], self.rpy[2])
        self.base_omega_W = Rcur @ self.base_omega_W
        
        # q = [global_base_position, global_base_quaternion, joint_positions]
        # dq = [global_base_velocity_linear, global_base_velocity_angular, joint_velocities]
        
        quatNow = eul2quat(self.rpy[0], self.rpy[1], self.rpy[2])
        self.q[0] = self.basePos[0]
        self.q[1] = self.basePos[1]
        self.q[2] = self.basePos[2]
        self.q[3] = quatNow[0]
        self.q[4] = quatNow[1]
        self.q[5] = quatNow[2]
        self.q[6] = quatNow[3]
        for i in range(self.model_nv - 6):
            self.q[i + 7] = self.motors_pos_cur[i]
        
        vCoM_W = np.array([
            self.baseLinVel[0],
            self.baseLinVel[1],
            self.baseLinVel[2]
        ])
        self.dq[:3] = vCoM_W
        self.dq[3:6] = self.base_omega_W[:3]
        for i in range(self.model_nv - 6):
            self.dq[i + 6] = self.motors_vel_cur[i]
        
        self.base_pos = self.q[:3]
        self.base_rpy = self.rpy
        self.base_rot = Rcur
        self.qOld = self.q.copy()
