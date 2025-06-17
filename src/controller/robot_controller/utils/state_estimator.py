import numpy as np
from scipy.linalg import inv, pinv
from scipy.special import erf as scipy_erf
from typing import Tuple
from src.controller.robot_controller.utils.eul_w_filter import Eul_W_filter
from src.task.robot.data_bus import *
from src.utils.math import *

class StateEst:
    def __init__(self, dtIn: float):
        self.dt = dtIn
        self.eul_w_filter = Eul_W_filter(dtIn)
        
        # Initialize matrices
        self.A = np.zeros((15, 15))
        self.B = np.zeros((15, 3))
        self.C = np.zeros((14, 15))
        
        # Set up A matrix
        self.A[0:3, 0:3] = np.eye(3)
        self.A[0:3, 3:6] = np.eye(3) * dtIn
        self.A[3:6, 3:6] = np.eye(3)
        self.A[6:12, 6:12] = np.eye(6)
        self.A[12:15, 12:15] = np.eye(3)
        self.A[3:6, 12:15] = np.eye(3) * dtIn
        
        # Set up B matrix
        self.B[3:6, 0:3] = np.eye(3) * dtIn
        
        # Set up C matrix
        e = np.zeros((2, 6))
        e[0, 2] = 1
        e[1, 5] = 1
        self.C[0:3, 0:3] = np.eye(3)
        self.C[3:6, 0:3] = np.eye(3)
        self.C[0:3, 6:9] = -np.eye(3)
        self.C[3:6, 9:12] = -np.eye(3)
        self.C[6:9, 3:6] = -np.eye(3)
        self.C[9:12, 3:6] = -np.eye(3)
        self.C[12:14, 6:12] = e
        
        # Initialize state and covariance
        self.P = np.eye(15)
        self.X = np.zeros(15)
        self.P0 = np.eye(15)
        self.X0 = np.zeros(15)
        self.Q = np.zeros((15, 15))
        self.Qu = np.zeros((15, 15))
        self.R = np.zeros((14, 14))
        self.peB_old = np.zeros((3, 2))
        self.peW_old = np.zeros((3, 2))
        self.peW = np.zeros((3, 2))
        
        # Initialize velocity filters
        self.vCoM_LP = [0, 0, 0]
        
        # Initialize Eul_W_filter parameters
        self.eul_w_filter.Q = np.eye(6)
        self.eul_w_filter.Q[0, 0] = 1e-8
        self.eul_w_filter.Q[1, 1] = 1e-8
        self.eul_w_filter.Q[2, 2] = 1e-8
        self.eul_w_filter.Q[3, 3] = 5e-7
        self.eul_w_filter.Q[4, 4] = 5e-7
        self.eul_w_filter.Q[5, 5] = 2e-5
        
        self.eul_w_filter.R = np.eye(6)
        self.eul_w_filter.R[0, 0] = 1e-6
        self.eul_w_filter.R[1, 1] = 1e-6
        self.eul_w_filter.R[2, 2] = 1e-6
        self.eul_w_filter.R[3, 3] = 1e-4
        self.eul_w_filter.R[4, 4] = 1e-4
        self.eul_w_filter.R[5, 5] = 1e-5
        
        # Initialize other variables
        self.flag_init = True
        self.startFlag = False
        self.offYaw = 0
        self.leg_contact = [True, True]
        
        # Trust region parameters
        self.TR_kh = 100
        self.TR_k = 100
        
        # Kalman filter parameters
        self.KF_Q_wPCoM = np.array([5e-6, 5e-6, 5e-6])
        self.KF_Q_wVCoM = np.array([1e-8, 2e-8, 1e-6])
        self.KF_Q_wfeW = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.KF_Q_waL = np.array([1e-8, 1e-8, 1e-8])
        self.KF_Q_waU = np.array([2e-4, 2e-4, 8e-4])
        
        self.KF_R_wfeL = np.array([1e-7, 1e-7, 1e-10, 6e-7, 8e-8, 1e-11])
        self.KF_R_wdfeL = np.array([1e-5, 2e-5, 1e-6, 1e-5, 2e-5, 1e-6])
        self.KF_R_wh = np.ones(2) * 1e-3
        
        # Contact force thresholds
        self.FcontactUpp = [280, 280]
        self.FcontactLow = [10, 10]
        self.FzThrehold = [50, 50]  # Example values, adjust as needed

        self.base_vel = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.delta_acc = np.zeros(3)
        self.fe_l_pos_W = self.peW[:, 0]
        self.fe_r_pos_W = self.peW[:, 1]

        self.Y = np.zeros(14)
    
    def init(self, Data):
        self.offYaw = Data.rpy[2]
        print(f"yawoffset: {Data.rpy[0]:.4f}, {Data.rpy[1]:.4f}, {Data.rpy[2]:.4f}")
        
        R = eul2Rot(Data.base_rpy[0], Data.base_rpy[1], 0)
        pWL = R @ Data.fe_l_pos_L
        pWR = R @ Data.fe_r_pos_L
        zOff = -(pWR[2] + pWL[2]) / 2.0 + 0.07
        
        self.X0 = np.zeros(15)
        self.X0[0:3] = [0, 0, zOff]
        self.X0[6:9] = [pWL[0], pWL[1], 0.07]
        self.X0[9:12] = [pWR[0], pWR[1], 0.07]
        
        self.P0 = np.zeros((15, 15))
        for i in range(15):
            self.P0[i, i] = 0.1
        
        self.flag_init = False
        self.X = self.X0
        self.P = self.P0
    
    def set(self, Data):
        if self.flag_init:
            self.offYaw = Data.rpy[2]
        
        self.acc = np.array(Data.baseAcc)
        self.eul = np.array([Data.rpy[0], Data.rpy[1] - np.deg2rad(0.0), Data.rpy[2]])
        self.eul_woOff = np.array([self.eul[0], self.eul[1], self.eul[2] - self.offYaw])
        self.Rrpy_woOff = eul2Rot(self.eul_woOff[0], self.eul_woOff[1], self.eul_woOff[2])
        
        self.omegaL = np.array(Data.baseAngVel)
        self.omegaW = Data.base_omega_W
        
        # Filter euler angles and angular velocities
        eul_woOff_ary = self.eul_woOff
        omegaL_ary = self.omegaL
        self.eul_w_filter.run(eul_woOff_ary, omegaL_ary)
        self.Eul_filtered, self.wL_filtered = self.eul_w_filter.getData()
        
        self.eul_woOff = self.Eul_filtered
        self.eul = np.array([self.eul_woOff[0], self.eul_woOff[1], self.eul_woOff[2] + self.offYaw])
        self.Rrpy_woOff = eul2Rot(self.eul_woOff[0], self.eul_woOff[1], self.eul_woOff[2])
        self.Rrpy = eul2Rot(self.eul[0], self.eul[1], self.eul[2])
        self.omegaL = self.wL_filtered
        self.omegaW = self.Rrpy_woOff @ self.omegaL
        
        accTmp = np.array(self.acc)
        self.freeAcc = eul2Rot(0.0, 0.0, self.offYaw).T @ accTmp
        
        self.phi = Data.phi
        self.legState = Data.legState
        self.fe_l_pos_L = np.array(Data.fe_l_pos_L)
        self.fe_r_pos_L = np.array(Data.fe_r_pos_L)
        self.fe_l_vel_L = np.array(Data.fe_l_vel_L)
        self.fe_r_vel_L = np.array(Data.fe_r_vel_L)
        self.fe_l_pos_W = np.array(Data.fe_l_pos_W)
        self.fe_r_pos_W = np.array(Data.fe_r_pos_W)
    
    def getTrustRegion_wt_h(self):
        C = np.zeros(6)
        Cv = np.zeros(6)
        Ch = np.zeros(2)
        aa = [0, 0]
        bb = [5.0, 5.0]
        
        for i in range(2):
            if self.legState == MotionState.Stand:
                aa[i] = 1.0
            else:
                aa[i] = self.phi * bb[i]
                if aa[i] > 1.0:
                    aa[i] = 1.0
                aa[i] = (1.0 - aa[i]) * 10000.0
                aa[i] = aa[i] + 1.0
        
        for i in range(2):
            C[i*3] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            C[i*3+1] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            C[i*3+2] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            
            Cv[i*3] = self.leg_contact[i] * aa[0] + (1 - self.leg_contact[i]) * self.TR_k
            Cv[i*3+1] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            Cv[i*3+2] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            
            Ch[i] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_kh
        
        self.Xi = np.diag(C)
        self.Xih = np.diag(Ch)
        self.Xiv = np.diag(Cv)
    
    def update(self):
        # Update leg contact states
        if self.legState == LegState.LSt:
            self.leg_contact[0] = True
            self.leg_contact[1] = False
        elif self.legState == LegState.RSt:
            self.leg_contact[1] = True
            self.leg_contact[0] = False
        else:
            self.leg_contact[0] = True
            self.leg_contact[1] = True
        
        self.peB = np.column_stack((self.fe_l_pos_L, self.fe_r_pos_L))
        self.getTrustRegion_wt_h()
        
        # Prediction step
        diagQ = np.zeros(15)
        diagR = np.zeros(14)
        
        diagQ[0:3] = self.KF_Q_wPCoM
        diagQ[3:6] = self.KF_Q_wVCoM
        diagQ[6:12] = self.Xi @ self.KF_Q_wfeW
        diagQ[12:15] = self.KF_Q_waL
        
        diagR[0:6] = self.KF_R_wfeL
        diagR[6:12] = self.Xiv @ self.KF_R_wdfeL
        diagR[12:14] = self.Xih @ self.KF_R_wh
        
        self.Qu = self.B @ np.diag(self.KF_Q_waU) @ self.B.T
        self.Q = np.diag(diagQ) + self.Qu
        self.R = np.diag(diagR)
        
        self.X = self.A @ self.X + self.B @ self.freeAcc
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        # Measurement evaluation
        omegaLVec = self.wL_filtered
        self.pbW = self.Rrpy_woOff @ self.peB
        velTmp = np.column_stack((self.fe_l_vel_L, self.fe_r_vel_L))
        
        self.vbW = np.zeros((3, 2))
        for i in range(2):
            self.vbW[:, i] = (self.leg_contact[i] * self.Rrpy_woOff @ 
                            (velTmp[:, i] + np.cross(omegaLVec, self.peB[:, i])) + 
                            (1 - self.leg_contact[i]) * (-self.base_vel))
            
            self.Y[12 + i] = (self.leg_contact[i] * 0.07 + 
                             (1 - self.leg_contact[i]) * (self.base_pos[2] + self.pbW[2, i]))
        
        if np.all(self.peB_old == 0) or np.all(self.peW_old == 0):
            self.vbW = np.zeros((3, 2))
        
        self.Y[0:3] = -self.pbW[:, 0]
        self.Y[3:6] = -self.pbW[:, 1]
        self.Y[6:9] = self.vbW[:, 0]
        self.Y[9:12] = self.vbW[:, 1]
        
        self.peB_old = self.peB
        self.peW_old = self.pbW
        
        # Correction step
        S_inv = self.C @ self.P @ self.C.T + self.R
        S_inv = inv(S_inv)
        
        self.K = self.P @ self.C.T @ S_inv
        self.P = (np.eye(15) - self.K @ self.C) @ self.P
        self.X = self.X + self.K @ (self.Y - self.C @ self.X)
        
        if not self.startFlag:
            self.X = self.X0
            self.P = self.P0
        
        if not self.flag_init:
            self.startFlag = True
        
        # Update state variables
        self.peW = np.column_stack((self.X[6:9], self.X[9:12]))
        self.base_pos = self.X[0:3]
        self.base_vel = self.X[3:6]
        self.delta_acc = self.X[12:15]
        self.fe_l_pos_W = self.peW[:, 0]
        self.fe_r_pos_W = self.peW[:, 1]
    
    def get(self, Data):
        Data.base_pos_est = self.base_pos
        Data.base_vel_est = self.base_vel
        Data.fe_l_pos_W_est = self.fe_l_pos_W
        Data.fe_r_pos_W_est = self.fe_r_pos_W
        Data.delta_acc = self.delta_acc
        Data.eul_est = self.eul_woOff
        Data.omegaW_est = self.omegaW
        
        Data.base_pos = Data.base_pos_est
        Data.base_vel = Data.base_vel_est
        Data.baseAcc[0] += self.delta_acc[0]
        Data.baseAcc[1] += self.delta_acc[1]
        Data.baseAcc[2] += self.delta_acc[2]
        Data.base_rpy = Data.eul_est
        Data.base_omega_W = Data.omegaW_est
        Data.base_rot = self.Rrpy_woOff
        
        Data.q[0:3] = Data.base_pos
        Data.dq[0:3] = Data.base_vel
        
        quatNow = eul2quat(Data.base_rpy[0], Data.base_rpy[1], Data.base_rpy[2])
        Data.q[3:7] = quatNow
        Data.dq[3:6] = Data.base_omega_W
        
        # Test data
        Data.AX = self.A @ self.X
        Data.BU = self.B @ self.freeAcc
        Data.freeAcc = self.freeAcc
        Data.CX = self.C @ self.X
        Data.Y = self.Y
        Data.pbW = self.pbW.flatten()
        Data.leg_contact = self.leg_contact.copy()
    
    def setF(self, Data):
        self.model_nv = Data.model_nv
        self.torJoint = np.zeros(self.model_nv - 6)
        for i in range(self.model_nv - 6):
            self.torJoint[i] = Data.motors_tor_cur[i]
        
        self.dyn_M = Data.dyn_M
        self.dyn_Non = Data.dyn_Non
        self.J_l = Data.J_l
        self.dJ_l = Data.dJ_l
        self.J_r = Data.J_r
        self.dJ_r = Data.dJ_r
        self.dq = Data.dq
    
    def updateF(self):
        tauAll = np.zeros(self.model_nv)
        tauAll[6:] = self.torJoint
        
        J_l_M_inv = self.J_l @ inv(self.dyn_M)
        J_r_M_inv = self.J_r @ inv(self.dyn_M)
        
        self.FLest = -pinv(J_l_M_inv @ self.J_l.T) @ (J_l_M_inv @ (tauAll - self.dyn_Non) + self.dJ_l @ self.dq)
        self.FRest = -pinv(J_r_M_inv @ self.J_r.T) @ (J_r_M_inv @ (tauAll - self.dyn_Non) + self.dJ_r @ self.dq)
    
    def getF(self, Data):
        Data.FL_est = self.FLest
        Data.FR_est = self.FRest
    
    @staticmethod
    def CphiFun(phi: float, s: float, W: float) -> float:
        return s * (scipy_erf(12 * phi / W - 6) + scipy_erf(12 * (1 - phi) / W - 6) - 1)
    
    @staticmethod
    def CzFun(pz: float, kp: float, kn: float) -> float:
        if pz >= 0:
            return np.exp(-kp * pz * pz)
        else:
            return np.exp(-kn * pz * pz)
    
    @staticmethod
    def erf(xIn: float) -> float:
        return 1 / (1 + np.exp(-xIn))

    def get_init(self) -> bool:
        return self.flag_init
