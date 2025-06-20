import numpy as np
from dataclasses import dataclass
from typing import List
from src.task.robot.data_bus import *

class Bezier1D:
    def __init__(self):
        self.P = []
    
    def getOut(self, t: float) -> float:
        """Evaluate 1D Bezier curve at parameter t"""
        n = len(self.P) - 1
        if n < 0:
            return 0.0
        
        result = 0.0
        for i, p in enumerate(self.P):
            # Bernstein polynomial
            coeff = np.math.comb(n, i) * (t**i) * ((1 - t)**(n - i))
            result += p * coeff
        return result

class FootPlacement:
    def __init__(self):
        self.kp_vx = 0.0
        self.kp_vy = 0.0
        self.kp_wz = 0.0
        self.legLength = 1.0
        self.stepHeight = 0.1
        self.phi = 0.0
        self.tSwing = 0.4
        self.posStart_W = np.zeros(3)
        self.posDes_W = np.zeros(3)
        self.hipPos_W = np.zeros(3)
        self.STPos_W = np.zeros(3)
        self.desV_W = np.zeros(3)
        self.curV_W = np.zeros(3)
        self.desWz_W = 0.0
        self.base_pos = np.zeros(3)
        self.legState = LegState.Lst
        
        # Private variables
        self.pDesCur = np.zeros(3)
        self.yawCur = 0.0
        self.theta0 = 0.0
        self.omegaZ_W = 0.0
        self.hip_width = 0.0
        self.stretchLeg = False
        self.zStretch = 0.0
        self.finish_Stretch = False
    
    def dataBusRead(self, robotState: DataBus):
        """Read data from DataBus"""
        self.posStart_W = robotState.swingStartPos_W.copy()
        self.desV_W = robotState.js_vel_des.copy()
        self.desWz_W = robotState.js_omega_des[2]
        self.curV_W = robotState.dq[:3].copy()
        self.phi = robotState.phi
        self.hipPos_W = robotState.posHip_W.copy()
        self.STPos_W = robotState.posST_W.copy()
        self.base_pos = robotState.base_pos.copy()
        self.tSwing = robotState.tSwing
        self.theta0 = robotState.theta0
        self.yawCur = robotState.rpy[2]
        self.omegaZ_W = robotState.base_omega_W[2]
        self.hip_width = robotState.width_hips
        self.legState = robotState.legState
    
    def dataBusWrite(self, robotState: DataBus):
        """Write data to DataBus"""
        robotState.swingDesPosCur_W = self.pDesCur.copy()
        robotState.swingDesPosFinal_W = self.posDes_W.copy()
        robotState.swing_fe_rpy_des_W = np.array([0, 0, robotState.base_rpy_des[2]])
        robotState.swing_fe_pos_des_W = self.pDesCur.copy()
    
    def getSwingPos(self):
        """Calculate the desired swing foot position"""
        KP = np.zeros((3, 3))
        KP[0, 0] = self.kp_vx
        KP[1, 1] = self.kp_vy
        
        # Rotation matrix for yaw
        Rz = np.array([
            [np.cos(self.yawCur), -np.sin(self.yawCur), 0],
            [np.sin(self.yawCur), np.cos(self.yawCur), 0],
            [0, 0, 1]
        ])
        KP = Rz @ KP @ Rz.T
        
        # Linear velocity component
        self.posDes_W = (self.hipPos_W + 
                         KP @ (self.desV_W - self.curV_W) * (-1) + 
                         0.5 * self.tSwing * self.curV_W + 
                         self.curV_W * (1 - self.phi) * self.tSwing)
        
        # Angular velocity component
        thetaF = (self.yawCur + self.theta0 + 
                 self.omegaZ_W * (1 - self.phi) * self.tSwing + 
                 0.5 * self.omegaZ_W * self.tSwing + 
                 self.kp_wz * (self.omegaZ_W - self.desWz_W))
        
        self.posDes_W[0] += 0.5 * self.hip_width * (np.cos(thetaF) - np.cos(self.yawCur + self.theta0))
        self.posDes_W[1] += 0.5 * self.hip_width * (np.sin(thetaF) - np.sin(self.yawCur + self.theta0))
        
        # Foot position offsets
        xOff_L = -0.07
        yOff_L = 0.04
        zOff_W = -0.035
        
        self.posDes_W[2] = self.base_pos[2] - self.legLength + zOff_W
        
        # Apply leg-specific offsets
        if self.legState == LegState.LSt:
            xOff_W = np.cos(self.yawCur) * xOff_L - np.sin(self.yawCur) * yOff_L
            yOff_W = np.sin(self.yawCur) * xOff_L + np.cos(self.yawCur) * yOff_L
        elif self.legState == LegState.RSt:
            xOff_W = np.cos(self.yawCur) * xOff_L - np.sin(self.yawCur) * (-yOff_L)
            yOff_W = np.sin(self.yawCur) * xOff_L + np.cos(self.yawCur) * (-yOff_L)
        
        self.posDes_W[0] += xOff_W
        self.posDes_W[1] += yOff_W
        
        # Cycloid trajectory for X and Y
        if self.phi < 1.0:
            phase = 2 * np.pi * self.phi
            self.pDesCur[0] = (self.posStart_W[0] + 
                              (self.posDes_W[0] - self.posStart_W[0]) / (2 * np.pi) * 
                              (phase - np.sin(phase)))
            self.pDesCur[1] = (self.posStart_W[1] + 
                              (self.posDes_W[1] - self.posStart_W[1]) / (2 * np.pi) * 
                              (phase - np.sin(phase)))
        
        # Z-stretching logic
        if self.phi >= 0.98:
            self.zStretch += -0.002
            if self.zStretch < -0.05:
                self.zStretch = -0.05
        else:
            self.zStretch = 0.0
        
        # Z trajectory using Bezier curve
        self.pDesCur[2] = (self.posStart_W[2] + 
                          self.Trajectory(0.2, self.stepHeight, self.posDes_W[2] - self.posStart_W[2]) + 
                          self.zStretch)
    
    def Trajectory(self, phase: float, hei: float, length: float) -> float:
        """Generate trajectory using Bezier curves"""
        bezier = Bezier1D()
        bezier.P = [0.0] * 5 + [1.0] * 3  # Parameters from original code
        
        if self.phi < phase:
            return hei * bezier.getOut(self.phi / phase)
        else:
            s = bezier.getOut((1.4 - self.phi) / (1.4 - phase))
            return hei * s + length * (1.0 - s) if s > 0 else length