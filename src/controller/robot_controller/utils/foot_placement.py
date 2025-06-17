import numpy as np
import math

class FootPlacement:
    def __init__(self,cfg):
        # Initialize parameters
        self.cfg = cfg
        self.kp_vx = cfg['kp_vx']
        self.kp_vy = cfg['kp_vy']
        self.kp_wz = cfg['kp_wz']
        self.hip_width = cfg['robot_size']['hip_width']
        self.leg_length = cfg['robot_size']['leg_length']
        self.step_height = cfg['step_height']
        self.tSwing = cfg['tSwing']
        
        # Initialize state variables
        self.phi = 0.0
        self.yawCur = 0.0
        self.theta0 = 0.0
        self.omegaZ_W = 0.0
        self.desWz_W = 0.0
        self.curV_W = np.zeros(3)
        self.desV_W = np.zeros(3)
        self.hipPos_W = np.zeros(3)
        self.STPos_W = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.posStart_W = np.zeros(3)
        self.posDes_W = np.zeros(3)
        self.pDesCur = np.zeros(3)
        self.zStretch = 0.0
        self.legState = "LSt"  # or "RSt"

    def getSwingPos(self):
        b = np.zeros(4)
        xNow = np.array([1, self.phi, self.phi**2, self.phi**3])

        # Create KP matrix and rotation matrix
        KP = np.zeros((3, 3))
        KP[0, 0] = self.kp_vx
        KP[1, 1] = self.kp_vy
        
        Rz = np.array([
            [math.cos(self.yawCur), -math.sin(self.yawCur), 0],
            [math.sin(self.yawCur), math.cos(self.yawCur), 0],
            [0, 0, 1]
        ])
        
        KP = Rz @ KP @ Rz.T

        # Calculate desired position for linear velocity
        self.posDes_W = (self.hipPos_W + KP @ (self.desV_W - self.curV_W) * (-1) + 
                        0.5 * self.tSwing * self.curV_W + 
                        self.curV_W * (1 - self.phi) * self.tSwing)

        # Calculate desired position for angular velocity
        thetaF = (self.yawCur + self.theta0 + self.omegaZ_W * (1 - self.phi) * self.tSwing + 
                 0.5 * self.omegaZ_W * self.tSwing + self.kp_wz * (self.omegaZ_W - self.desWz_W))
        
        self.posDes_W[0] += 0.5 * self.hip_width * (math.cos(thetaF) - math.cos(self.yawCur + self.theta0))
        self.posDes_W[1] += 0.5 * self.hip_width * (math.sin(thetaF) - math.sin(self.yawCur + self.theta0))

        # Foot-end position offsets
        xOff_L = -0.07
        yOff_L = 0.04
        zOff_W = -0.035

        self.posDes_W[2] = self.base_pos[2] - self.legLength + zOff_W

        # Calculate world frame offsets based on leg state
        if self.legState == "LSt":
            xOff_W = math.cos(self.yawCur) * xOff_L - math.sin(self.yawCur) * yOff_L
            yOff_W = math.sin(self.yawCur) * xOff_L + math.cos(self.yawCur) * yOff_L
        elif self.legState == "RSt":
            xOff_W = math.cos(self.yawCur) * xOff_L - math.sin(self.yawCur) * (-yOff_L)
            yOff_W = math.sin(self.yawCur) * xOff_L + math.cos(self.yawCur) * (-yOff_L)

        self.posDes_W[0] += xOff_W
        self.posDes_W[1] += yOff_W

        # Cycloid trajectories for x and y
        if self.phi < 1.0:
            self.pDesCur[0] = (self.posStart_W[0] + 
                              (self.posDes_W[0] - self.posStart_W[0]) / (2 * math.pi) * 
                              (2 * math.pi * self.phi - math.sin(2 * math.pi * self.phi)))
            self.pDesCur[1] = (self.posStart_W[1] + 
                              (self.posDes_W[1] - self.posStart_W[1]) / (2 * math.pi) * 
                              (2 * math.pi * self.phi - math.sin(2 * math.pi * self.phi)))

        # Handle z-stretching
        if self.phi >= 0.98:
            self.zStretch += -0.002
            if self.zStretch < -0.05:
                self.zStretch = -0.05
        else:
            self.zStretch = 0

        # Calculate z position using trajectory function
        self.pDesCur[2] = (self.posStart_W[2] + 
                          self.Trajectory(0.2, self.stepHeight, self.posDes_W[2] - self.posStart_W[2]) + 
                          self.zStretch)

    def Trajectory(self, phase, hei, len):
        class Bezier1D:
            def __init__(self):
                self.P = []
            
            def getOut(self, t):
                n = len(self.P) - 1
                result = 0.0
                for i in range(n + 1):
                    result += (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) * \
                             (t**i) * ((1 - t)**(n - i)) * self.P[i]
                return result

        Bswpid = Bezier1D()
        para0 = 5
        para1 = 3
        
        # Initialize control points
        Bswpid.P = [0.0] * para0 + [1.0] * para1

        if self.phi < phase:
            output = hei * Bswpid.getOut(self.phi / phase)
        else:
            s = Bswpid.getOut((1.4 - self.phi) / (1.4 - phase))
            output = hei * s + len * (1.0 - s) if s > 0 else len

        return output