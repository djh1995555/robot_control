import numpy as np
from enum import Enum, auto

class LegState(Enum):
    LSt = auto()  # Left stance
    RSt = auto()  # Right stance
    DSt = auto()  # Double stance

class MotionState(Enum):
    Stand = auto()
    Walk = auto()
    Walk2Stand = auto()

class GaitScheduler:
    def __init__(self, tSwing: float = 0.4, dt: float = 0.001):
        """
        Initialize the gait scheduler
        
        Args:
            tSwing: Swing phase duration (seconds)
            dt: Time step (seconds)
        """
        self.tSwing = tSwing
        self.dt = dt
        self.phi = 0.0
        self.isIni = False
        self.start_walk = True
        self.firstleg = LegState.LSt
        self.legState = LegState.DSt
        self.legStateNext = self.firstleg
        self.motionState = MotionState.Stand
        self.enableNextStep = False
        self.touchDown = False
        self.stepNumDes = 1
        self.stepNumCur = 0
        self.FzThrehold = 100.0
        self.Fz_L_m = 0.0
        self.Fz_R_m = 0.0
        
        # Initialize vectors and matrices
        self.FLest = np.zeros(6)
        self.FRest = np.zeros(6)
        self.torJoint = np.zeros(0)
        self.fe_r_pos_W = np.zeros(3)
        self.fe_l_pos_W = np.zeros(3)
        self.swingStartPos_W = np.zeros(3)
        self.posHip_W = np.zeros(3)
        self.posST_W = np.zeros(3)
        self.hip_r_pos_W = np.zeros(3)
        self.hip_l_pos_W = np.zeros(3)
        self.dq = np.zeros(0)
        self.stanceStartPos_W = np.zeros(3)
        self.fe_r_rot_W = np.eye(3)
        self.fe_l_rot_W = np.eye(3)
        self.dyn_M = np.zeros((0, 0))
        self.dyn_Non = np.zeros(0)
        self.J_l = np.zeros((6, 0))
        self.J_r = np.zeros((6, 0))
        self.dJ_l = np.zeros((6, 0))
        self.dJ_r = np.zeros((6, 0))
        self.theta0 = 0.0
        self.model_nv = 0

    def dataBusRead(self, robotState):
        """Read data from DataBus"""
        if self.motionState != MotionState.Stand and self.stepNumCur == 0:
            self.legState = self.firstleg
        
        self.model_nv = robotState.model_nv
        self.torJoint = np.zeros(self.model_nv - 6)
        for i in range(self.model_nv - 6):
            self.torJoint[i] = robotState.motors_tor_cur[i]
        
        self.dyn_M = robotState.dyn_M
        self.dyn_Non = robotState.dyn_Non
        self.J_l = robotState.J_l
        self.dJ_l = robotState.dJ_l
        self.J_r = robotState.J_r
        self.dJ_r = robotState.dJ_r
        self.Fz_L_m = robotState.fL[2]
        self.Fz_R_m = robotState.fR[2]
        self.hip_l_pos_W = robotState.hip_l_pos_W
        self.hip_r_pos_W = robotState.hip_r_pos_W
        self.fe_r_pos_W = robotState.fe_r_pos_W
        self.fe_l_pos_W = robotState.fe_l_pos_W
        self.fe_l_rot_W = robotState.fe_l_rot_W
        self.fe_r_rot_W = robotState.fe_r_rot_W
        self.dq = robotState.dq
        self.motionState = robotState.motionState

    def dataBusWrite(self, robotState):
        """Write data to DataBus"""
        robotState.tSwing = self.tSwing
        robotState.swingStartPos_W = self.swingStartPos_W.copy()
        robotState.stanceDesPos_W = self.stanceStartPos_W.copy()
        robotState.posHip_W = self.posHip_W.copy()
        robotState.posST_W = self.posST_W.copy()
        robotState.theta0 = self.theta0
        robotState.legState = self.legState
        robotState.legStateNext = self.legStateNext
        robotState.phi = self.phi
        robotState.FL_est = self.FLest.copy()
        robotState.FR_est = self.FRest.copy()
        
        if self.legState == LegState.LSt:
            robotState.stance_fe_pos_cur_W = self.fe_l_pos_W.copy()
            robotState.stance_fe_rot_cur_W = self.fe_l_rot_W.copy()
        elif self.legState == LegState.RSt:
            robotState.stance_fe_pos_cur_W = self.fe_r_pos_W.copy()
            robotState.stance_fe_rot_cur_W = self.fe_r_rot_W.copy()
        
        robotState.motionState = self.motionState

    def pseudoInv_SVD(self, matrix: np.ndarray) -> np.ndarray:
        """Compute pseudo-inverse using SVD"""
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = np.max(s) * 1e-10
        s_inv = np.array([1/x if x > threshold else 0 for x in s])
        return vh.T @ np.diag(s_inv) @ u.T

    def step(self):
        """Perform one step of gait scheduling"""
        # Compute estimated foot forces
        tauAll = np.zeros(self.model_nv)
        tauAll[6:] = self.torJoint
        
        term_l = (self.J_l @ np.linalg.inv(self.dyn_M) @ (tauAll - self.dyn_Non) + 
                 self.dJ_l @ self.dq)
        self.FLest = -self.pseudoInv_SVD(
            self.J_l @ np.linalg.inv(self.dyn_M) @ self.J_l.T
        ) @ term_l
        
        term_r = (self.J_r @ np.linalg.inv(self.dyn_M) @ (tauAll - self.dyn_Non) + 
                 self.dJ_r @ self.dq)
        self.FRest = -self.pseudoInv_SVD(
            self.J_r @ np.linalg.inv(self.dyn_M) @ self.J_r.T
        ) @ term_r

        dPhi = 0.0

        # State machine transitions
        if self.motionState == MotionState.Walk2Stand:
            self.enableNextStep = False
            self.start_walk = False
            if self.touchDown:
                self.motionState = MotionState.Stand

        # Phase update
        if self.motionState == MotionState.Stand:
            dPhi = 0
            self.phi = 0
            self.isIni = False
            self.enableNextStep = False
            self.stepNumCur = 0
        elif self.motionState == MotionState.Walk:
            self.enableNextStep = True
            dPhi = 1.0 / self.tSwing * self.dt
        elif self.motionState == MotionState.Walk2Stand:
            dPhi = 1.0 / self.tSwing * self.dt

        self.phi += dPhi
        if self.enableNextStep:
            self.touchDown = False

        # Initialize gait if starting to walk
        if not self.isIni and self.start_walk:
            self.isIni = True
            self.legState = self.firstleg
            if self.legState == LegState.LSt:
                self.swingStartPos_W = self.fe_r_pos_W.copy()
                self.stanceStartPos_W = self.fe_l_pos_W.copy()
            else:
                self.swingStartPos_W = self.fe_l_pos_W.copy()
                self.stanceStartPos_W = self.fe_r_pos_W.copy()

        # Leg state transitions
        if (self.legState == LegState.LSt and 
            self.FRest[2] >= 280 and self.phi >= 0.6):
            if self.enableNextStep:
                self.legState = LegState.RSt
                self.swingStartPos_W = self.fe_l_pos_W.copy()
                self.stanceStartPos_W = self.fe_r_pos_W.copy()
                self.phi = 0
                self.stepNumCur += 1
        elif (self.legState == LegState.RSt and 
              self.FLest[2] >= 280 and self.phi >= 0.6):
            if self.enableNextStep:
                self.legState = LegState.LSt
                self.swingStartPos_W = self.fe_r_pos_W.copy()
                self.stanceStartPos_W = self.fe_l_pos_W.copy()
                self.phi = 0
                self.stepNumCur += 1

        # Touchdown detection when not enabling next step
        if not self.enableNextStep:
            if self.legState == LegState.LSt and self.FRest[2] >= 200:
                self.touchDown = True
                self.stepNumCur += 1
                self.legState = LegState.DSt
            if self.legState == LegState.RSt and self.FLest[2] >= 200:
                self.touchDown = True
                self.stepNumCur += 1
                self.legState = LegState.DSt

        # Phase clamping
        if self.phi >= 1:
            self.phi = 1

        # Update hip and stance positions based on current leg state
        if self.legState == LegState.LSt:
            self.posHip_W = self.hip_r_pos_W.copy()
            self.posST_W = self.fe_l_pos_W.copy()
            self.theta0 = -np.pi * 0.5
            if self.motionState == MotionState.Walk:
                self.legStateNext = LegState.RSt
            elif self.motionState == MotionState.Walk2Stand:
                self.legStateNext = LegState.DSt
        elif self.legState == LegState.RSt:
            self.posHip_W = self.hip_l_pos_W.copy()
            self.posST_W = self.fe_r_pos_W.copy()
            self.theta0 = np.pi * 0.5
            if self.motionState == MotionState.Walk:
                self.legStateNext = LegState.LSt
            elif self.motionState == MotionState.Walk2Stand:
                self.legStateNext = LegState.DSt
        else:
            self.posHip_W = self.hip_l_pos_W.copy()
            self.posST_W = self.fe_r_pos_W.copy()
            self.theta0 = np.pi * 0.5
            self.legStateNext = LegState.DSt

    def start(self):
        """Start the walking motion"""
        self.start_walk = True

    def stop(self):
        """Stop the walking motion"""
        self.start_walk = False
        self.motionState = MotionState.Stand