import numpy as np
from src.controller.robot_controller.utils.eul_w_filter import EulWFilter
class StateEstimator:
    def __init__(self, dtIn):
        self.dt = dtIn
        self.eul_w_filter = EulWFilter(dtIn)  # Assuming EulWFilter is defined as previously shown
        
        # Initialize matrices
        self.A = np.zeros((15, 15))  # State transition matrix
        self.B = np.zeros((15, 3))   # Control input matrix
        self.C = np.zeros((14, 15))  # Observation matrix
        
        # Initialize A matrix blocks
        self.A[0:3, 0:3] = np.eye(3)
        self.A[0:3, 3:6] = np.eye(3) * self.dt
        self.A[3:6, 3:6] = np.eye(3)
        self.A[6:12, 6:12] = np.eye(6)
        self.A[12:15, 12:15] = np.eye(3)
        self.A[3:6, 12:15] = np.eye(3) * self.dt
        
        # Initialize B matrix blocks
        self.B[3:6, 0:3] = np.eye(3) * self.dt
        
        # Initialize C matrix blocks
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
        
        # Initialize covariance and state matrices
        self.P = np.eye(15)  # State covariance matrix
        self.X = np.zeros(15)  # State vector
        self.P0 = np.eye(15)  # Initial state covariance
        self.X0 = np.zeros(15)  # Initial state
        
        # Process and measurement noise covariance matrices
        self.Q = np.zeros((15, 15))
        self.Qu = np.zeros((3, 3))
        self.R = np.zeros((14, 14))
        
        # Initialize other variables
        self.peB_old = np.zeros(3)
        self.peW_old = np.zeros(3)
        self.vCoM_LP = np.zeros(3)
        
        # Configure the Euler angle filter
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
        
    def init(self, Data):
        """Initialize the state estimator with initial data"""
        self.offYaw = Data.rpy[2]
        print(f"yawoffset: {Data.rpy[0]:.4f}, {Data.rpy[1]:.4f}, {Data.rpy[2]:.4f}")
        
        # Calculate rotation matrix (ignoring yaw for initialization)
        R = self.eul2Rot(Data.base_rpy[0], Data.base_rpy[1], 0)
        
        # Transform foot positions to world frame
        pWL = R @ Data.fe_l_pos_L
        pWR = R @ Data.fe_r_pos_L
        
        # Calculate z offset
        zOff = -(pWR[2] + pWL[2]) / 2.0 + 0.07
        
        # Initialize state vector
        self.X0 = np.zeros(15)
        self.X0[:3] = [0, 0, zOff]                     # Position
        self.X0[3:6] = [0, 0, 0]                       # Velocity
        self.X0[6:9] = [pWL[0], pWL[1], 0.07]          # Left foot position
        self.X0[9:12] = [pWR[0], pWR[1], 0.07]         # Right foot position
        self.X0[12:15] = [0, 0, 0]                     # Acceleration
        
        # Initialize covariance matrix
        self.P0 = np.zeros((15, 15))
        np.fill_diagonal(self.P0, 0.1)  # Diagonal elements set to 0.1
        
        self.flag_init = False
        self.X = self.X0.copy()
        self.P = self.P0.copy()
        
        # Kalman filter parameters
        self.TR_kh = 100
        self.TR_k = 100
        
        # Process noise covariances
        self.KF_Q_wPCoM = np.array([5e-6, 5e-6, 5e-6])
        self.KF_Q_wVCoM = np.array([1e-8, 2e-8, 1e-6])
        self.KF_Q_wfeW = np.array([
            [1e-6, 1e-6, 1e-6],
            [1e-6, 1e-6, 1e-6]
        ])
        self.KF_Q_waL = np.array([1e-8, 1e-8, 1e-8])
        
        # Measurement noise covariances
        self.KF_R_wfeL = np.array([
            [1e-7, 1e-7, 1e-10],
            [6e-7, 8e-8, 1e-11]
        ])
        self.KF_R_wdfeL = np.array([
            [1e-5, 2e-5, 1e-6],
            [1e-5, 2e-5, 1e-6]
        ])
        self.KF_R_wh = np.ones(2) * 1e-3
        
        self.KF_Q_waU = np.array([2e-4, 2e-4, 8e-4])

    def eul2Rot(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        # Implement your Euler to rotation matrix conversion here
        # This is a placeholder - replace with your actual implementation
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def set(self, Data):
        """Update the state estimator with new sensor data"""
        if self.flag_init:
            self.offYaw = Data.rpy[2]
        
        # Store acceleration and Euler angles
        self.acc = np.array([Data.baseAcc[0], Data.baseAcc[1], Data.baseAcc[2]])
        self.eul = np.array([Data.rpy[0], Data.rpy[1] - np.deg2rad(0.0), Data.rpy[2]])
        self.eul_woOff = np.array([self.eul[0], self.eul[1], self.eul[2] - self.offYaw])
        self.Rrpy_woOff = self.eul2Rot(*self.eul_woOff)
        
        # Store angular velocities
        self.omegaL = np.array([Data.baseAngVel[0], Data.baseAngVel[1], Data.baseAngVel[2]])
        self.omegaW = Data.base_omega_W
        
        # Run the Euler angle filter
        eul_woOff_ary = self.eul_woOff.tolist()
        omegaL_ary = self.omegaL.tolist()
        self.eul_w_filter.run(eul_woOff_ary, omegaL_ary)
        Eul_filtered, wL_filtered = self.eul_w_filter.getData()
        
        # Update filtered values
        self.eul_woOff = np.array(Eul_filtered)
        self.eul = np.array([self.eul_woOff[0], self.eul_woOff[1], self.eul_woOff[2] + self.offYaw])
        self.Rrpy_woOff = self.eul2Rot(*self.eul_woOff)
        self.Rrpy = self.eul2Rot(*self.eul)
        self.omegaL = np.array(wL_filtered)
        self.omegaW = self.Rrpy_woOff @ self.omegaL
        
        # Calculate free acceleration
        accTmp = np.array([self.acc[0], self.acc[1], self.acc[2]])
        self.freeAcc = self.eul2Rot(0.0, 0.0, self.offYaw).T @ accTmp
        
        # Store other state variables
        self.phi = Data.phi
        self.legState = Data.legState
        self.fe_l_pos_L = Data.fe_l_pos_L
        self.fe_r_pos_L = Data.fe_r_pos_L
        self.fe_l_vel_L = Data.fe_l_vel_L
        self.fe_r_vel_L = Data.fe_r_vel_L
        self.fe_l_pos_W = Data.fe_l_pos_W
        self.fe_r_pos_W = Data.fe_r_pos_W

    def getTrustRegion_wt_h(self):
        """Calculate trust region weights based on contact state and phase"""
        C = np.zeros(6)
        Cv = np.zeros(6)
        Ch = np.zeros(2)
        
        # Calculate phase-dependent weights
        if self.legState == "Stand":
            aa = [1.0, 1.0]
        else:
            aa = [self.phi * 5.0, self.phi * 5.0]
            aa = [min(a, 1.0) for a in aa]
            aa = [(1.0 - a) * 10000.0 + 1.0 for a in aa]
        
        # Calculate trust region weights for each contact point
        for i in range(2):
            C[i*3] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            C[i*3+1] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            C[i*3+2] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            
            Cv[i*3] = self.leg_contact[i] * aa[0] + (1 - self.leg_contact[i]) * self.TR_k
            Cv[i*3+1] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            Cv[i*3+2] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_k
            
            Ch[i] = self.leg_contact[i] + (1 - self.leg_contact[i]) * self.TR_kh
        
        # Create diagonal matrices
        self.Xi = np.diag(C)
        self.Xih = np.diag(Ch)
        self.Xiv = np.diag(Cv)
        
    def update(self):
        """Update the state estimate using Kalman filter"""
        # Update contact states based on leg state
        if self.legState == "LSt":
            self.leg_contact = [True, False]
        elif self.legState == "RSt":
            self.leg_contact = [False, True]
        else:
            self.leg_contact = [True, True]

        # Update foot positions in body frame
        self.peB = np.zeros((3, 2))
        self.peB[:, 0] = self.fe_l_pos_L
        self.peB[:, 1] = self.fe_r_pos_L

        # Get trust region weights
        self.getTrustRegion_wt_h()

        # Prediction step - setup process noise covariance
        diagQ = np.zeros(15)
        diagR = np.zeros(14)
        
        diagQ[0:3] = self.KF_Q_wPCoM
        diagQ[3:6] = self.KF_Q_wVCoM
        diagQ[6:12] = np.diag(self.Xi @ self.KF_Q_wfeW.reshape(6))  # Xi is diagonal matrix
        diagQ[12:15] = self.KF_Q_waL

        diagR[0:6] = self.KF_R_wfeL.reshape(6)
        diagR[6:12] = np.diag(self.Xiv @ self.KF_R_wdfeL.reshape(6))  # Xiv is diagonal matrix
        diagR[12:14] = np.diag(self.Xih @ self.KF_R_wh.reshape(2))    # Xih is diagonal matrix

        self.Qu = self.B @ np.diag(self.KF_Q_waU) @ self.B.T
        self.Q = np.diag(diagQ) + self.Qu
        self.R = np.diag(diagR)

        # State prediction
        self.X = self.A @ self.X + self.B @ self.freeAcc
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Measurement evaluation
        omegaLVec = np.array([self.wL_filtered[0], self.wL_filtered[1], self.wL_filtered[2]])
        self.pbW = self.Rrpy_woOff @ self.peB
        
        velTmp = np.zeros((3, 2))
        velTmp[:, 0] = self.fe_l_vel_L
        velTmp[:, 1] = self.fe_r_vel_L
        
        self.vbW = np.zeros((3, 2))
        self.Y = np.zeros(14)
        
        for i in range(2):
            # Calculate foot velocity in world frame
            cross_term = np.cross(omegaLVec, self.peB[:, i])
            self.vbW[:, i] = (self.leg_contact[i] * self.Rrpy_woOff @ (velTmp[:, i] + cross_term) + 
                            (1 - self.leg_contact[i]) * (-self.base_vel))
            
            # Height measurement
            self.Y[12 + i] = (self.leg_contact[i] * 0.07 + 
                            (1 - self.leg_contact[i]) * (self.base_pos[2] + self.pbW[2, i]))
        
        # Zero velocities if first update
        if np.all(self.peB_old == 0) or np.all(self.peW_old == 0):
            self.vbW.fill(0)
        
        # Update measurement vector
        self.Y[0:3] = -self.pbW[:, 0]
        self.Y[3:6] = -self.pbW[:, 1]
        self.Y[6:9] = self.vbW[:, 0]
        self.Y[9:12] = self.vbW[:, 1]
        
        # Store old values
        self.peB_old = self.peB.copy()
        self.peW_old = self.pbW.copy()

        # Kalman correction step
        S_inv = self.C @ self.P @ self.C.T + self.R
        S_inv = np.linalg.inv(S_inv)
        
        self.K = self.P @ self.C.T @ S_inv
        self.P = (np.eye(15) - self.K @ self.C) @ self.P
        self.X = self.X + self.K @ (self.Y - self.C @ self.X)

        # Reset to initial state if not started
        if not self.startFlag:
            self.X = self.X0.copy()
            self.P = self.P0.copy()

        if not self.flag_init:
            self.startFlag = True

        # Extract estimated states
        self.peW = np.zeros((3, 2))
        self.peW[:, 0] = self.X[6:9]
        self.peW[:, 1] = self.X[9:12]
        
        self.base_pos = self.X[0:3]
        self.base_vel = self.X[3:6]
        self.delta_acc = self.X[12:15]
        
        self.fe_l_pos_W = self.peW[:, 0]
        self.fe_r_pos_W = self.peW[:, 1]

        # Optional: Low-pass filter for CoM velocity
        # self.vCoM = np.array([
        #     self.vxLP_1O.run(self.X[3]),
        #     self.vyLP_1O.run(self.X[4]),
        #     self.vzLP_1O.run(self.X[5])
        # ])
        
    def get(self, Data):
        """Update the DataBus object with estimated states"""
        Data.base_pos_est = self.base_pos.copy()
        Data.base_vel_est = self.base_vel.copy()
        Data.fe_l_pos_W_est = self.fe_l_pos_W.copy()
        Data.fe_r_pos_W_est = self.fe_r_pos_W.copy()
        Data.delta_acc = self.delta_acc.copy()
        Data.eul_est = self.eul_woOff.copy()  # Without yaw offset
        Data.omegaW_est = self.omegaW.copy()

        Data.base_pos = Data.base_pos_est.copy()
        Data.base_vel = Data.base_vel_est.copy()
        
        # Update acceleration with delta_acc
        Data.baseAcc[0] += self.delta_acc[0]
        Data.baseAcc[1] += self.delta_acc[1]
        Data.baseAcc[2] += self.delta_acc[2]
        
        Data.base_rpy = Data.eul_est.copy()
        Data.base_omega_W = Data.omegaW_est.copy()
        Data.base_rot = self.Rrpy_woOff.copy()

        # Update position and velocity in q and dq
        Data.q[0:3] = Data.base_pos
        Data.dq[0:3] = Data.base_vel

        # Convert Euler angles to quaternion
        quatNow = self.eul2quat(Data.base_rpy[0], Data.base_rpy[1], Data.base_rpy[2])
        Data.q[3] = quatNow.x
        Data.q[4] = quatNow.y
        Data.q[5] = quatNow.z
        Data.q[6] = quatNow.w
        Data.dq[3:6] = Data.base_omega_W

        # For debugging/test purposes
        Data.AX = self.A @ self.X
        Data.BU = self.B @ self.freeAcc
        Data.freeAcc = self.freeAcc.copy()
        Data.CX = self.C @ self.X
        Data.Y = self.Y.copy()
        Data.pbW = self.pbW.flatten()
        Data.leg_contact[0] = self.leg_contact[0]
        Data.leg_contact[1] = self.leg_contact[1]

    def setF(self, Data):
        """Set force-related parameters from DataBus"""
        self.model_nv = Data.model_nv
        self.torJoint = np.zeros(self.model_nv - 6)
        for i in range(self.model_nv - 6):
            self.torJoint[i] = Data.motors_tor_cur[i]
        
        self.dyn_M = Data.dyn_M.copy()
        self.dyn_Non = Data.dyn_Non.copy()
        self.J_l = Data.J_l.copy()
        self.dJ_l = Data.dJ_l.copy()
        self.J_r = Data.J_r.copy()
        self.dJ_r = Data.dJ_r.copy()
        self.dq = Data.dq.copy()

    def updateF(self):
        """Update foot contact force estimates"""
        tauAll = np.zeros(self.model_nv)
        tauAll[6:] = self.torJoint
        
        # Calculate left foot force
        Jl_M_inv = self.J_l @ np.linalg.inv(self.dyn_M)
        self.FLest = -self.pseudoInv_SVD(Jl_M_inv @ self.J_l.T) @ (
            Jl_M_inv @ (tauAll - self.dyn_Non) + self.dJ_l @ self.dq
        )
        
        # Calculate right foot force
        Jr_M_inv = self.J_r @ np.linalg.inv(self.dyn_M)
        self.FRest = -self.pseudoInv_SVD(Jr_M_inv @ self.J_r.T) @ (
            Jr_M_inv @ (tauAll - self.dyn_Non) + self.dJ_r @ self.dq
        )

    def getF(self, Data):
        """Update DataBus with estimated foot forces"""
        Data.FL_est = self.FLest.copy()
        Data.FR_est = self.FRest.copy()

    def CphiFun(self, phi, s, W):
        """Phase-based weighting function"""
        return s * (self.erf(12 * phi / W - 6) + self.erf(12 * (1 - phi) / W - 6) - 1)

    def CzFun(self, pz, kp, kn):
        """Height-based weighting function"""
        if pz >= 0:
            return np.exp(-kp * pz**2)
        else:
            return np.exp(-kn * pz**2)

    def erf(self, xIn):
        """Sigmoid approximation of error function"""
        return 1 / (1 + np.exp(-xIn))

    def get_init(self):
        """Get initialization flag"""
        return self.flag_init

    def pseudoInv_SVD(self, matrix):
        """Compute pseudo-inverse using SVD"""
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        tol = 1e-6 * max(matrix.shape) * np.max(s)
        s_inv = np.array([1/x if x > tol else 0 for x in s])
        return vh.T @ np.diag(s_inv) @ u.T

    def eul2quat(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Implement your Euler to quaternion conversion here
        # This is a placeholder implementation
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return type('Quaternion', (), {'x': qx, 'y': qy, 'z': qz, 'w': qw})