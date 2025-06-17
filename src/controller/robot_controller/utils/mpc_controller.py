import numpy as np
from scipy.linalg import block_diag
import qpoases
from src.task.robot.data_bus import DataBus

# Constants
MPC_N = 10
CH = 3
NX = 12
NU = 13

NCFR_SINGLE = 4
NCFR = NCFR_SINGLE * 2

NCSTXYA = 1
NCSTXY_SINGLE = NCSTXYA * 4
NCSTXY = NCSTXY_SINGLE * 2

NCSTZA = 2
NCSTZ_SINGLE = NCSTZA * 4
NCSTZ = NCSTZ_SINGLE * 2
NC = NCFR + NCSTXY + NCSTZ

class MPC:
    def __init__(self, dt):
        self.m = 77.35
        self.g = -9.8
        self.miu = 0.5
        self.delta_foot = np.array([0.073, 0.125, 0.025, 0.025])
        
        self.max = np.array([1000.0, 1000.0, -3.0 * self.m * self.g, 20.0, 80.0, 100.0])
        self.min = np.array([-1000.0, -1000.0, 0.0, -20.0, -80.0, -100.0])
        
        # Initialize matrices
        self.Ac = [np.zeros((NX, NX)) for _ in range(MPC_N)]
        self.Bc = [np.zeros((NX, NU)) for _ in range(MPC_N)]
        self.A = [np.zeros((NX, NX)) for _ in range(MPC_N)]
        self.B = [np.zeros((NX, NU)) for _ in range(MPC_N)]
        
        self.Cc = np.zeros((NX, 1))
        self.C = np.zeros((NX, 1))
        
        self.Aqp = np.zeros((NX * MPC_N, NX))
        self.Aqp1 = np.zeros((NX * MPC_N, NX * MPC_N))
        self.Bqp1 = np.zeros((NX * MPC_N, NU * MPC_N))
        self.Bqp = np.zeros((NX * MPC_N, NU * CH))
        self.Cqp1 = np.zeros((NX * MPC_N, 1))
        self.Cqp = np.zeros((NX * MPC_N, 1))
        
        self.Ufe = np.zeros((NU * CH, 1))
        self.Ufe_pre = np.zeros((NU, 1))
        self.Xd = np.zeros((NX * MPC_N, 1))
        self.X_cur = np.zeros((NX, 1))
        self.X_cal = np.zeros((NX, 1))
        self.X_cal_pre = np.zeros((NX, 1))
        self.dX_cal = np.zeros((NX, 1))
        
        self.L = np.zeros((NX * MPC_N, NX * MPC_N))
        self.K = np.zeros((NU * CH, NU * CH))
        self.M = np.zeros((NU * CH, NU * CH))
        self.alpha = 0.0
        self.H = np.zeros((NU * CH, NU * CH))
        self.c = np.zeros((NU * CH, 1))
        
        self.u_low = np.zeros((NU * CH, 1))
        self.u_up = np.zeros((NU * CH, 1))
        self.As = np.zeros((NC * CH, NU * CH))
        self.bs = np.zeros((NC * CH, 1))
        
        self.pCoM = np.zeros((3, 1))
        self.pf2com = np.zeros((6, 1))
        self.pf2comd = np.zeros((6, 1))
        self.pe = np.zeros((6, 1))
        self.pf2comi = [np.zeros((6, 1)) for _ in range(MPC_N)]
        
        self.Ic = np.array([[12.61, 0, 0.37],
                           [0, 11.15, 0.01],
                           [0.37, 0.01, 2.15]])
        
        self.R_curz = [np.zeros((3, 3)) for _ in range(MPC_N)]
        self.R_cur = np.zeros((3, 3))
        self.R_w2f = np.zeros((3, 3))
        self.R_f2w = np.zeros((3, 3))
        
        self.legStateCur = 0
        self.legStateNext = 0
        self.legState = [0] * MPC_N
        self.dt = dt
        
        # QP problem setup
        self.QP = qpoases.QProblem(NU * CH, NC * CH)
        options = qpoases.Options()
        options.printLevel = qpoases.PL_LOW
        self.QP.setOptions(options)
        
        self.qp_H = np.zeros((NU * CH, NU * CH))
        self.qp_As = np.zeros((NC * CH, NU * CH))
        self.qp_c = np.zeros((NU * CH, 1))
        self.qp_lbA = np.zeros((NC * CH, 1))
        self.qp_ubA = np.zeros((NC * CH, 1))
        self.qp_lu = np.zeros((NU * CH, 1))
        self.qp_uu = np.zeros((NU * CH, 1))
        self.nWSR = 100
        self.cpu_time = 0.1
        self.xOpt_iniGuess = np.zeros((NU * CH, 1))
        
        self.qp_cpuTime = 0.0
        self.qp_Status = 0
        self.qp_nWSR = 0
        self.EN = False
    
    def set_weight(self, u_weight, L_diag, K_diag):
        self.alpha = u_weight
        
        # Create L matrix
        L_diag_N = np.zeros((1, NX * MPC_N))
        for i in range(MPC_N):
            L_diag_N[0, i*NX:(i+1)*NX] = L_diag
        
        self.L = np.diag(L_diag_N[0])
        
        # Create K matrix
        K_diag_N = np.zeros((1, NU * CH))
        for i in range(CH):
            K_diag_N[0, i*NU:(i+1)*NU] = K_diag
        
        self.K = np.diag(K_diag_N[0])
        
        # Apply rotation matrices
        for i in range(MPC_N):
            R = self.R_curz[i]
            self.L[i*NX+3:i*NX+6, i*NX+3:i*NX+6] = R @ self.L[i*NX+3:i*NX+6, i*NX+3:i*NX+6] @ R.T
            self.L[i*NX+6:i*NX+9, i*NX+6:i*NX+9] = R @ self.L[i*NX+6:i*NX+9, i*NX+6:i*NX+9] @ R.T
            self.L[i*NX+9:i*NX+12, i*NX+9:i*NX+12] = R @ self.L[i*NX+9:i*NX+12, i*NX+9:i*NX+12] @ R.T
        
        for i in range(CH):
            R = self.R_curz[i]
            self.K[i*NU:i*NU+3, i*NU:i*NU+3] = R @ self.K[i*NU:i*NU+3, i*NU:i*NU+3] @ R.T
            self.K[i*NU+3:i*NU+6, i*NU+3:i*NU+6] = R @ self.K[i*NU+3:i*NU+6, i*NU+3:i*NU+6] @ R.T
            self.K[i*NU+6:i*NU+9, i*NU+6:i*NU+9] = R @ self.K[i*NU+6:i*NU+9, i*NU+6:i*NU+9] @ R.T
            self.K[i*NU+9:i*NU+12, i*NU+9:i*NU+12] = R @ self.K[i*NU+9:i*NU+12, i*NU+9:i*NU+12] @ R.T
    
    def data_bus_read(self, data_bus):
        # Set current state
        self.X_cur[:3] = data_bus.base_rpy.reshape(3, 1)
        self.X_cur[3:6] = data_bus.q[:3].reshape(3, 1)
        self.X_cur[6:9] = data_bus.dq[3:6].reshape(3, 1)
        self.X_cur[9:12] = data_bus.dq[:3].reshape(3, 1)
        
        if self.EN:
            # Set desired state
            for i in range(MPC_N - 1):
                self.Xd[i*NX:(i+1)*NX] = self.Xd[(i+1)*NX:(i+2)*NX]
            
            self.Xd[(MPC_N-1)*NX:(MPC_N-1)*NX+3] = data_bus.js_eul_des[:3].reshape(3, 1)
            self.Xd[(MPC_N-1)*NX+3:(MPC_N-1)*NX+6] = data_bus.js_pos_des[:3].reshape(3, 1)
            self.Xd[(MPC_N-1)*NX+6:(MPC_N-1)*NX+9] = data_bus.js_omega_des[:3].reshape(3, 1)
            self.Xd[(MPC_N-1)*NX+9:(MPC_N-1)*NX+12] = data_bus.js_vel_des[:3].reshape(3, 1)
        else:
            for i in range(MPC_N):
                self.Xd[i*NX:i*NX+3] = self.X_cur[:3]
                self.Xd[i*NX+3:i*NX+6] = self.X_cur[3:6]
                self.Xd[i*NX+6:i*NX+9] = self.X_cur[6:9]
                self.Xd[i*NX+9:i*NX+12] = self.X_cur[9:12]
        
        # Set rotation matrices
        self.R_cur = self.eul2Rot(self.X_cur[0], self.X_cur[1], self.X_cur[2])
        for i in range(MPC_N):
            self.R_curz[i] = self.Rz3(self.X_cur[2])
        
        self.pCoM = self.X_cur[3:6]
        self.pe[:3] = data_bus.fe_l_pos_W.reshape(3, 1)
        self.pe[3:6] = data_bus.fe_r_pos_W.reshape(3, 1)
        
        self.pf2com[:3] = self.pe[:3] - self.pCoM
        self.pf2com[3:6] = self.pe[3:6] - self.pCoM
        self.pf2comd[:3] = self.pe[:3] - self.Xd[3:6]
        self.pf2comd[3:6] = self.pe[3:6] - self.Xd[3:6]
        
        # Set leg states
        self.legStateCur = data_bus.legState
        self.legStateNext = data_bus.legStateNext
        for i in range(MPC_N):
            aa = i * self.dt / 0.4
            phip = data_bus.phi + aa
            self.legState[i] = self.legStateNext if phip > 1 else self.legStateCur
        
        # Set foot rotation matrices
        R_slop = self.eul2Rot(data_bus.slop[0], data_bus.slop[1], data_bus.slop[2])
        if self.legStateCur == LegState.RSt:
            self.R_f2w = data_bus.fe_r_rot_W
        elif self.legStateCur == LegState.LSt:
            self.R_f2w = data_bus.fe_l_rot_W
        else:
            self.R_f2w = R_slop
        self.R_w2f = self.R_f2w.T
    
    def cal(self):
        if self.EN:
            # QP preparation
            for i in range(MPC_N):
                self.Ac[i][:3, 6:9] = self.R_curz[i].T
                self.Ac[i][3:6, 9:12] = np.eye(3)
                self.A[i] = np.eye(NX) + self.dt * self.Ac[i]
            
            for i in range(MPC_N):
                self.pf2comi[i] = self.pf2com
                Ic_W_inv = np.linalg.inv(self.R_curz[i] @ self.Ic @ self.R_curz[i].T)
                self.Bc[i][6:9, :3] = Ic_W_inv @ self.CrossProduct_A(self.pf2comi[i][:3])
                self.Bc[i][6:9, 3:6] = Ic_W_inv
                self.Bc[i][6:9, 6:9] = Ic_W_inv @ self.CrossProduct_A(self.pf2comi[i][3:6])
                self.Bc[i][6:9, 9:12] = Ic_W_inv
                self.Bc[i][9:12, :3] = np.eye(3) / self.m
                self.Bc[i][9:12, 6:9] = np.eye(3) / self.m
                self.Bc[i][NX-1, NU-1] = 1.0 / self.m
                self.B[i] = self.dt * self.Bc[i]
            
            # Construct Aqp, Aqp1, Bqp1, Bqp matrices
            # ... (similar to C++ implementation)
            
            # Construct delta_U
            delta_U = np.zeros((NU * CH, 1))
            for i in range(CH):
                if self.legState[i] == LegState.LSt:
                    delta_U[i*NU+2] = self.m * self.g
                elif self.legState[i] == LegState.RSt:
                    delta_U[i*NU+8] = self.m * self.g
                else:
                    delta_U[i*NU+2] = 0.5 * self.m * self.g
                    delta_U[i*NU+8] = 0.5 * self.m * self.g
            
            # Construct H and c matrices
            self.H = 2 * (self.Bqp.T @ self.L @ self.Bqp + self.alpha * self.K) + 1e-10 * np.eye(NX * MPC_N)
            self.c = 2 * self.Bqp.T @ self.L @ (self.Aqp @ self.X_cur - self.Xd) + 2 * self.alpha * self.K @ delta_U
            
            # Construct friction constraints
            Asfr111 = np.array([
                [-1.0, 0.0, -1.0/np.sqrt(2.0)*self.miu],
                [1.0, 0.0, -1.0/np.sqrt(2.0)*self.miu],
                [0.0, -1.0, -1.0/np.sqrt(2.0)*self.miu],
                [0.0, 1.0, -1.0/np.sqrt(2.0)*self.miu]
            ])
            Asfr11 = Asfr111 @ self.R_w2f
            Asfr1 = np.zeros((NCFR, NU))
            Asfr1[:NCFR_SINGLE, :3] = Asfr11
            Asfr1[NCFR_SINGLE:, 6:9] = Asfr11
            
            Asfr = np.zeros((NCFR * CH, NU * CH))
            for i in range(CH):
                Asfr[i*NCFR:(i+1)*NCFR, i*NU:(i+1)*NU] = Asfr1
            
            # Construct moment constraints
            # ... (similar to C++ implementation)
            
            # Combine all constraints
            self.As[:NCFR*CH] = Asfr
            self.As[NCFR*CH:NCFR*CH+NCSTXY*CH] = Astxy
            self.As[NCFR*CH+NCSTXY*CH:] = Astz
            
            # QP problem setup
            Guess_value = np.zeros((NU * CH, 1))
            for i in range(CH):
                if self.legState[i] == LegState.DSt:
                    Guess_value[i*NU+2] = -0.5 * self.m * self.g
                    Guess_value[i*NU+8] = -0.5 * self.m * self.g
                    Guess_value[i*NU+12] = self.m * self.g
                    for j in range(6):
                        self.u_low[i*NU+j] = self.min[j]
                        self.u_low[i*NU+j+6] = self.min[j]
                        self.u_up[i*NU+j] = self.max[j]
                        self.u_up[i*NU+j+6] = self.max[j]
                    self.u_low[i*NU+12] = self.m * self.g
                    self.u_up[i*NU+12] = self.m * self.g
                elif self.legState[i] == LegState.LSt:
                    # ... (similar to C++ implementation)
                elif self.legState[i] == LegState.RSt:
                    # ... (similar to C++ implementation)
            
            # Set QP bounds
            lbA = -1e7 * np.ones((NC * CH, 1))
            ubA = 1e7 * np.ones((NC * CH, 1))
            
            for i in range(CH):
                if self.legState[i] == LegState.DSt:
                    ubA[i*NCFR:(i+1)*NCFR] = 0
                    ubA[NCFR*CH + i*NCSTXY:NCFR*CH + (i+1)*NCSTXY] = 0
                    ubA[NCFR*CH + NCSTXY*CH + i*NCSTZ:NCFR*CH + NCSTXY*CH + (i+1)*NCSTZ] = 0
                elif self.legState[i] == LegState.LSt:
                    # ... (similar to C++ implementation)
                elif self.legState[i] == LegState.RSt:
                    # ... (similar to C++ implementation)
            
            # Solve QP problem
            nWSR = 1000000
            cpu_time = self.dt
            
            res = self.QP.init(self.qp_H, self.qp_c, self.qp_As, 
                              self.qp_lu, self.qp_uu, 
                              self.qp_lbA, self.qp_ubA, 
                              nWSR, cpu_time, self.xOpt_iniGuess)
            
            self.qp_Status = qpoases.getSimpleStatus(res)
            self.qp_nWSR = nWSR
            self.qp_cpuTime = cpu_time
            
            if res != qpoases.SUCCESSFUL_RETURN:
                print("QP failed!")
            
            xOpt = np.zeros((NU * CH, 1))
            self.QP.getPrimalSolution(xOpt)
            if self.qp_Status == 0:
                self.Ufe = xOpt
            
            # Calculate next state
            self.dX_cal = self.Ac[0] @ self.X_cur + self.Bc[0] @ self.Ufe[:NU]
            delta_X = np.zeros((NX, 1))
            for i in range(3):
                delta_X[i] = 0.5 * self.dX_cal[i+6] * self.dt**2
                delta_X[i+3] = 0.5 * self.dX_cal[i+9] * self.dt**2
                delta_X[i+6] = self.dX_cal[i+6] * self.dt
                delta_X[i+9] = self.dX_cal[i+9] * self.dt
            
            self.X_cal = (self.Aqp @ self.X_cur + self.Bqp @ self.Ufe)[:NX] + delta_X
            self.Ufe_pre = self.Ufe[:NU]
            self.QP.reset()
        else:
            self.Ufe = np.zeros((NU * CH, 1))
            self.Ufe[2] = -0.5 * self.m * self.g
            self.Ufe[8] = -0.5 * self.m * self.g
            self.Ufe[12] = self.m * self.g
            self.Ufe_pre = np.zeros((NU, 1))
    
    def data_bus_write(self, data_bus):
        data_bus.Xd = self.Xd
        data_bus.X_cur = self.X_cur
        data_bus.fe_react_tau_cmd = self.Ufe
        data_bus.X_cal = self.X_cal
        data_bus.dX_cal = self.dX_cal
        
        data_bus.qp_nWSR_MPC = self.nWSR
        data_bus.qp_cpuTime_MPC = self.cpu_time
        data_bus.qpStatus_MPC = self.qp_Status
        
        data_bus.Fr_ff = self.Ufe[:12]
        
        k = 5
        data_bus.des_ddq[:2] = self.dX_cal[9:11].reshape(2, 1)
        data_bus.des_ddq[5] = k * (self.Xd[8] - data_bus.dq[5])
        
        data_bus.des_dq[:3] = self.Xd[9:12].reshape(3, 1)
        data_bus.des_dq[3:5] = np.zeros((2, 1))
        data_bus.des_dq[5] = self.Xd[8]
        
        data_bus.des_delta_q[:2] = data_bus.des_dq[:2] * self.dt
        data_bus.des_delta_q[5] = data_bus.des_dq[5] * self.dt
        
        data_bus.base_rpy_des = np.array([0.005, 0.00, self.Xd[2]])
        data_bus.base_pos_des = self.Xd[3:6].reshape(3, 1)
    
    def enable(self):
        self.EN = True
    
    def disable(self):
        self.EN = False
    
    def get_ENA(self):
        return self.EN
    
    def Rz3(self, yaw):
        # Create rotation matrix around Z axis
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
    
    def CrossProduct_A(self, v):
        # Create cross product matrix from vector
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])