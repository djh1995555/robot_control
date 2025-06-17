import numpy as np
from typing import List
from enum import Enum
from src.controller.robot_controller.utils.priority_tasks import PriorityTasks
# import qpoases
from qpsolvers import solve_qp

class LegState(Enum):
    LSt = 0
    RSt = 1
    DSt = 2

class MotionState(Enum):
    Stand = 0
    Walk = 1
    Walk2Stand = 2

class WBCPriority:
    def __init__(self, model_nv: int, qp_nv: int, qp_nc: int, miu: float, dt: float):
        self.model_nv = model_nv
        self.miu = miu
        self.time_step = dt
        self.QP_nv = qp_nv
        self.QP_nc = qp_nc
        
        # Initialize matrices and vectors
        self.Sf = np.zeros((6, model_nv))
        self.Sf[:6, :6] = np.eye(6)
        
        self.St_qpV2 = np.zeros((model_nv, model_nv - 6))
        self.St_qpV2[6:, :] = np.eye(model_nv - 6)
        
        self.St_qpV1 = np.zeros((model_nv, 6))
        self.St_qpV1[:6, :6] = np.eye(6)
        
        # Contact force limits
        self.f_z_low = 10
        self.f_z_upp = 1400
        
        # Torque limits
        self.tau_upp_stand_L = np.array([15, 30, 40])
        self.tau_low_stand_L = np.array([-15, -30, -40])
        self.tau_upp_walk_L = np.array([15, 40, 40])
        self.tau_low_walk_L = np.array([-15, -40, -40])
        
        # QP problem setup
        # self.QP_prob = qpoases.QProblem(qp_nv, qp_nc)
        # options = qpoases.Options()
        # options.setToMPC()
        # options.printLevel = qpoases.PL_LOW
        # self.QP_prob.setOptions(options)
        
        # Initialize vectors
        self.eigen_xOpt = np.zeros(qp_nv)
        self.eigen_ddq_Opt = np.zeros(model_nv)
        self.eigen_fr_Opt = np.zeros(12)
        self.eigen_tau_Opt = np.zeros(model_nv - 6)
        
        self.delta_q_final_kin = np.zeros(model_nv)
        self.dq_final_kin = np.zeros(model_nv)
        self.ddq_final_kin = np.zeros(model_nv)
        
        self.base_rpy_cur = np.zeros(3)
        
        # Initialize tasks
        self.kin_tasks_walk = PriorityTasks()
        self.kin_tasks_stand = PriorityTasks()
        
        # Add walk tasks
        walk_tasks = ["static_Contact", "Roll_Pitch_Yaw_Pz", "RedundantJoints", 
                     "PxPy", "SwingLeg", "HandTrackJoints", "PosRot"]
        for task in walk_tasks:
            self.kin_tasks_walk.add_task(task)
        
        walk_order = ["static_Contact", "PosRot", "SwingLeg", "RedundantJoints", "HandTrackJoints"]
        self.kin_tasks_walk.build_priority(walk_order)
        
        # Add stand tasks
        stand_tasks = ["static_Contact", "CoMTrack", "HandTrackJoints", "HipRPY", 
                      "HeadRP", "Pz", "CoMXY_HipRPY", "Roll_Pitch_Yaw", "fixedWaist"]
        for task in stand_tasks:
            self.kin_tasks_stand.add_task(task)
        
        stand_order = ["static_Contact", "CoMXY_HipRPY", "Pz", "HandTrackJoints", "HeadRP"]
        self.kin_tasks_stand.build_priority(stand_order)
        
        # QP problem matrices
        self.qp_H = np.zeros((qp_nv, qp_nv))
        self.qp_A = np.zeros((qp_nc, qp_nv))
        self.qp_g = np.zeros(qp_nv)
        self.qp_lbA = np.zeros(qp_nc)
        self.qp_ubA = np.zeros(qp_nc)
        self.xOpt_iniGuess = np.zeros(qp_nv)
        
        # State variables
        self.legStateCur = LegState.DSt
        self.motionStateCur = MotionState.Stand
        
        # Weight matrices
        self.Q1 = np.eye(12)
        self.Q2 = np.eye(6)
        
    def data_bus_read(self, robot_state):
        # Foot-end offset posture
        self.fe_L_rot_L_off = robot_state.fe_L_rot_L_off
        self.fe_R_rot_L_off = robot_state.fe_R_rot_L_off
        
        # Desired values
        self.base_rpy_des = robot_state.base_rpy_des
        self.base_rpy_cur = np.array([robot_state.rpy[0], robot_state.rpy[1], robot_state.rpy[2]])
        self.base_pos_des = robot_state.base_pos_des
        self.swing_fe_pos_des_W = robot_state.swing_fe_pos_des_W
        self.swing_fe_rpy_des_W = robot_state.swing_fe_rpy_des_W
        self.stance_fe_pos_cur_W = robot_state.stance_fe_pos_cur_W
        self.stance_fe_rot_cur_W = robot_state.stance_fe_rot_cur_W
        self.stanceDesPos_W = robot_state.stanceDesPos_W
        self.hd_l_pos_cur_W = robot_state.hd_l_pos_W
        self.hd_r_pos_cur_W = robot_state.hd_r_pos_W
        self.hd_l_rot_cur_W = robot_state.hd_l_rot_W
        self.hd_r_rot_cur_W = robot_state.hd_r_rot_W
        self.fe_l_pos_cur_W = robot_state.fe_l_pos_W
        self.fe_r_pos_cur_W = robot_state.fe_r_pos_W
        self.fe_l_rot_cur_W = robot_state.fe_l_rot_W
        self.fe_r_rot_cur_W = robot_state.fe_r_rot_W
        self.des_ddq = robot_state.des_ddq
        self.des_dq = robot_state.des_dq
        self.des_delta_q = robot_state.des_delta_q
        self.des_q = robot_state.des_q
        
        # State update
        self.J_base = robot_state.J_base
        self.dJ_base = robot_state.dJ_base
        self.base_rot = robot_state.base_rot
        self.base_pos = robot_state.base_pos
        self.hip_link_pos = robot_state.hip_link_pos
        self.hip_link_rot = robot_state.hip_link_rot
        self.J_hip_link = robot_state.J_hip_link
        
        self.Jfe = np.zeros((12, self.model_nv))
        self.Jfe[:6, :] = robot_state.J_l
        self.Jfe[6:, :] = robot_state.J_r
        
        self.dJfe = np.zeros((12, self.model_nv))
        self.dJfe[:6, :] = robot_state.dJ_l
        self.dJfe[6:, :] = robot_state.dJ_r
        
        self.J_hd_l = robot_state.J_hd_l
        self.J_hd_r = robot_state.J_hd_r
        self.dJ_hd_l = robot_state.dJ_hd_l
        self.dJ_hd_r = robot_state.dJ_hd_r
        self.Fr_ff = robot_state.Fr_ff
        self.dyn_M = robot_state.dyn_M
        self.dyn_M_inv = robot_state.dyn_M_inv
        self.dyn_Ag = robot_state.dyn_Ag
        self.dyn_dAg = robot_state.dyn_dAg
        self.dyn_Non = robot_state.dyn_Non
        self.dq = robot_state.dq
        self.q = robot_state.q
        self.legStateCur = robot_state.legState
        self.motionStateCur = robot_state.motionState
        
        if self.legStateCur == LegState.LSt:
            self.Jc = robot_state.J_l
            self.dJc = robot_state.dJ_l
            self.Jsw = robot_state.J_r
            self.dJsw = robot_state.dJ_r
            self.fe_pos_sw_W = robot_state.fe_r_pos_W
            self.fe_rot_sw_W = robot_state.fe_r_rot_W
        else:
            self.Jc = robot_state.J_r
            self.dJc = robot_state.dJ_r
            self.Jsw = robot_state.J_l
            self.dJsw = robot_state.dJ_l
            self.fe_pos_sw_W = robot_state.fe_l_pos_W
            self.fe_rot_sw_W = robot_state.fe_l_rot_W
        
        self.Jcom = robot_state.Jcom_W
        self.pCoMCur = robot_state.pCoM_W
        
    def data_bus_write(self, robot_state):
        robot_state.wbc_ddq_final = self.eigen_ddq_Opt
        robot_state.wbc_tauJointRes = self.tauJointRes
        robot_state.wbc_FrRes = self.eigen_fr_Opt
        # robot_state.qp_cpuTime = self.cpu_time
        # robot_state.qp_nWSR = self.nWSR
        # robot_state.qp_status = self.qpStatus
        
        robot_state.wbc_delta_q_final = self.delta_q_final_kin
        robot_state.wbc_dq_final = self.dq_final_kin
        robot_state.wbc_ddq_final = self.ddq_final_kin
        
        # robot_state.qp_status = self.qpStatus
        # robot_state.qp_nWSR = self.nWSR
        # robot_state.qp_cpuTime = self.cpu_time
        
    def compute_tau(self):
        # Construct QP problem matrices
        eigen_qp_A1 = np.zeros((6, self.QP_nv))
        eigen_qp_A1[:6, :6] = self.Sf @ self.dyn_M @ self.St_qpV1
        eigen_qp_A1[:6, 6:18] = -self.Sf @ self.Jfe.T
        
        eqRes = -self.Sf @ self.dyn_M @ self.ddq_final_kin - self.Sf @ self.dyn_Non + self.Sf @ self.Jfe.T @ self.Fr_ff
        
        if self.motionStateCur == MotionState.Stand:
            Rfe = self.fe_l_rot_cur_W
        else:
            Rfe = self.stance_fe_rot_cur_W
            
        Mw2b = np.zeros((12, 12))
        Mw2b[:3, :3] = Rfe.T
        Mw2b[3:6, 3:6] = Rfe.T
        Mw2b[6:9, 6:9] = Rfe.T
        Mw2b[9:12, 9:12] = Rfe.T
        
        W = np.zeros((16, 12))
        W[0, 0] = 1
        W[0, 2] = np.sqrt(2) / 2.0 * self.miu
        W[1, 0] = -1
        W[1, 2] = np.sqrt(2) / 2.0 * self.miu
        W[2, 1] = 1
        W[2, 2] = np.sqrt(2) / 2.0 * self.miu
        W[3, 1] = -1
        W[3, 2] = np.sqrt(2) / 2.0 * self.miu
        W[4:8, 2:6] = np.eye(4)
        W[8:16, 6:12] = W[:8, :6]
        W = W @ Mw2b
        
        f_low = np.zeros(16)
        f_upp = np.zeros(16)
        
        if self.motionStateCur == MotionState.Stand:
            tau_upp_fe = self.tau_upp_stand_L
            tau_low_fe = self.tau_low_stand_L
        else:
            tau_upp_fe = self.tau_upp_walk_L
            tau_low_fe = self.tau_low_walk_L
            
        f_upp[:8] = [1e10, 1e10, 1e10, 1e10, self.f_z_upp, tau_upp_fe[0], tau_upp_fe[1], tau_upp_fe[2]]
        f_upp[8:16] = f_upp[:8]
        f_low[:8] = [0, 0, 0, 0, self.f_z_low, tau_low_fe[0], tau_low_fe[1], tau_low_fe[2]]
        f_low[8:16] = f_low[:8]
        
        if self.motionStateCur in [MotionState.Walk, MotionState.Walk2Stand]:
            if self.legStateCur == LegState.LSt:
                f_upp[12:16] = 0
                f_low[12:16] = 0
                f_low[8:12] = -1e-7
            elif self.legStateCur == LegState.RSt:
                f_upp[4:8] = 0
                f_low[4:8] = 0
                f_low[:4] = -1e-7
                
        eigen_qp_A2 = np.zeros((16, 18))
        eigen_qp_A2[:, 6:18] = W
        
        neqRes_low = f_low - W @ self.Fr_ff
        neqRes_upp = f_upp - W @ self.Fr_ff
        
        eigen_qp_A_final = np.zeros((self.QP_nc, self.QP_nv))
        eigen_qp_A_final[:6, :] = eigen_qp_A1
        eigen_qp_A_final[6:, :] = eigen_qp_A2
        
        eigen_qp_lbA = np.zeros(self.QP_nc)
        eigen_qp_ubA = np.zeros(self.QP_nc)
        eigen_qp_lbA[:6] = eqRes
        eigen_qp_lbA[6:] = neqRes_low
        eigen_qp_ubA[:6] = eqRes
        eigen_qp_ubA[6:] = neqRes_upp
        
        # Construct QP cost matrix
        self.qp_H = np.zeros((self.QP_nv, self.QP_nv))
        if self.motionStateCur == MotionState.Stand:
            self.qp_H[:6, :6] = self.Q2 * 2.0 * 1e7
            self.qp_H[6:18, 6:18] = self.Q1 * 2.0 * 1e1
            self.qp_H[9, 9] *= 100
            self.qp_H[10, 10] *= 100
            self.qp_H[15, 15] *= 100
            self.qp_H[16, 16] *= 100
        else:
            self.qp_H[:6, :6] = self.Q2 * 2.0 * 1e7
            self.qp_H[6:18, 6:18] = self.Q1 * 2.0 * 1e1
            
        # # Copy to QP solver format
        # self.copy_eigen_to_real_t(self.qp_H, eigen_qp_H)
        # self.copy_eigen_to_real_t(self.qp_A, eigen_qp_A_final)
        # self.copy_eigen_to_real_t(self.qp_lbA, eigen_qp_lbA)
        # self.copy_eigen_to_real_t(self.qp_ubA, eigen_qp_ubA)
        
        # # Solve QP
        # self.nWSR = 200
        # self.cpu_time = self.time_step
        # res = self.QP_prob.init(self.qp_H, self.qp_g, self.qp_A, None, None, 
        #                        self.qp_lbA, self.qp_ubA, self.nWSR, self.cpu_time, 
        #                        self.xOpt_iniGuess)
        
        # self.qpStatus = qpoases.getSimpleStatus(res)
        
        # # Get solution
        # xOpt = np.zeros(self.QP_nv)
        # self.QP_prob.getPrimalSolution(xOpt)
        
        # if res == qpoases.SUCCESSFUL_RETURN:
        #     self.eigen_xOpt = xOpt
        G = np.vstack([self.qp_A, -self.qp_A])
        h = np.hstack([self.qp_ubA, -self.qp_lbA])
        q = np.zeros(self.QP_nv)
        x = solve_qp(self.qp_H, q, G, h, None, None, solver="quadprog")
            
        self.eigen_ddq_Opt = self.ddq_final_kin.copy()
        self.eigen_ddq_Opt[:6] += x[:6]
        self.eigen_fr_Opt = self.Fr_ff + x[6:18]
        
        # Compute joint torques
        tauRes = self.dyn_M @ self.eigen_ddq_Opt + self.dyn_Non - self.Jfe.T @ self.eigen_fr_Opt
        self.tauJointRes = tauRes[6:]
        
        # self.last_nWSR = self.nWSR
        # self.last_cpu_time = self.cpu_time
        
    def compute_ddq(self, pin_kin_dyn):
        # Task definitions for walk and stand states
        # ... (implementation similar to C++ version)
        
        # Compute based on current motion state
        if self.motionStateCur in [MotionState.Walk, MotionState.Walk2Stand]:
            self.kin_tasks_walk.compute_all(
                self.des_delta_q, self.des_dq, self.des_ddq, 
                self.dyn_M, self.dyn_M_inv, self.dq
            )
            self.delta_q_final_kin = self.kin_tasks_walk.out_delta_q
            self.dq_final_kin = self.kin_tasks_walk.out_dq
            self.ddq_final_kin = self.kin_tasks_walk.out_ddq
        elif self.motionStateCur == MotionState.Stand:
            self.kin_tasks_stand.compute_all(
                self.des_delta_q, self.des_dq, self.des_ddq, 
                self.dyn_M, self.dyn_M_inv, self.dq
            )
            self.delta_q_final_kin = self.kin_tasks_stand.out_delta_q
            self.dq_final_kin = self.kin_tasks_stand.out_dq
            self.ddq_final_kin = self.kin_tasks_stand.out_ddq
        else:
            self.delta_q_final_kin = np.zeros(self.model_nv)
            self.dq_final_kin = np.zeros(self.model_nv)
            self.ddq_final_kin = np.zeros(self.model_nv)
            
    def copy_eigen_to_real_t(self, target, source):
        """Copy numpy array to qpOASES real_t array"""
        count = 0
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                target[count] = source[i, j] if not np.isinf(source[i, j]) else qpoases.INFTY
                count += 1
                
    def set_q_ini(self, q_ini_des, q_ini_cur):
        self.qIniDes = q_ini_des
        self.qIniCur = q_ini_cur