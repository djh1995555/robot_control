import numpy as np
from scipy.linalg import block_diag
from src.controller.robot_controller.utils.priority_tasks import PriorityTasks
from qpsolvers import available_solvers, print_matrix_vector, solve_qp
class WBCPriority():
    def __init__(self, model_nv_In, QP_nvIn, QP_ncIn, miu_In, dt):
        self.timeStep = dt
        self.model_nv = model_nv_In
        self.miu = miu_In
        self.QP_nc = QP_ncIn
        self.QP_nv = QP_nvIn
        
        # Initialize matrices
        self.Sf = np.zeros((6, self.model_nv))
        self.Sf[:6, :6] = np.eye(6)
        
        self.St_qpV2 = np.zeros((self.model_nv, self.model_nv - 6))
        self.St_qpV2[6:, :] = np.eye(self.model_nv - 6)
        
        self.St_qpV1 = np.zeros((self.model_nv, 6))
        self.St_qpV1[:6, :6] = np.eye(6)
        
        # Force limits
        self.f_z_low = 10
        self.f_z_upp = 1400
        
        # Torque limits
        self.tau_upp_stand_L = np.array([15, 30, 40])
        self.tau_low_stand_L = -self.tau_upp_stand_L
        
        self.tau_upp_walk_L = np.array([15, 40, 40])
        self.tau_low_walk_L = -self.tau_upp_walk_L
        
        # Initialize QP problem
        # self.QP_prob = qpoases.QProblem(self.QP_nv, self.QP_nc)
        # options = qpoases.Options()
        # options.setToMPC()
        # options.printLevel = qpoases.PL_LOW
        # self.QP_prob.setOptions(options)
        
        # Initialize vectors
        self.eigen_xOpt = np.zeros(self.QP_nv)
        self.eigen_ddq_Opt = np.zeros(self.model_nv)
        self.eigen_fr_Opt = np.zeros(12)
        self.eigen_tau_Opt = np.zeros(self.model_nv - 6)
        
        self.delta_q_final_kin = np.zeros(self.model_nv)
        self.dq_final_kin = np.zeros(self.model_nv)
        self.ddq_final_kin = np.zeros(self.model_nv)
        
        self.base_rpy_cur = np.zeros(3)
        
        # Initialize task priorities for different motion states
        self._initialize_task_priorities()
        
    def _initialize_task_priorities(self):
        """Initialize task priorities for walk and stand states"""
        # Walk tasks
        self.kin_tasks_walk = PriorityTasks()
        self.kin_tasks_walk.add_task("static_Contact")
        self.kin_tasks_walk.add_task("Roll_Pitch_Yaw_Pz")
        self.kin_tasks_walk.add_task("RedundantJoints")
        self.kin_tasks_walk.add_task("PxPy")
        self.kin_tasks_walk.add_task("SwingLeg")
        self.kin_tasks_walk.add_task("HandTrackJoints")
        self.kin_tasks_walk.add_task("PosRot")
        
        task_order_walk = [
            "static_Contact",
            "PosRot",
            "SwingLeg",
            "RedundantJoints",
            "HandTrackJoints"
        ]
        self.kin_tasks_walk.build_priority(task_order_walk)
        
        # Stand tasks
        self.kin_tasks_stand = PriorityTasks()
        self.kin_tasks_stand.add_task("static_Contact")
        self.kin_tasks_stand.add_task("CoMTrack")
        self.kin_tasks_stand.add_task("HandTrackJoints")
        self.kin_tasks_stand.add_task("HipRPY")
        self.kin_tasks_stand.add_task("HeadRP")
        self.kin_tasks_stand.add_task("Pz")
        self.kin_tasks_stand.add_task("CoMXY_HipRPY")
        self.kin_tasks_stand.add_task("Roll_Pitch_Yaw")
        self.kin_tasks_stand.add_task("fixedWaist")
        
        task_order_stand = [
            "static_Contact",
            "CoMXY_HipRPY",
            "Pz",
            "HandTrackJoints",
            "HeadRP"
        ]
        self.kin_tasks_stand.build_priority(task_order_stand)
    
    def update_state(self, robot_state):
        """Read data from robot state bus"""
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
        
        if self.legStateCur == "LSt":
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

    
    def get_output(self, state):
        state.wbc_ddq_final = self.eigen_ddq_Opt
        state.wbc_tauJointRes = self.tauJointRes
        state.wbc_FrRes = self.eigen_fr_Opt
        state.qp_cpuTime = self.cpu_time
        state.qp_nWSR = self.nWSR
        state.qp_status = self.qpStatus
        
        state.wbc_delta_q_final = self.delta_q_final_kin
        state.wbc_dq_final = self.dq_final_kin
        state.wbc_ddq_final = self.ddq_final_kin
        
        state.qp_status = self.qpStatus
        state.qp_nWSR = self.nWSR
        state.qp_cpuTime = self.cpu_time
    
    def compute_tau(self):
        """Compute joint torques using QP optimization"""
        # Construct QP problem
        eigen_qp_A1 = np.zeros((6, self.QP_nv))
        eigen_qp_A1[:6, :6] = self.Sf @ self.dyn_M @ self.St_qpV1
        eigen_qp_A1[:6, 6:18] = -self.Sf @ self.Jfe.T
        
        eqRes = -self.Sf @ self.dyn_M @ self.ddq_final_kin - self.Sf @ self.dyn_Non + self.Sf @ self.Jfe.T @ self.Fr_ff
        
        if self.motionStateCur == "Stand":
            Rfe = self.fe_l_rot_cur_W
        else:
            Rfe = self.stance_fe_rot_cur_W
            
        Mw2b = block_diag(Rfe.T, Rfe.T, Rfe.T, Rfe.T)
        
        W = np.zeros((16, 12))
        W[0, 0] = 1
        W[0, 2] = np.sqrt(2)/2.0 * self.miu
        W[1, 0] = -1
        W[1, 2] = np.sqrt(2)/2.0 * self.miu
        W[2, 1] = 1
        W[2, 2] = np.sqrt(2)/2.0 * self.miu
        W[3, 1] = -1
        W[3, 2] = np.sqrt(2)/2.0 * self.miu
        W[4:8, 2:6] = np.eye(4)
        W[8:, 6:] = W[:8, :6]
        W = W @ Mw2b
        
        f_low = np.zeros(16)
        f_upp = np.zeros(16)
        
        if self.motionStateCur == "Stand":
            tau_upp_fe = self.tau_upp_stand_L
            tau_low_fe = self.tau_low_stand_L
        else:
            tau_upp_fe = self.tau_upp_walk_L
            tau_low_fe = self.tau_low_walk_L
            
        f_upp[:8] = [1e10, 1e10, 1e10, 1e10, self.f_z_upp, *tau_upp_fe]
        f_upp[8:] = f_upp[:8]
        f_low[:8] = [0, 0, 0, 0, self.f_z_low, *tau_low_fe]
        f_low[8:] = f_low[:8]
        
        if self.motionStateCur in ["Walk", "Walk2Stand"]:
            if self.legStateCur == "LSt":
                f_upp[12:16] = 0
                f_low[12:16] = 0
                f_low[8:12] = -1e-7
            elif self.legStateCur == "RSt":
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
        
        eigen_qp_H = np.zeros((self.QP_nv, self.QP_nv))
        Q2 = np.eye(6)
        Q1 = np.eye(12)
        
        if self.motionStateCur == "Stand":
            eigen_qp_H[:6, :6] = Q2 * 2.0 * 1e7
            eigen_qp_H[6:18, 6:18] = Q1 * 2.0 * 1e1
            eigen_qp_H[9, 9] *= 100
            eigen_qp_H[10, 10] *= 100
            eigen_qp_H[15, 15] *= 100
            eigen_qp_H[16, 16] *= 100
        else:
            eigen_qp_H[:6, :6] = Q2 * 2.0 * 1e7
            eigen_qp_H[6:18, 6:18] = Q1 * 2.0 * 1e1
            
        # Convert to qpOASES format
        # qp_H = self._eigen_to_real_t(eigen_qp_H)
        # qp_A = self._eigen_to_real_t(eigen_qp_A_final)
        # qp_lbA = self._eigen_to_real_t(eigen_qp_lbA)
        # qp_ubA = self._eigen_to_real_t(eigen_qp_ubA)
        # qp_g = np.zeros(self.QP_nv)
        # xOpt_iniGuess = np.zeros(self.QP_nv)
        
        # self.nWSR = 200
        # self.cpu_time = self.timeStep
        
        # res = self.QP_prob.init(qp_H, qp_g, qp_A, None, None, qp_lbA, qp_ubA, self.nWSR, self.cpu_time, xOpt_iniGuess)
        # self.qpStatus = qpOASES.getSimpleStatus(res)
        
        # xOpt = np.zeros(self.QP_nv)
        # self.QP_prob.getPrimalSolution(xOpt)
        
        # if res == qpOASES.SUCCESSFUL_RETURN:
        #     self.eigen_xOpt = xOpt

        P = eigen_qp_H
        q = np.zeros(self.QP_nv)
        G = np.vstack([eigen_qp_A_final, -eigen_qp_A_final])
        h = np.hstack([eigen_qp_ubA, -eigen_qp_lbA]) 
        self.eigen_xOpt = solve_qp(P, q, G, h, None, None, solver='quadprog')
        self.eigen_ddq_Opt = self.ddq_final_kin.copy()
        self.eigen_ddq_Opt[:6] += self.eigen_xOpt[:6]
        self.eigen_fr_Opt = self.Fr_ff + self.eigen_xOpt[6:18]
        
        if self.qpStatus != 0:
            pass  # Handle QP failure
        
        tauRes = self.dyn_M @ self.eigen_ddq_Opt + self.dyn_Non - self.Jfe.T @ self.eigen_fr_Opt
        self.tauJointRes = tauRes[6:]
        
        self.last_nWSR = self.nWSR
        self.last_cpu_time = self.cpu_time
    
    def compute_ddq(self, pin_kin_dyn):
        """Compute desired accelerations"""
        if self.motionStateCur in ["Walk", "Walk2Stand"]:
            self._setup_walk_tasks()
            self.kin_tasks_walk.compute_all(
                self.des_delta_q, self.des_dq, self.des_ddq, 
                self.dyn_M, self.dyn_M_inv, self.dq
            )
            self.delta_q_final_kin = self.kin_tasks_walk.out_delta_q
            self.dq_final_kin = self.kin_tasks_walk.out_dq
            self.ddq_final_kin = self.kin_tasks_walk.out_ddq
        elif self.motionStateCur == "Stand":
            self._setup_stand_tasks()
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
    
    def _setup_walk_tasks(self):
        """Configure tasks for walking state"""
        # Static contact task
        id = self.kin_tasks_walk.get_id("static_Contact")
        self.kin_tasks_walk.taskLib[id].errX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(6) * 0
        self.kin_tasks_walk.taskLib[id].kd = np.eye(6) * 0
        self.kin_tasks_walk.taskLib[id].J = self.Jc
        self.kin_tasks_walk.taskLib[id].dJ = self.dJc
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # Redundant joints task
        id = self.kin_tasks_walk.get_id("RedundantJoints")
        self.kin_tasks_walk.taskLib[id].errX = np.zeros(5)
        self.kin_tasks_walk.taskLib[id].errX[0] = 0 - self.q[21]
        self.kin_tasks_walk.taskLib[id].errX[1] = 0 - self.q[22]
        self.kin_tasks_walk.taskLib[id].errX[2] = 0 - self.q[23]
        self.kin_tasks_walk.taskLib[id].errX[3] = 0 - self.q[24]
        self.kin_tasks_walk.taskLib[id].errX[4] = 0 - self.q[25]
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(5)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(5)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(5)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(5) * 100
        self.kin_tasks_walk.taskLib[id].kd = np.eye(5) * 20
        self.kin_tasks_walk.taskLib[id].J = np.zeros((5, self.model_nv))
        self.kin_tasks_walk.taskLib[id].J[0, 20] = 1
        self.kin_tasks_walk.taskLib[id].J[1, 21] = 1
        self.kin_tasks_walk.taskLib[id].J[2, 22] = 1
        self.kin_tasks_walk.taskLib[id].J[3, 23] = 1
        self.kin_tasks_walk.taskLib[id].J[4, 24] = 1
        self.kin_tasks_walk.taskLib[id].dJ = np.zeros((5, self.model_nv))
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # Roll/Pitch/Yaw/Pz task
        id = self.kin_tasks_walk.get_id("Roll_Pitch_Yaw_Pz")
        self.kin_tasks_walk.taskLib[id].errX = np.zeros(4)
        desRot = self.eul2rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_walk.taskLib[id].errX[:3] = self.diff_rot(self.base_rot, desRot)
        self.kin_tasks_walk.taskLib[id].errX[3] = self.base_pos_des[2] - self.q[2]
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(4)
        self.kin_tasks_walk.taskLib[id].derrX[:3] = -self.dq[3:6]
        self.kin_tasks_walk.taskLib[id].derrX[3] = 0 - self.dq[2]
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(4)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(4)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(4) * 100
        self.kin_tasks_walk.taskLib[id].kd = np.eye(4) * 10
        taskMap = np.zeros((4, 6))
        taskMap[0, 3] = 1
        taskMap[1, 4] = 1
        taskMap[2, 5] = 1
        taskMap[3, 2] = 1
        self.kin_tasks_walk.taskLib[id].J = taskMap @ self.J_base
        self.kin_tasks_walk.taskLib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # PxPy task
        id = self.kin_tasks_walk.get_id("PxPy")
        self.kin_tasks_walk.taskLib[id].errX = self.des_dq[:2] * self.timeStep
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(2)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(2)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(2)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(2) * 100
        self.kin_tasks_walk.taskLib[id].kd = np.eye(2) * 50
        taskMap = np.zeros((2, 6))
        taskMap[0, 0] = 1
        taskMap[1, 1] = 1
        self.kin_tasks_walk.taskLib[id].J = taskMap @ self.J_base
        self.kin_tasks_walk.taskLib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # PosRot task
        id = self.kin_tasks_walk.get_id("PosRot")
        self.kin_tasks_walk.taskLib[id].errX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].errX[:3] = self.base_pos_des - self.q[:3]
        
        # Clamp position errors
        for i in range(2):
            if abs(self.kin_tasks_walk.taskLib[id].errX[i]) >= 0.02:
                self.kin_tasks_walk.taskLib[id].errX[i] = 0.02 * np.sign(self.kin_tasks_walk.taskLib[id].errX[i])
        
        if self.kin_tasks_walk.taskLib[id].errX[2] > 0.005:
            self.kin_tasks_walk.taskLib[id].errX[2] = 0.005
            
        desRot = self.eul2rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_walk.taskLib[id].errX[3:6] = self.diff_rot(self.base_rot, desRot)
        self.kin_tasks_walk.taskLib[id].errX[4] -= 0.05 * self.dq[4]
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(6) * 500
        self.kin_tasks_walk.taskLib[id].kp[0, 0] = 100
        self.kin_tasks_walk.taskLib[id].kp[4, 4] = 800
        self.kin_tasks_walk.taskLib[id].kd = np.eye(6) * 10
        self.kin_tasks_walk.taskLib[id].kd[4, 4] = 10
        self.kin_tasks_walk.taskLib[id].J = self.J_base
        self.kin_tasks_walk.taskLib[id].dJ = self.dJ_base
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # SwingLeg task
        id = self.kin_tasks_walk.get_id("SwingLeg")
        self.kin_tasks_walk.taskLib[id].errX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].errX[:3] = self.swing_fe_pos_des_W - self.fe_pos_sw_W
        desRot = self.eul2rot(*self.swing_fe_rpy_des_W)
        self.kin_tasks_walk.taskLib[id].errX[3:6] = self.diff_rot(self.fe_rot_sw_W, desRot)
        self.kin_tasks_walk.taskLib[id].errX[4] *= 2
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(6)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(6) * 500
        self.kin_tasks_walk.taskLib[id].kd = np.eye(6) * 20
        self.kin_tasks_walk.taskLib[id].J = self.Jsw
        self.kin_tasks_walk.taskLib[id].J[:, 22:25] = 0  # Exclude waist joints
        self.kin_tasks_walk.taskLib[id].dJ = self.dJsw
        self.kin_tasks_walk.taskLib[id].dJ[:, 22:25] = 0  # Exclude waist joints
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
        
        # HandTrackJoints task
        l_hip_pitch = self.q[28] - self.q[34]
        r_hip_pitch = self.q[34] - self.q[28]
        target_arm_q = np.array([
            0.475 - 0.75*r_hip_pitch, -1.12, 1.9, 0.86, -0.356, 0, 0,
            -0.475 + 0.75*l_hip_pitch, -1.12, -1.9, 0.86, 0.356, 0, 0
        ])
        
        id = self.kin_tasks_walk.get_id("HandTrackJoints")
        self.kin_tasks_walk.taskLib[id].errX = target_arm_q - self.q[7:21]
        self.kin_tasks_walk.taskLib[id].derrX = np.zeros(14)
        self.kin_tasks_walk.taskLib[id].ddxDes = np.zeros(14)
        self.kin_tasks_walk.taskLib[id].dxDes = np.zeros(14)
        self.kin_tasks_walk.taskLib[id].kp = np.eye(14) * 200
        self.kin_tasks_walk.taskLib[id].kd = np.eye(14) * 10
        self.kin_tasks_walk.taskLib[id].J = np.zeros((14, self.model_nv))
        self.kin_tasks_walk.taskLib[id].J[:14, 6:20] = np.eye(14)
        self.kin_tasks_walk.taskLib[id].dJ = np.zeros((14, self.model_nv))
        self.kin_tasks_walk.taskLib[id].W = np.eye(self.model_nv)
    
    def _setup_stand_tasks(self):
        """Configure tasks for standing state"""
        # Similar to _setup_walk_tasks but for standing configuration
        # Implementation would follow the same pattern as above
        
    def _eigen_to_real_t(self, eigen_matrix):
        """Convert Eigen matrix to qpOASES real_t array"""
        return eigen_matrix.flatten().astype(np.float64)
    
    def eul2rot(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        # Implementation depends on your convention
        pass
    
    def diff_rot(self, R1, R2):
        """Compute difference between two rotation matrices"""
        # Implementation depends on your needs
        pass
    
    def set_q_ini(self, q_ini_des, q_ini_cur):
        """Set initial joint positions"""
        self.qIniDes = q_ini_des
        self.qIniCur = q_ini_cur