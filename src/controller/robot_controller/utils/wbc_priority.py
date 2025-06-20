import numpy as np
from typing import List
from src.controller.robot_controller.utils.priority_tasks import PriorityTasks
# import qpoases
from qpsolvers import solve_qp
from src.task.robot.data_bus import *

class WBCPriority:
    def __init__(self, model_nv: int, qp_nv: int, qp_nc: int, miu: float, dt: float):
        self.model_nv = model_nv
        self.miu = miu
        self.time_step = dt
        self.QP_nv = qp_nv
        self.QP_nc = qp_nc
        
        # Initialize matrices and vectors
        self.Sf = np.zeros((6, self.model_nv))
        self.Sf[:6, :6] = np.eye(6)
        
        self.St_qpV2 = np.zeros((self.model_nv, self.model_nv - 6))
        self.St_qpV2[6:, :] = np.eye(self.model_nv - 6)
        
        self.St_qpV1 = np.zeros((self.model_nv, 6))
        self.St_qpV1[:6, :6] = np.eye(6)
        
        # Contact force limits
        self.f_z_low = 10
        self.f_z_upp = 1400
        
        # Torque limits
        self.tau_upp_stand_L = np.array([15, 30, 40])
        self.tau_low_stand_L = np.array([-15, -30, -40])
        self.tau_upp_walk_L = np.array([15, 40, 40])
        self.tau_low_walk_L = np.array([-15, -40, -40])

        self.pCoMDes = np.zeros(3)
        
        # QP problem setup
        # self.QP_prob = qpoases.QProblem(qp_nv, qp_nc)
        # options = qpoases.Options()
        # options.setToMPC()
        # options.printLevel = qpoases.PL_LOW
        # self.QP_prob.setOptions(options)
        
        # Initialize vectors
        self.eigen_xOpt = np.zeros(qp_nv)
        self.eigen_ddq_Opt = np.zeros(self.model_nv)
        self.eigen_fr_Opt = np.zeros(12)
        self.eigen_tau_Opt = np.zeros(self.model_nv - 6)
        
        self.delta_q_final_kin = np.zeros(self.model_nv)
        self.dq_final_kin = np.zeros(self.model_nv)
        self.ddq_final_kin = np.zeros(self.model_nv)
        
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
        self.legStateCur = LegState.LSt
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

    def task_definition(self):
        # -------- walk -------------
        # static_Contact task
        id = self.kin_tasks_walk.get_id("static_Contact")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(6) * 0
        self.kin_tasks_walk.task_lib[id].kd = np.eye(6) * 0
        self.kin_tasks_walk.task_lib[id].J = self.Jc
        self.kin_tasks_walk.task_lib[id].dJ = self.dJc
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # RedundantJoints task
        id = self.kin_tasks_walk.get_id("RedundantJoints")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(5)
        self.kin_tasks_walk.task_lib[id].err_x[0] = 0 - self.q[21]
        self.kin_tasks_walk.task_lib[id].err_x[1] = 0 - self.q[22]
        self.kin_tasks_walk.task_lib[id].err_x[2] = 0 - self.q[23]
        self.kin_tasks_walk.task_lib[id].err_x[3] = 0 - self.q[24]
        self.kin_tasks_walk.task_lib[id].err_x[4] = 0 - self.q[25]
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(5)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(5)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(5)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(5) * 100
        self.kin_tasks_walk.task_lib[id].kd = np.eye(5) * 20
        self.kin_tasks_walk.task_lib[id].J = np.zeros((5, self.model_nv))
        self.kin_tasks_walk.task_lib[id].J[0, 20] = 1
        self.kin_tasks_walk.task_lib[id].J[1, 21] = 1
        self.kin_tasks_walk.task_lib[id].J[2, 22] = 1
        self.kin_tasks_walk.task_lib[id].J[3, 23] = 1
        self.kin_tasks_walk.task_lib[id].J[4, 24] = 1
        self.kin_tasks_walk.task_lib[id].dJ = np.zeros((5, self.model_nv))
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # Roll_Pitch_Yaw_Pz task
        id = self.kin_tasks_walk.get_id("Roll_Pitch_Yaw_Pz")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(4)
        desRot = eul2Rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_walk.task_lib[id].err_x[:3] = diffRot(self.base_rot, desRot)
        self.kin_tasks_walk.task_lib[id].err_x[3] = self.base_pos_des[2] - self.q[2]
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(4)
        self.kin_tasks_walk.task_lib[id].derr_x[:3] = -self.dq[3:6]
        self.kin_tasks_walk.task_lib[id].derr_x[3] = 0 - self.dq[2]
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(4)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(4)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(4) * 100
        self.kin_tasks_walk.task_lib[id].kd = np.eye(4) * 10
        taskMap = np.zeros((4, 6))
        taskMap[0, 3] = 1
        taskMap[1, 4] = 1
        taskMap[2, 5] = 1
        taskMap[3, 2] = 1
        self.kin_tasks_walk.task_lib[id].J = taskMap @ self.J_base
        self.kin_tasks_walk.task_lib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # PxPy task
        id = self.kin_tasks_walk.get_id("PxPy")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(2)
        self.kin_tasks_walk.task_lib[id].err_x = self.des_dq[:2] * self.time_step
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(2)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(2)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(2)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(2) * 100
        self.kin_tasks_walk.task_lib[id].kd = np.eye(2) * 50
        taskMap = np.zeros((2, 6))
        taskMap[0, 0] = 1
        taskMap[1, 1] = 1
        self.kin_tasks_walk.task_lib[id].J = taskMap @ self.J_base
        self.kin_tasks_walk.task_lib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # PosRot task
        id = self.kin_tasks_walk.get_id("PosRot")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].err_x[:3] = self.base_pos_des - self.q[:3]
        if abs(self.kin_tasks_walk.task_lib[id].err_x[0]) >= 0.02:
            self.kin_tasks_walk.task_lib[id].err_x[0] = 0.02 * np.sign(self.kin_tasks_walk.task_lib[id].err_x[0])
        if abs(self.kin_tasks_walk.task_lib[id].err_x[1]) >= 0.02:
            self.kin_tasks_walk.task_lib[id].err_x[1] = 0.02 * np.sign(self.kin_tasks_walk.task_lib[id].err_x[1])
        if self.kin_tasks_walk.task_lib[id].err_x[2] > 0.005:
            self.kin_tasks_walk.task_lib[id].err_x[2] = 0.005
        desRot = eul2Rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_walk.task_lib[id].err_x[3:6] = diffRot(self.base_rot, desRot)
        self.kin_tasks_walk.task_lib[id].err_x[4] -= 0.05 * self.dq[4]
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(6) * 500
        self.kin_tasks_walk.task_lib[id].kp[3:, 3:] = np.eye(3) * 500
        self.kin_tasks_walk.task_lib[id].kp[0, 0] = 100
        self.kin_tasks_walk.task_lib[id].kp[4, 4] = 800
        self.kin_tasks_walk.task_lib[id].kd = np.eye(6) * 10
        self.kin_tasks_walk.task_lib[id].kd[4, 4] = 10
        self.kin_tasks_walk.task_lib[id].J = self.J_base
        self.kin_tasks_walk.task_lib[id].dJ = self.dJ_base
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # SwingLeg task
        id = self.kin_tasks_walk.get_id("SwingLeg")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].err_x[:3] = self.swing_fe_pos_des_W - self.fe_pos_sw_W
        desRot = eul2Rot(self.swing_fe_rpy_des_W[0], self.swing_fe_rpy_des_W[1], self.swing_fe_rpy_des_W[2])
        self.kin_tasks_walk.task_lib[id].err_x[3:6] = diffRot(self.fe_rot_sw_W, desRot)
        self.kin_tasks_walk.task_lib[id].err_x[4] *= 2
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(6)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(6) * 500
        self.kin_tasks_walk.task_lib[id].kd = np.eye(6) * 20
        self.kin_tasks_walk.task_lib[id].J = self.Jsw
        self.kin_tasks_walk.task_lib[id].J[:, 22:25] = 0  # exclude waist joints
        self.kin_tasks_walk.task_lib[id].dJ = self.dJsw
        self.kin_tasks_walk.task_lib[id].dJ[:, 22:25] = 0  # exclude waist joints
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)
        
        # HandTrackJoints task
        l_hip_pitch = self.q[28] - self.q[34]
        r_hip_pitch = self.q[34] - self.q[28]
        target_arm_q = np.array([
            0.475 - 0.75*r_hip_pitch, -1.12, 1.9, 0.86, -0.356, 0, 0,
            -0.475 + 0.75*l_hip_pitch, -1.12, -1.9, 0.86, 0.356, 0, 0
        ])
        
        id = self.kin_tasks_walk.get_id("HandTrackJoints")
        self.kin_tasks_walk.task_lib[id].err_x = np.zeros(14)
        self.kin_tasks_walk.task_lib[id].err_x = target_arm_q - self.q[7:21]
        self.kin_tasks_walk.task_lib[id].derr_x = np.zeros(14)
        self.kin_tasks_walk.task_lib[id].ddx_des = np.zeros(14)
        self.kin_tasks_walk.task_lib[id].dx_des = np.zeros(14)
        self.kin_tasks_walk.task_lib[id].kp = np.eye(14) * 200
        self.kin_tasks_walk.task_lib[id].kd = np.eye(14) * 10
        self.kin_tasks_walk.task_lib[id].J = np.zeros((14, self.model_nv))
        self.kin_tasks_walk.task_lib[id].J[:14, 6:20] = np.eye(14)
        self.kin_tasks_walk.task_lib[id].dJ = np.zeros((14, self.model_nv))
        self.kin_tasks_walk.task_lib[id].W = np.eye(self.model_nv)

        # -------- stand -------------
        # static_Contact task
        id = self.kin_tasks_stand.get_id("static_Contact")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(12)
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(12)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(12)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(12)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(12) * 0
        self.kin_tasks_stand.task_lib[id].kd = np.eye(12) * 0
        self.kin_tasks_stand.task_lib[id].J = np.zeros((12, self.model_nv))
        taskCtMap = np.zeros((3, 3))
        taskCtMap[1, 1] = 1
        taskCtMap[2, 2] = 1
        taskCtMap = self.fe_l_rot_cur_W @ taskCtMap @ self.fe_l_rot_cur_W.T  # disable ankle roll joint
        self.kin_tasks_stand.task_lib[id].J = self.Jfe
        self.kin_tasks_stand.task_lib[id].J[3:6] = taskCtMap @ self.kin_tasks_stand.task_lib[id].J[3:6]
        self.kin_tasks_stand.task_lib[id].J[9:12] = taskCtMap @ self.kin_tasks_stand.task_lib[id].J[9:12]
        self.kin_tasks_stand.task_lib[id].J[:, 22:25] = 0  # exclude waist joints
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((12, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # HipRPY task
        id = self.kin_tasks_stand.get_id("HipRPY")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(3)
        desRot = eul2Rot(0, 0, 0)
        self.kin_tasks_stand.task_lib[id].err_x[:3] = diffRot(self.hip_link_rot, desRot)
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(3) * 1000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(3) * 50
        taskMapRPY = np.zeros((3, 6))
        taskMapRPY[0, 3] = 1
        taskMapRPY[1, 4] = 1
        taskMapRPY[2, 5] = 1
        self.kin_tasks_stand.task_lib[id].J = taskMapRPY @ self.J_hip_link
        self.kin_tasks_stand.task_lib[id].J[:, 22:25] = 0
        self.kin_tasks_stand.task_lib[id].J[:, 6:20] = 0
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((3, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # Pz task
        id = self.kin_tasks_stand.get_id("Pz")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(1)
        self.kin_tasks_stand.task_lib[id].err_x[0] = self.base_pos_des[2] - self.q[2]
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(1)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(1)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(1)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(1) * 2000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(1) * 10
        taskMap = np.zeros((1, 6))
        taskMap[0, 2] = 1
        self.kin_tasks_stand.task_lib[id].J = taskMap @ self.J_base
        self.kin_tasks_stand.task_lib[id].J[:, 22:25] = 0
        self.kin_tasks_stand.task_lib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_stand.task_lib[id].dJ[:, 22:25] = 0
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # CoMTrack task
        id = self.kin_tasks_stand.get_id("CoMTrack")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].err_x = self.pCoMDes[:2] - self.pCoMCur[:2]
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(2) * 2000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(2) * 100
        self.kin_tasks_stand.task_lib[id].J = self.Jcom[:2]
        self.kin_tasks_stand.task_lib[id].J[:, 6:20] = 0
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((2, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # CoMXY_HipRPY task
        id = self.kin_tasks_stand.get_id("CoMXY_HipRPY")
        taskMapRPY = np.zeros((3, 6))
        taskMapRPY[0, 3] = 1
        taskMapRPY[1, 4] = 1
        taskMapRPY[2, 5] = 1
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(5)
        self.kin_tasks_stand.task_lib[id].err_x[:2] = self.pCoMDes[:2] - self.pCoMCur[:2]
        desRot = eul2Rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_stand.task_lib[id].err_x[2:5] = diffRot(self.hip_link_rot, desRot)
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(5)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(5)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(5)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(5) * 250
        self.kin_tasks_stand.task_lib[id].kp[2:5, 2:5] = np.eye(3) * 1000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(5) * 10
        self.kin_tasks_stand.task_lib[id].kd[2:5, 2:5] = np.eye(3) * 10
        self.kin_tasks_stand.task_lib[id].J = np.zeros((5, self.model_nv))
        self.kin_tasks_stand.task_lib[id].J[:2] = self.Jcom[:2]
        self.kin_tasks_stand.task_lib[id].J[2:5] = taskMapRPY @ self.J_hip_link
        self.kin_tasks_stand.task_lib[id].J[2:5, 22:25] = 0  # exclude waist joints
        self.kin_tasks_stand.task_lib[id].J[2:5, 6:20] = 0  # exclude arm joints
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((5, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        self.kin_tasks_stand.task_lib[id].W[22] = 200
        self.kin_tasks_stand.task_lib[id].W[23] = 200
        
        # HandTrackJoints task
        target_arm_q = np.array([
            0.475, -1.12, 1.9, 0.86, -0.356, 0, 0,
            -0.475, -1.12, -1.9, 0.86, 0.356, 0, 0
        ])
        
        id = self.kin_tasks_stand.get_id("HandTrackJoints")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(14)
        self.kin_tasks_stand.task_lib[id].err_x = target_arm_q - self.q[7:21]
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(14)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(14)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(14)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(14) * 2000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(14) * 100
        self.kin_tasks_stand.task_lib[id].J = np.zeros((14, self.model_nv))
        self.kin_tasks_stand.task_lib[id].J[:14, 6:20] = np.eye(14)
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((14, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # HeadRP task
        id = self.kin_tasks_stand.get_id("HeadRP")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].err_x[0] = 0 - self.q[21]
        self.kin_tasks_stand.task_lib[id].err_x[1] = self.base_rpy_cur[1] - self.q[22]
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(2)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(2) * 100
        self.kin_tasks_stand.task_lib[id].kd = np.eye(2) * 10
        self.kin_tasks_stand.task_lib[id].J = np.zeros((2, self.model_nv))
        self.kin_tasks_stand.task_lib[id].J[0, 20] = 1
        self.kin_tasks_stand.task_lib[id].J[1, 21] = 1
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((2, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # Roll_Pitch_Yaw task
        id = self.kin_tasks_stand.get_id("Roll_Pitch_Yaw")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(3)
        desRot = eul2Rot(self.base_rpy_des[0], self.base_rpy_des[1], self.base_rpy_des[2])
        self.kin_tasks_stand.task_lib[id].err_x = diffRot(self.base_rot, desRot)
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].derr_x = -self.dq[3:6]
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(3) * 2000
        self.kin_tasks_stand.task_lib[id].kd = np.eye(3) * 100
        taskMap = np.zeros((3, 6))
        taskMap[0, 3] = 1
        taskMap[1, 4] = 1
        taskMap[2, 5] = 1
        self.kin_tasks_stand.task_lib[id].J = taskMap @ self.J_base
        self.kin_tasks_stand.task_lib[id].dJ = taskMap @ self.dJ_base
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)
        
        # fixedWaist task
        id = self.kin_tasks_stand.get_id("fixedWaist")
        self.kin_tasks_stand.task_lib[id].err_x = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].err_x[0] = 0 - self.q[23]
        self.kin_tasks_stand.task_lib[id].err_x[1] = 0 - self.q[24]
        self.kin_tasks_stand.task_lib[id].err_x[2] = 0 - self.q[25]
        self.kin_tasks_stand.task_lib[id].derr_x = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].ddx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].dx_des = np.zeros(3)
        self.kin_tasks_stand.task_lib[id].kp = np.eye(3) * 200
        self.kin_tasks_stand.task_lib[id].kd = np.eye(3) * 20
        self.kin_tasks_stand.task_lib[id].J = np.zeros((3, self.model_nv))
        self.kin_tasks_stand.task_lib[id].J[0, 22] = 1
        self.kin_tasks_stand.task_lib[id].J[1, 23] = 1
        self.kin_tasks_stand.task_lib[id].J[2, 24] = 1
        self.kin_tasks_stand.task_lib[id].dJ = np.zeros((3, self.model_nv))
        self.kin_tasks_stand.task_lib[id].W = np.eye(self.model_nv)

    def compute_ddq(self, pin_kin_dyn):
        # Task definitions for walk and stand states
        # ... (implementation similar to C++ version)
        self.task_definition()

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