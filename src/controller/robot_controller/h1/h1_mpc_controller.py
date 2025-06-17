import os
from src.controller.robot_controller.robot_numerical_controler import RobotNumerialContoller
from src.controller.robot_controller.utils.foot_placement import FootPlacement
from src.controller.robot_controller.utils.gait_scheduler import GaitScheduler
from src.controller.robot_controller.utils.joystick_interpreter import JoyStickInterpreter
from src.controller.robot_controller.utils.kin_dyn_solver import Pin_KinDyn
from src.controller.robot_controller.utils.motor_controller import PVT_Ctr
# from src.controller.robot_controller.utils.mpc_controller import MPC
from src.controller.robot_controller.utils.state_estimator import StateEst
from src.controller.robot_controller.utils.wbc_priority import WBCPriority
import numpy as np
from src.utils.math import *
from src.utils.env import *
from src.task.robot.data_bus import *
class H1MPCContoller(RobotNumerialContoller):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_components(self, model, data, state):
        self.dt = model.opt.timestep
        self.kinDynSolver = Pin_KinDyn(self.cfg)
        self.model_nv = self.kinDynSolver.model_nv
        self.WBC_solv = WBCPriority(self.model_nv, 18, 22, 0.7, self.dt)
        self.gaitScheduler = GaitScheduler(0.4, self.dt)
        cfg_path = os.path.join(ROBOT_CONTROL_ROOT_DIR, self.cfg['motor_control_cfg_path'])
        self.pvtCtr = PVT_Ctr(self.dt, cfg_path)
        self.footPlacement = FootPlacement()
        self.jsInterp = JoyStickInterpreter(self.dt)
        self.StateModule = StateEst(self.dt)

        # Initialize variables
        self.stand_legLength = 1.01
        self.foot_height = 0.07
        self.xv_des = 0.7
        
        self.footPlacement.kp_vx = 0.03
        self.footPlacement.kp_vy = 0.035
        self.footPlacement.kp_wz = 0.03
        self.footPlacement.stepHeight = 0.12
        self.footPlacement.legLength = self.stand_legLength
        
        # Initialize desired positions
        self.motors_pos_des = [0] * (self.kinDynSolver.model_nv - 6)
        self.motors_pos_cur = [0] * (self.kinDynSolver.model_nv - 6)
        self.motors_vel_des = [0] * (self.kinDynSolver.model_nv - 6)
        self.motors_vel_cur = [0] * (self.kinDynSolver.model_nv - 6)
        self.motors_tau_des = [0] * (self.kinDynSolver.model_nv - 6)
        self.motors_tau_cur = [0] * (self.kinDynSolver.model_nv - 6)
        
        fe_l_pos_L_des = np.array([-0.018, 0.113, -self.stand_legLength])
        fe_r_pos_L_des = np.array([-0.018, -0.116, -self.stand_legLength])
        fe_l_eul_L_des = np.array([-0.000, -0.008, -0.000])
        fe_r_eul_L_des = np.array([0.000, -0.008, 0.000])
        fe_l_rot_des = eul2Rot(fe_l_eul_L_des[0], fe_l_eul_L_des[1], fe_l_eul_L_des[2])
        fe_r_rot_des = eul2Rot(fe_r_eul_L_des[0], fe_r_eul_L_des[1], fe_r_eul_L_des[2])
        
        self.hd_l_des = np.array([0.475, -1.12, 1.9, 0.86, -0.356, 0, 0])
        self.hd_r_des = np.array([-0.475, -1.12, -1.9, 0.86, 0.356, 0, 0])
        
        # Compute initial joint positions
        self.resLeg = self.kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des)
        qIniDes = np.zeros(model.nq)
        qIniDes[7:model.nq] = self.resLeg.jointPosRes
        qIniDes[7:14] = self.hd_l_des
        qIniDes[14:21] = self.hd_r_des
        self.WBC_solv.set_q_ini(qIniDes, state.q)

        
        # Simulation loop
        self.startSteppingTime = 3
        self.startWalkingTime = 5
        self.load_debug_info(state)
        
    def generate_action(self, state, sim_time):
        simTime = sim_time
        
        if simTime > 1 and self.StateModule.get_init():
            print("init state module")
            self.StateModule.init(state)
        
        self.StateModule.set(state)
        self.StateModule.update()
        self.StateModule.get(state)
        
        # Update kinematics and dynamics
        self.kinDynSolver.dataBusRead(state)
        self.kinDynSolver.computeJ_dJ()
        self.kinDynSolver.computeDyn()
        self.kinDynSolver.dataBusWrite(state)
        
        self.StateModule.setF(state)
        self.StateModule.updateF()
        self.StateModule.getF(state)
        
        # Motion control
        if simTime > self.startWalkingTime:
            self.jsInterp.setWzDesLPara(0, 1)
            self.jsInterp.setVxDesLPara(self.xv_des, 2.0)
            state.motionState = self.MotionState.Walk
        else:
            self.jsInterp.set_ini_pos(state.q[0], state.q[1], state.base_rpy[2])
        
        if simTime >= self.startSteppingTime:
            self.jsInterp.step()
            self.jsInterp.set_ini_pos(state.q[0], state.q[1], self.stand_legLength + self.foot_height, state.base_rpy[2])
            self.jsInterp.dataBusWrite(state)
            
            self.gaitScheduler.start()
            state.motionState = MotionState.Walk
            self.gaitScheduler.dataBusRead(state)
            self.gaitScheduler.step()
            self.gaitScheduler.dataBusWrite(state)
            
            self.footPlacement.dataBusRead(state)
            self.footPlacement.getSwingPos()
            self.footPlacement.dataBusWrite(state)
        
        # WBC
        state.des_ddq = np.zeros(self.model_nv)
        state.des_dq = np.zeros(self.model_nv)
        state.des_delta_q = np.zeros(self.model_nv)
        state.Fr_ff = np.array([
            0, 0, 370, 0, 0, 0,
            0, 0, 370, 0, 0, 0
        ])
        
        if simTime > self.startWalkingTime + 1:
            state.des_delta_q[0:2] = np.array([self.jsInterp.vx_W *self.dt, 
                                                    self.jsInterp.vy_W * self.dt])
            state.des_delta_q[5] = self.jsInterp.wz_L * self.dt
            state.des_dq[0:2] = np.array([self.jsInterp.vx_W, self.jsInterp.vy_W])
            state.des_dq[5] = self.jsInterp.wz_L
            
            k = 5
            state.des_ddq[0:2] = k * (self.jsInterp.vx_W - state.dq[0]), k * (self.jsInterp.vy_W - state.dq[1])
            state.des_ddq[5] = k * (self.jsInterp.wz_L - state.dq[5])
        
        # WBC calculation
        self.WBC_solv.data_bus_read(state)
        self.WBC_solv.compute_ddq(self.kinDynSolver)
        self.WBC_solv.compute_tau()
        self.WBC_solv.data_bus_write(state)
        
        # Get final joint commands
        if simTime <= self.startSteppingTime:
            temp = self.resLeg.jointPosRes.copy()
            temp[0:7] = self.hd_l_des
            temp[7:14] = self.hd_r_des
            state.motors_pos_des = temp
            state.motors_vel_des = self.motors_vel_des
            state.motors_tor_des = self.motors_tau_des
        else:
            pos_des = self.kinDynSolver.integrateDIY(state.q, state.wbc_delta_q_final)
            state.motors_pos_des = pos_des[7:7+self.kinDynSolver.model_nv-6]
            state.motors_vel_des = state.wbc_dq_final
            state.motors_tor_des = state.wbc_tauJointRes
        
        # PVT control
        self.pvtCtr.data_bus_read(state)
        if simTime <= 3:
            self.pvtCtr.cal_motors_pvt_with_limit(100.0/1000.0/180.0*np.pi)
        else:
            kp = 1.0
            kd = 1.0
            
            # Set PD gains for left leg
            self.pvtCtr.setJointPD(400 * kp, 15 * kd, "J_hip_l_roll")
            self.pvtCtr.setJointPD(200 * kp, 10 * kd, "J_hip_l_yaw")
            self.pvtCtr.setJointPD(300 * kp, 10 * kd, "J_hip_l_pitch")
            self.pvtCtr.setJointPD(300 * kp, 14 * kd, "J_knee_l_pitch")
            self.pvtCtr.setJointPD(300 * kp, 18 * kd, "J_ankle_l_pitch")
            self.pvtCtr.setJointPD(300 * kp, 16 * kd, "J_ankle_l_roll")
            
            # Set PD gains for right leg
            self.pvtCtr.setJointPD(400 * kp, 15 * kd, "J_hip_r_roll")
            self.pvtCtr.setJointPD(200 * kp, 10 * kd, "J_hip_r_yaw")
            self.pvtCtr.setJointPD(300 * kp, 10 * kd, "J_hip_r_pitch")
            self.pvtCtr.setJointPD(300 * kp, 14 * kd, "J_knee_r_pitch")
            self.pvtCtr.setJointPD(300 * kp, 18 * kd, "J_ankle_r_pitch")
            self.pvtCtr.setJointPD(300 * kp, 16 * kd, "J_ankle_r_roll")
            
            self.pvtCtr.cal_motors_pvt()
        
        self.pvtCtr.data_bus_write(state)
        self.load_debug_info(state)
        return state.motors_tor_out
    
    def load_debug_info(self, state):
        None