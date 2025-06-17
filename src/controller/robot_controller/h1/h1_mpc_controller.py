import math
from src.controller.robot_controller.robot_numerical_controler import RobotNumerialContoller
from src.controller.robot_controller.utils.foot_placement import FootPlacement
from src.controller.robot_controller.utils.gait_scheduler import GaitScheduler
from controller.robot_controller.utils.kin_dyn_solver import KinDynSolver
from src.controller.robot_controller.utils.motor_controller import MotorController
from src.controller.robot_controller.utils.mpc_controller import MPCController
from src.controller.robot_controller.utils.priority_tasks import PriorityTasks
from src.controller.robot_controller.utils.state_estimator import StateEstimator
from src.controller.robot_controller.utils.wbc_priority import WBCPriority
from src.utils.math import *

class H1MPCContoller(RobotNumerialContoller):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_components(self, state, sim_model, sim_data):
        self.kin_dyn_solver = KinDynSolver(self.cfg)
        self.model_nv = self.kin_dyn_solver.model_nv
        self.wbc_priority = WBCPriority(self.model_nv, 18, 22, 0.7, sim_model.opt.timestep)
        self.gait_scheduler = GaitScheduler(0.4, sim_model.opt.timestep)
        self.foot_placement = FootPlacement(self.cfg)
        self.motor_controller = MotorController(sim_model.opt.timestep)
        self.state_estimator = StateEstimator(sim_model.opt.timestep)

        pos_l_foot_end = np.array([-0.018, 0.113, -self.cfg['robot_size']['leg_length']])
        pos_r_foot_end = np.array([-0.018, -0.116, -self.cfg['robot_size']['leg_length']])

        euler_l_foot_end = np.array([-0.000, -0.008, -0.000])
        euler_r_foot_end = np.array([-0.000, -0.008, -0.000])
        R_l_leg = eul2rot(euler_l_foot_end[0], euler_l_foot_end[1], euler_l_foot_end[2])
        R_r_leg = eul2rot(euler_r_foot_end[0], euler_r_foot_end[1], euler_r_foot_end[2])
        self.stand_leg_state = self.kin_dyn_solver.computeInK_Leg(R_l_leg, pos_l_foot_end, R_r_leg, pos_r_foot_end)
        q_init = np.zeros(sim_model.nq)

        q_init[7:sim_model.nq-7] = self.stand_leg_state.jointPosRes
        self.hand_l_pos = np.array([0.475, -1.12, 1.9, 0.86, -0.356, 0, 0])
        self.hand_r_pos = np.array([-0.475, -1.12, -1.9, 0.86, 0.356, 0, 0])
        q_init[7:14] = self.hand_l_pos
        q_init[14:21] = self.hand_r_pos
        self.wbc_priority.set_q_ini(q_init, state.q)

        self.state_estimator.init(state)

    def generate_action(self, state):
        self.state_estimator.set(state)
        self.state_estimator.update()
        self.state_estimator.get(state)

        self.kin_dyn_solver.update(state)
        self.kin_dyn_solver.compute_Jacobians()
        self.kin_dyn_solver.compute_dynamic_matrix()
        self.kin_dyn_solver.set_state(state)

        self.state_estimator.setF(state)
        self.state_estimator.updateF()
        self.state_estimator.getF(state)

        state.ddq_des = np.zeros(self.model_nv)
        state.dq_des = np.zeros(self.model_nv)
        state.delta_q_des = np.zeros(self.model_nv)
        state.Fr_ff = np.array([0,0,370,0,0,0,0,0,370,0,0,0])

        self.wbc_priority.compute_ddq(self.kin_dyn_solver)
        self.wbc_priority.compute_tau()

        stand_state = self.stand_leg_state.copy()
        stand_state[0:7] = self.hand_l_pos
        stand_state[7:14] = self.hand_r_pos
        state.momotor_pos_des = stand_state
        state.motor_vel_des = np.zeros(self.model_nv - 6)
        state.motor_torque_des = np.zeros(self.model_nv - 6)

        self.motor_controller.get_state(state)
        self.motor_controller.calMotorsPVT(100.0/1000.0/180.0/math.pi)
        self.motor_controller.set_state(state)

        return state.motor_torque_out