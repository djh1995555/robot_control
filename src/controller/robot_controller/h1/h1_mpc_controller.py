from src.controller.robot_controller.robot_numerical_controler import RobotNumerialContoller
from src.controller.robot_controller.utils.foot_placement import FootPlacement
from src.controller.robot_controller.utils.gait_scheduler import GaitScheduler
from src.controller.robot_controller.utils.kin_dyn_solver import KinDynSolver
from src.controller.robot_controller.utils.motor_controller import MotorController
from src.controller.robot_controller.utils.mpc_controller import MPCController
from src.controller.robot_controller.utils.priority_tasks import PriorityTasks
from src.controller.robot_controller.utils.robot_model import RobotModel
from src.controller.robot_controller.utils.wbc_priority import WBCPriority


class H1MPCContoller(RobotNumerialContoller):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kin_dyn_solver = KinDynSolver(cfg)
        self.gait_scheduler = GaitScheduler(cfg)
        self.foot_placement = FootPlacement(cfg)
        self.mpc_controller = MPCController(cfg)
        self.wbc_priority = WBCPriority(cfg)
        self.motor_controller = MotorController(cfg)

    def generate_action(self, state):
        self.kin_dyn_solver.update(state)
        self.kin_dyn_solver.compute_Jacobians()
        return 0.0