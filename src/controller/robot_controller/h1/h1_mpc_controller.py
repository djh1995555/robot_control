from src.controller.robot_controller.robot_numerical_controler import RobotNumerialContoller


class H1MPCContoller(RobotNumerialContoller):
    def __init__(self, cfg):
        super().__init__(cfg)

    def generate_action(self, data):
        return 0.0