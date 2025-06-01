from src.controller.base_controller.base_numerical_controler import BaseNumericalContoller


class RobotNumerialContoller(BaseNumericalContoller):
    def __init__(self, cfg):
        super().__init__(cfg)