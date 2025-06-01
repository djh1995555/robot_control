from src.controller.base_controller.base_numerical_controler import BaseNumericalContoller
import numpy as np

class FlywheelLQRContoller(BaseNumericalContoller):
    def __init__(self, cfg):
        super().__init__(cfg)

    def generate_action(self, data):
        # 状态变量获取
        x1 = data.joint('arm_joint').qpos[0]  # 摆杆旋转角度
        x2 = data.joint('arm_joint').qvel[0]  # 摆杆旋转速度
        x3 = data.joint('wheel_joint').qpos[0]  # 飞轮旋转角度
        x4 = data.joint('wheel_joint').qvel[0]  # 飞轮旋转速度
        x = np.array([x1, x2, x3, x4])

        # K = np.array([-318.83592063, -93.21717676, -1.0, -1.58843839])
        K = np.array([-2.46709998e+02, -7.21295944e+01, 0.0, -1.00000000e+00])
        return -K.dot(x)