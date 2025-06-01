from src.task.base_task import BaseTask
import numpy as np
import math
from scipy.spatial.transform import Rotation
 
class WheelRobot(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, data):
        # face a random direction
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        # rotate and pitch slightly
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        data.qpos[3:7] = rotation.as_quat()

        data.actuator('motor_l_wheel').ctrl = [0]
        data.actuator('motor_r_wheel').ctrl = [0]
        return data
    
    def mujoco_action_adoption(self, action, model, data):
        data.actuator('motor_l_wheel').ctrl[0] = action[0]
        data.actuator('motor_r_wheel').ctrl[0] = action[1]
        return data