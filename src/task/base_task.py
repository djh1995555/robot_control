
import mujoco
class BaseTask():
    def __init__(self, cfg):
        self.cfg = cfg

    def set_data_logger(self, data_logger):
        self.data_logger = data_logger

    def init_state(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        None
        
    def mujoco_state_adoption(self, mj_data: mujoco.MjData):
        return mj_data
            
    def init_mujoco_state(self, mj_data: mujoco.MjData):
        return mj_data
  
    def mujoco_action_adoption(self, action, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        return mj_data