class BaseTask():
    def __init__(self, cfg):
        self.cfg = cfg

    def set_data_logger(self, data_logger):
        self.data_logger = data_logger

    def mujoco_init_state(self, model, data):
        # init state using mujoco
        # init mujoco state from cfg
        return data
    
    def mujoco_state_adoption(self, model, data):
        return data
  
    def mujoco_action_adoption(self, action, model, data):
        return data