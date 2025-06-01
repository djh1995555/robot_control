class BaseTask():
    def __init__(self, cfg):
        self.cfg = cfg

    def mujoco_init_state(self):
        None
  
    def mujoco_action_adoption(self, action, model, data):
        None