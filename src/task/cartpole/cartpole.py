from src.task.base_task import BaseTask
 
class Cartpole(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, data):
        data.qpos[0] = self.cfg['model_cfg']['init_state']['cart_pos']
        data.qpos[5] = self.cfg['model_cfg']['init_state']['hinge_ange']
        data.qvel[0] = self.cfg['model_cfg']['init_state']['cart_v']
        data.qvel[5] = self.cfg['model_cfg']['init_state']['hinge_v']
        return data
    
    def mujoco_action_adoption(self, action, model, data):
        data.actuator('cart_force').ctrl[0] = action
        return data