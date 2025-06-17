from src.task.base_task import BaseTask
import mujoco
class Cartpole(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, mj_data: mujoco.MjData):
        mj_data.qpos[0] = self.cfg['model_cfg']['init_state']['cart_pos']
        mj_data.qpos[5] = self.cfg['model_cfg']['init_state']['hinge_ange']
        mj_data.qvel[0] = self.cfg['model_cfg']['init_state']['cart_v']
        mj_data.qvel[5] = self.cfg['model_cfg']['init_state']['hinge_v']
        return mj_data
    
    def mujoco_action_adoption(self, action, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        mj_data.actuator('cart_force').ctrl[0] = action
        return mj_data