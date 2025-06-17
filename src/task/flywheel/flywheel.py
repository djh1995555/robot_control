from src.task.base_task import BaseTask
import mujoco
class Flywheel(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, mj_data: mujoco.MjData):
        mj_data.qpos[0] = self.cfg['model_cfg']['init_state']['arm_joint_pos']
        mj_data.qpos[1] = self.cfg['model_cfg']['init_state']['wheel_joint_ange']
        mj_data.qvel[0] = self.cfg['model_cfg']['init_state']['arm_joint_v']
        mj_data.qvel[1] = self.cfg['model_cfg']['init_state']['wheel_joint_v']
        return mj_data
    
    def mujoco_action_adoption(self, action, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        mj_data.actuator('arm_torque').ctrl[0] = action
        return mj_data