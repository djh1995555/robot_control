from src.task.base_task import BaseTask
 
class Flywheel(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, data):
        data.qpos[0] = self.cfg['model_cfg']['init_state']['arm_joint_pos']
        data.qpos[1] = self.cfg['model_cfg']['init_state']['wheel_joint_ange']
        data.qvel[0] = self.cfg['model_cfg']['init_state']['arm_joint_v']
        data.qvel[1] = self.cfg['model_cfg']['init_state']['wheel_joint_v']
        return data
    
    def mujoco_action_adoption(self, action, model, data):
        data.actuator('arm_torque').ctrl[0] = action
        return data