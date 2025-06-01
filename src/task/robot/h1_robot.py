from src.task.robot.legged_robot import LeggedRobot
 
class H1Robot(LeggedRobot):
    def __init__(self, cfg):
        super().__init__(cfg)

    def mujoco_init_state(self, data):
        return data
    
    def mujoco_action_adoption(self, action, model, data):
        return data