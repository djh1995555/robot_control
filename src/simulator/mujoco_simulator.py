import time
from src.simulator.base_simulator import BaseSimulator
import mujoco
import mujoco.viewer

class MujocoSimulator(BaseSimulator):
    def __init__(self, task, controller, cfg):
        super().__init__(task, controller, cfg)
        self.cfg = cfg['mujoco_cfg']
        self.sim_model = mujoco.MjModel.from_xml_path(self.cfg['xml_path'])
        self.sim_data = mujoco.MjData(self.sim_model)

        self.reset()
        
        self.state = self.task.init_state(self.sim_model, self.sim_data)
        self.controller.init_components(self.sim_model, self.sim_data, self.state)

        self.sim_data = self.task.init_mujoco_state(self.sim_data)
        mujoco.mj_forward(self.sim_model, self.sim_data)

    def reset(self):
        self.controller.reset(self.sim_data)
    
    def run_simulation(self):
        # Close the viewer automatically after 30 wall-seconds.
        with mujoco.viewer.launch_passive(self.sim_model, self.sim_data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < self.cfg['sim_duration']:
                self.data_logger.add_data('timestamp', time.time())
                step_start = time.time()
                # todo: LQR的输入要统一化，这里用task做一个适配层
                state = self.task.mujoco_state_adoption(self.sim_data)
                action = self.controller.generate_action(state, self.sim_data.time)
                self.sim_data = self.task.mujoco_action_adoption(action, self.sim_model, self.sim_data)
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.sim_model, self.sim_data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.sim_data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.sim_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        self.data_logger.fill_lost_data()
