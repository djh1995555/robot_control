import time
from src.simulator.base_simulator import BaseSimulator
import mujoco
import mujoco.viewer

class MujocoSimulator(BaseSimulator):
    def __init__(self, task, controller, cfg):
        super().__init__(task, controller, cfg)
        self.cfg = cfg['mujoco_cfg']
        self.model = mujoco.MjModel.from_xml_path(self.cfg['xml_path'])
        self.data = mujoco.MjData(self.model)

        self.data = self.task.mujoco_init_state(self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_mj_data(self):
        return self.data
    
    def run_simulation(self):
        # Close the viewer automatically after 30 wall-seconds.
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < self.cfg['sim_duration']:
                current_time = time.time()
                step_start = time.time()
                # todo: LQR的输入要统一化，这里用task做一个适配层
                action = self.controller.generate_action(self.data)
                self.data = self.task.mujoco_action_adoption(action, self.model, self.data)
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.model, self.data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
