from src.simulator.base_simulator import BaseSimulator
class IssacGymSimulator(BaseSimulator):
    def __init__(self, task, controller, cfg):
        super().__init__(task, controller, cfg)

    def get_mj_data(self):
        None

    def run_simulation(self):
        None