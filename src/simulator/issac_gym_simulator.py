from src.simulator.base_simulator import BaseSimulator
class IssacGymSimulator(BaseSimulator):
    def __init__(self, task, controller, cfg):
        super().__init__(task, controller, cfg)

    def run_simulation(self):
        None