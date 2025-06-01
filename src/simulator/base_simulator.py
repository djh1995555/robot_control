from src.utils.visualizer.data_logger import DataLogger
from src.utils.visualizer.report_generator import ReportGenerator

class BaseSimulator():
    def __init__(self, task, controller, cfg):
        self.cfg = cfg
        self.task = task
        self.controller = controller
        self.data_logger = DataLogger()
        self.controller.set_data_logger(self.data_logger)
        self.task.set_data_logger(self.data_logger)
        self.report_generator = ReportGenerator(cfg)

    def reset(self):
        None

    def run_simulation(self):
        None

    def generate_report(self):
        self.report_generator.generate_report(self.data_logger)