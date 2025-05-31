class BaseSimulator():
    def __init__(self, task, controller, cfg):
        self.cfg = cfg
        self.task = task
        self.controller = controller