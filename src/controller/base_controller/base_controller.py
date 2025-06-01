class BaseContoller():
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, data):
        None

    def set_data_logger(self, data_logger):
        self.data_logger = data_logger

    def record_logger(self):
        None

    def generate_action(self, data):
        None