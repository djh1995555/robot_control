class BaseContoller():
    def __init__(self, cfg):
        self.cfg = cfg

    def init_components(self, mj_model, mj_data):
        None
        
    def reset(self, data):
        None

    def set_data_logger(self, data_logger):
        self.data_logger = data_logger

    def record_logger(self):
        None

    def generate_action(self, state):
        None