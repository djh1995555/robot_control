from src.controller.base_controller.base_controller import BaseContoller
 
class BaseRLContoller(BaseContoller):
    def __init__(self, cfg):
        super().__init__(cfg)
