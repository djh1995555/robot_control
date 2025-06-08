from pinocchio import pin

class KinDynSolver():
    def __init__(self, cfg):
        urdf_path = cfg['urdf_path']
        self.model_biped = pin.Model()
        pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer(), self.model_biped)
        self.model_biped_fixed = pin.buildModelFromUrdf(urdf_path)