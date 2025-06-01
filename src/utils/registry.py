class Registry():
    def __init__(self):
        self.task_classes = {}
        self.controller_classes = {}
        self.simulator_classes = {}
    
    def register_task(self, name: str, task_class):
        self.task_classes[name] = task_class

    def register_controller(self, name: str, controller_class):
        self.controller_classes[name] = controller_class
    
    def register_simulator(self, name: str, simulator_class):
        self.simulator_classes[name] = simulator_class

    def get_components(self, simulator_name, task_name, controller_name, cfg):
        task = self.task_classes[task_name](cfg)
        controller = self.controller_classes[controller_name](cfg)
        simulator = self.simulator_classes[simulator_name](task, controller, cfg)
        mj_data = simulator.get_mj_data()
        controller.reset(mj_data)
        return simulator

registry = Registry()