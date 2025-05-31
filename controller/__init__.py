from controller.cartpole_controller.cartpole_lqr_controler import CartpoleLQRContoller

from utils.task_registry import task_registry
task_registry.register_controller( "cartpole_lqr", CartpoleLQRContoller)
