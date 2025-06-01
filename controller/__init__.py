from controller.cartpole_controller.cartpole_lqr_controler import CartpoleLQRContoller

from utils.registry import registry
registry.register_controller( "cartpole_lqr", CartpoleLQRContoller)
