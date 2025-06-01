from src.controller.cartpole_controller.cartpole_lqr_controler import CartpoleLQRContoller
from src.controller.flywheel_controller.flywheel_lqr_controler import FlywheelLQRContoller
from src.controller.wheel_robot_controller.wheel_robot_lqr_controler import WheelRobotLQRContoller
from src.controller.robot_controller.h1.h1_mpc_controller import H1MPCContoller

from src.utils.registry import registry
registry.register_controller( "cartpole_lqr", CartpoleLQRContoller)
registry.register_controller( "flywheel_lqr", FlywheelLQRContoller)
registry.register_controller( "wheel_robot_lqr", WheelRobotLQRContoller)
registry.register_controller( "h1_mpc", H1MPCContoller)
