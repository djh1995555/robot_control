from src.task.cartpole.cartpole import Cartpole
from src.task.flywheel.flywheel import Flywheel
from src.task.wheel_robot.wheel_robot import WheelRobot
from src.task.robot.h1_robot import H1Robot

from src.utils.registry import registry
registry.register_task( "cartpole", Cartpole)
registry.register_task( "flywheel", Flywheel)
registry.register_task( "wheel_robot", WheelRobot)
registry.register_task( "h1", H1Robot)
