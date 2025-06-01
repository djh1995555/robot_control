from env.task.cartpole.cartpole import Cartpole

from utils.registry import registry
registry.register_task( "cartpole", Cartpole)
