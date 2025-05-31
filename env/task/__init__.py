from env.task.cartpole.cartpole import Cartpole

from utils.task_registry import task_registry
task_registry.register_task( "cartpole", Cartpole)
