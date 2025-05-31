from env.simulator.mujoco_simulator import MujocoSimulator
from env.simulator.issac_gym_simulator import IssacGymSimulator

from utils.task_registry import task_registry
task_registry.register_simulator( "mujoco", MujocoSimulator)
task_registry.register_simulator( "issac_gym", IssacGymSimulator)
