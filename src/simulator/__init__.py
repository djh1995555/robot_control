from src.simulator.mujoco_simulator import MujocoSimulator
from src.simulator.issac_gym_simulator import IssacGymSimulator

from src.utils.registry import registry
registry.register_simulator( "mujoco", MujocoSimulator)
registry.register_simulator( "issac_gym", IssacGymSimulator)
