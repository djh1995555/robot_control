from env.simulator.mujoco_simulator import MujocoSimulator
from env.simulator.issac_gym_simulator import IssacGymSimulator

from utils.registry import registry
registry.register_simulator( "mujoco", MujocoSimulator)
registry.register_simulator( "issac_gym", IssacGymSimulator)
