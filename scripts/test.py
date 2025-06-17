import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import warnings
import mujoco
import mujoco.viewer
from jax.numpy.linalg import inv
from controller.robot_controller.utils.kin_dyn_solver_back import KinDynSolver

cfg = dict()
cfg['urdf_path'] = 'resources/robot/h1/urdf/h1.urdf'
kin_dyn_solver = KinDynSolver(cfg)
kin_dyn_solver.compute_Jacobians()