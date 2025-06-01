from src.controller.base_controller.base_numerical_controler import BaseNumericalContoller
from jax.numpy.linalg import inv
import jax.numpy as jnp
import numpy as np

SENSOR_CART_POS = 0
SENSOR_CART_VEL = 1
SENSOR_POLE_ANG = 2
SENSOR_POLE_ANGVEL = 3

class CartpoleLQRContoller(BaseNumericalContoller):
    def __init__(self, cfg):
        super().__init__(cfg)

    def linearize_cartpole(self, params):
        """
        Linearize around upright equilibrium (x=0, theta=0, x_dot=0, theta_dot=0).
        Returns A, B (4x4, 4x1).
        """
        mc, mp, l, g = params

        # For small theta around 0, the linearized system:
        # A = [[0,    0,    1,        0],
        #      [0,    0,    0,        1],
        #      [0,  mp*g/mc, 0,       0],
        #      [0,  (mc+mp)*g/(mc*l), 0, 0]]
        # B = [[0],
        #      [0],
        #      [1/mc],
        #      [1/(mc*l)]]
        A = jnp.array([
            [0.,              0.,              1.,         0.],
            [0.,              0.,              0.,         1.],
            [0.,    (mp*g)/mc,                 0.,         0.],
            [0., (mc+mp)*g/(mc*l),             0.,         0.]
        ])
        B = jnp.array([
            [0.],
            [0.],
            [1./mc],
            [1./(mc*l)]
        ])
        return A, B

    def compute_lqr_gain(self, A, B, Q, R):
        """
        Solve the continuous-time algebraic Riccati equation for K.
        K = R^{-1} B^T P
        """
        # CARE: A'P + P A - P B R^-1 B' P + Q = 0
        # We'll do a direct numeric approach or use a known solver
        # For brevity, let's do a manual iteration (not the most robust, but simple).
        # In practice, you'd do something like slycot or a robust solver.
        # This is a naive iterative approach:

        P = jnp.eye(A.shape[0])
        for _ in range(200):
            dP = A.T @ P + P @ A - P @ B @ inv(R) @ B.T @ P + Q
            P = P + 0.01 * dP  # gradient step
        K = inv(R) @ B.T @ P
        return K

    def generate_action(self, data):
        # Read sensor data
        x = data.sensordata[SENSOR_CART_POS]
        x_dot = data.sensordata[SENSOR_CART_VEL]
        theta_raw = data.sensordata[SENSOR_POLE_ANG]
        theta = ((theta_raw + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]

        # build model
        mc = 1.0
        mp = 1.0
        l = 1.0
        g = 9.81
        params_jax = jnp.array([mc, mp, l, g])
        Q_lqr = jnp.diag(jnp.array([50.0, 100.0, 5.0, 20.0]))  # penalize x, theta, x_dot, theta_dot
        R_lqr = jnp.array([[0.1]])                            # penalize input force

        # Linearize around upright equilibrium
        A, B = self.linearize_cartpole(params_jax)
        K = self.compute_lqr_gain(A, B, Q_lqr, R_lqr)  # shape (1,4)

        # If your MuJoCo model uses a different zero angle for upright,
        # shift the sensor reading here, e.g.:
        # theta = theta - np.pi/2   # if sensor=+1.57 rad means "upright"

        state = np.array([x, theta, x_dot, theta_dot])
        
        # LQR control law:  u = -(K state)
        action = -(K @ state)[0]
        return action