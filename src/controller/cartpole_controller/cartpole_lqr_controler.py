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

    def record_logger(self):
        self.data_logger.add_data('cart_pos', self.x)
        self.data_logger.add_data('cart_v', self.x_dot)
        self.data_logger.add_data('pole_pos', self.theta)
        self.data_logger.add_data('pole_v', self.theta_dot)
        self.data_logger.add_data('action', self.action)
        self.data_logger.add_data('K0', self.K[0][0])
        self.data_logger.add_data('K1', self.K[0][1])
        self.data_logger.add_data('K2', self.K[0][2])
        self.data_logger.add_data('K3', self.K[0][3])

    def generate_action(self, data):
        # Read sensor data
        self.x = data.sensordata[SENSOR_CART_POS]
        self.x_dot = data.sensordata[SENSOR_CART_VEL]
        theta_raw = data.sensordata[SENSOR_POLE_ANG]
        self.theta = ((theta_raw + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        self.theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]
        state = np.array([self.x, self.theta, self.x_dot, self.theta_dot])

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
        self.K = self.compute_lqr_gain(A, B, Q_lqr, R_lqr)  # shape (1,4)
        
        # LQR control law:  u = -(K state)
        self.action = -(self.K @ state)[0]

        self.record_logger()
        return self.action