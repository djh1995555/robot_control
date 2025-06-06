import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import warnings
import mujoco
import mujoco.viewer
from jax.numpy.linalg import inv

def linearize_cartpole(params):
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

def compute_lqr_gain(A, B, Q, R):
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

def mujoco_lqr_controller(data, K):
    """
    1) Read state [x, theta, x_dot, theta_dot] from sensor data.
    2) Compute control:  u = -(K @ state).
    3) Return u as the force on the cart.
    """
    # Read sensor data
    x = data.sensordata[SENSOR_CART_POS]
    x_dot = data.sensordata[SENSOR_CART_VEL]
    theta_raw = data.sensordata[SENSOR_POLE_ANG]
    theta = ((theta_raw + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    theta_dot = data.sensordata[SENSOR_POLE_ANGVEL]

    # If your MuJoCo model uses a different zero angle for upright,
    # shift the sensor reading here, e.g.:
    # theta = theta - np.pi/2   # if sensor=+1.57 rad means "upright"

    state = np.array([x, theta, x_dot, theta_dot])
    
    # LQR control law:  u = -(K state)
    force = -(K @ state)[0]
    return force

mc = 1.0
mp = 1.0
l = 1.0
g = 9.81
params_jax = jnp.array([mc, mp, l, g])

# Cost matrices
Q_lqr = jnp.diag(jnp.array([50.0, 100.0, 5.0, 20.0]))  # penalize x, theta, x_dot, theta_dot
R_lqr = jnp.array([[0.1]])                            # penalize input force

# Linearize around upright equilibrium
A, B = linearize_cartpole(params_jax)
K = compute_lqr_gain(A, B, Q_lqr, R_lqr)  # shape (1,4)
print("LQR gain K =", K)

model = mujoco.MjModel.from_xml_path('resources/cartpole/cart_pole.xml')
data = mujoco.MjData(model)

# Indices for sensor data (assuming same order as your XML):
SENSOR_CART_POS = 0
SENSOR_CART_VEL = 1
SENSOR_POLE_ANG = 2
SENSOR_POLE_ANGVEL = 3


###############################################################################
# 4. Set Initial Conditions (optional)
###############################################################################
# If you only have 1 DOF for cart x and 1 DOF for pole hinge, their qpos indices might be:
data.qpos[0] = 2.5   # cart x at 0
data.qpos[5] = 0.5   # pole hinge angle (slightly tilted)
data.qvel[0] = -0.15
data.qvel[5] = -0.35

# Re-forward for consistency
mujoco.mj_forward(model, data)

sim_time = 0.0
sim_duration = 10

time_log = []
force_log = []
x_log = []
theta_log = []
xdot_log = []
thetadot_log = []

with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < sim_duration:
    current_time = time.time()
    step_start = time.time()
    force = mujoco_lqr_controller(data, K)

    if (current_time - start) > 17:
        disturbance = -40
    elif (current_time - start) > 12:
        disturbance = 0
    elif (current_time - start) > 7:
        disturbance = 40
    else:
        disturbance = 0

    data.ctrl[0] = force + disturbance
    if model.nu > 1:
        data.ctrl[1] = 0.0 
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(model, data)

    sim_time += model.opt.timestep
    time_log.append(sim_time)
    force_log.append(force)
    x_log.append(data.sensordata[SENSOR_CART_POS])
    theta_log.append(data.sensordata[SENSOR_POLE_ANG])
    xdot_log.append(data.sensordata[SENSOR_CART_VEL])
    thetadot_log.append(data.sensordata[SENSOR_POLE_ANGVEL])

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

###############################################################################
# 6. Plot the Results
###############################################################################
time_log = np.array(time_log)
force_log = np.array(force_log)
x_log = np.array(x_log)
theta_log = np.array(theta_log)
xdot_log = np.array(xdot_log)
thetadot_log = np.array(thetadot_log)

# Force vs Time
plt.figure()
plt.plot(time_log, force_log, label="Force")
plt.title("LQR Cart Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.show()

# 4 Subplots for x, theta, x_dot, theta_dot
fig, axs = plt.subplots(2, 2, figsize=(10,6), sharex=True)
fig.suptitle("Cart-Pole States (LQR)")

axs[0,0].plot(time_log, x_log, 'b')
axs[0,0].set_ylabel("x (m)")
axs[0,0].set_title("Cart Position")
axs[0,0].grid(True)

axs[0,1].plot(time_log, theta_log, 'r')
axs[0,1].set_ylabel("theta (rad)")
axs[0,1].set_title("Pole Angle")
axs[0,1].grid(True)

axs[1,0].plot(time_log, xdot_log, 'g')
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("x_dot (m/s)")
axs[1,0].set_title("Cart Velocity")
axs[1,0].grid(True)

axs[1,1].plot(time_log, thetadot_log, 'm')
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("theta_dot (rad/s)")
axs[1,1].set_title("Pole Angular Velocity")
axs[1,1].grid(True)

plt.tight_layout()
plt.show()

print("LQR-based simulation finished.")