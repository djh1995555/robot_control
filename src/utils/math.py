import numpy as np
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def eul2Rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix"""
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def eul2quat(roll: float, pitch: float, yaw: float):
    """Convert Euler angles to quaternion"""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_quat()
