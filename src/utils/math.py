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

def diffRot(Rcur, Rdes):
    """
    Compute the difference between two rotation matrices as a 3D vector.
    
    Parameters:
    Rcur -- current rotation matrix (3x3 numpy array)
    Rdes -- desired rotation matrix (3x3 numpy array)
    
    Returns:
    w -- 3D vector representing the rotation difference
    """
    R = Rcur.T @ Rdes
    w = np.zeros(3)
    
    # Check if R is identity (within tolerance)
    if np.allclose(R, np.eye(3), atol=1e-5) and abs(R[0,0]) + abs(R[1,1]) + abs(R[2,2]) - 3 < 1e-3:
        w = np.zeros(3)
    # Check if R is diagonal (within tolerance)
    elif np.allclose((R - np.diag(np.diag(R))), np.zeros((3,3)), atol=1e-5):
        w = np.array([R[0,0] + 1, R[1,1] + 1, R[2,2] + 1])
        w = w * np.pi / 2.0
    else:
        l = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
        sita = np.arctan2(np.linalg.norm(l), R[0,0] + R[1,1] + R[2,2] - 1)
        w = sita * l / np.linalg.norm(l)
    
    w = Rcur @ w
    return w