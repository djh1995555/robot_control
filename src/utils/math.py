import numpy as np

def eul2rot(roll, pitch, yaw):
    """
    将欧拉角转换为旋转矩阵 (Z-Y-X顺序)
    
    参数:
        roll:  绕X轴的旋转角度(弧度)
        pitch: 绕Y轴的旋转角度(弧度)
        yaw:   绕Z轴的旋转角度(弧度)
        
    返回:
        3x3 旋转矩阵
    """
    # 绕Z轴旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,           0,            1]
    ])
    
    # 绕Y轴旋转矩阵
    Ry = np.array([
        [np.cos(pitch),  0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # 绕X轴旋转矩阵
    Rx = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # 组合旋转 (Z-Y-X顺序)
    return Rz @ Ry @ Rx

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

