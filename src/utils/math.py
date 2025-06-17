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

# 计算加权伪逆
# 主要用于任务空间到关节空间的映射，特别是在加权最小二乘（Weighted Least Squares）问题中
# 常用于层级任务控制，计算任务的零空间投影或优先级调整
def pseudoInv_right_weighted(J, W):
    return np.linalg.pinv(J.T @ W @ J) @ J.T @ W

# 计算动态伪逆
# 主要用于动力学控制（如加速度级控制），考虑机器人的动力学特性（如质量矩阵 M）
# 在操作空间动力学（Operational Space Dynamics）中，用于计算加速度命令对应的关节加速度。
def dyn_pseudoInv(J, M_inv, use_dynamics=True):
    if use_dynamics:
        return M_inv @ J.T @ np.linalg.pinv(J @ M_inv @ J.T)
    else:
        return np.linalg.pinv(J)