import numpy as np

def normalize_quaternion(q):
    """
    归一化四元数
    
    参数:
        q: 四元数 [x, y, z, w]
        
    返回:
        numpy.ndarray: 归一化后的四元数
    """
    norm = np.sqrt(np.sum(np.square(q)))
    if norm < 1e-10:
        return np.array([0, 0, 0, 1])  # 返回单位四元数
    return q / norm

def create_tool_transform(tool_position, tool_quaternion):
    """
    创建工具变换矩阵
    
    Args:
        tool_position (list): 工具位置 [x, y, z] 单位: mm
        tool_quaternion (list): 工具四元数 [x, y, z, w]
        
    Returns:
        numpy.ndarray: 4x4 工具变换矩阵
    """
    # 确保四元数归一化
    tool_quaternion = normalize_quaternion(tool_quaternion)
    
    # 从四元数创建旋转矩阵
    x, y, z, w = tool_quaternion
    tool_rotation = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])
    
    # 创建4x4变换矩阵
    tool_transform = np.eye(4)
    tool_transform[:3, :3] = tool_rotation
    tool_transform[:3, 3] = tool_position
    
    return tool_transform

def rotation_matrix_to_quaternion(R):
    """
    将3x3旋转矩阵转换为四元数 [qx, qy, qz, qw]
    
    Args:
        R (numpy.ndarray): 3x3旋转矩阵
        
    Returns:
        list: 四元数 [qx, qy, qz, qw]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    # 归一化四元数
    return normalize_quaternion([qx, qy, qz, qw])
