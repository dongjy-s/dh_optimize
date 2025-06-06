import numpy as np
from .forward_kinematics import RokaeRobot

def forward_kinematics_with_params(joint_angles, dh_params):
    """
    使用指定的DH参数计算前向运动学
    
    参数:
        joint_angles: 关节角度
        dh_params: DH参数列表
        
    返回:
        numpy.ndarray: 包含位置和四元数的数组 [x, y, z, qx, qy, qz, qw]
    """
    robot = RokaeRobot()
    
    # 更新DH参数
    for i in range(6):
        robot.modified_dh_params[i][0] = dh_params[i*4]      # theta_offset
        robot.modified_dh_params[i][1] = dh_params[i*4+1]    # d
        robot.modified_dh_params[i][2] = dh_params[i*4+2]    # alpha
        robot.modified_dh_params[i][3] = dh_params[i*4+3]    # a
    
    # 计算前向运动学
    result = robot.forward_kinematics(joint_angles, verbose=False, use_tool=True)
    
    if result['valid']:
        # 返回位置和四元数
        # result['position'] 包含 [x, y, z, qx, qy, qz, qw]
        return result['position']
    else:
        # 如果关节角度无效，返回一个大误差
        return np.array([1e6, 1e6, 1e6, 1, 0, 0, 0])  # 大位置误差和单位四元数

def normalize_quaternion(q):
    """归一化四元数"""
    norm = np.sqrt(np.sum(np.square(q)))
    if norm < 1e-10:
        return np.array([0, 0, 0, 1])  # 返回单位四元数
    return q / norm

def quaternion_to_euler(q):
    """
    将四元数转换为欧拉角（ZYX顺序，即RPY - Roll, Pitch, Yaw）
    
    Args:
        q: 四元数 [qx, qy, qz, qw]
        
    Returns:
        numpy.ndarray: 欧拉角 [roll, pitch, yaw]（单位：弧度）
    """
    # 归一化四元数
    q = normalize_quaternion(q)
    qx, qy, qz, qw = q
    
    # 计算欧拉角
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 使用pi/2，如果sinp = ±1
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def euler_angle_error(euler1, euler2):
    """
    计算两组欧拉角之间的误差
    
    Args:
        euler1, euler2: 两组欧拉角 [roll, pitch, yaw]
        
    Returns:
        numpy.ndarray: 每个轴的角度误差 [roll_err, pitch_err, yaw_err]
    """
    # 计算各轴角度差
    error = euler1 - euler2
    
    # 将角度差映射到[-pi, pi]范围内
    for i in range(3):
        while error[i] > np.pi:
            error[i] -= 2 * np.pi
        while error[i] < -np.pi:
            error[i] += 2 * np.pi
    
    return error

def quaternion_angular_error_axes(q1, q2):
    """
    计算两个四元数之间在各轴上的角度误差
    
    Args:
        q1, q2: 两个四元数 [qx, qy, qz, qw]
        
    Returns:
        numpy.ndarray: 各轴角度误差 [roll_err, pitch_err, yaw_err]（单位：弧度）
    """
    # 将四元数转换为欧拉角
    euler1 = quaternion_to_euler(q1)
    euler2 = quaternion_to_euler(q2)
    
    # 计算欧拉角误差
    return euler_angle_error(euler1, euler2)

def quaternion_angular_error(q1, q2):
    """
    计算两个四元数之间的总角度误差（以弧度为单位）
    
    Args:
        q1, q2: 两个四元数 [qx, qy, qz, qw]
        
    Returns:
        float: 角度误差（弧度）
    """
    # 确保四元数已归一化
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    # 四元数表示旋转时，q和-q表示相同的旋转，所以需要选择较小的角度
    # 比较q1和q2与q1和-q2之间的角度，取较小值
    dot_product1 = np.sum(q1 * q2)
    dot_product2 = np.sum(q1 * (-q2))
    
    # 选择绝对值较大的点积（角度较小）
    dot_product = max(np.abs(dot_product1), np.abs(dot_product2))
    
    # 确保点积在[-1, 1]范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算角度误差
    angle = 2.0 * np.arccos(dot_product)
    
    return angle

def error_function(dh_params, joint_angles, measured_positions, measured_quaternions=None, position_weight=1.0, quaternion_weight=0.5):
    """
    计算预测位置与测量位置之间的误差，以及预测姿态与测量姿态之间的误差（如果提供）
    
    Args:
        dh_params: DH参数
        joint_angles: 关节角度
        measured_positions: 测量的位置坐标
        measured_quaternions: 测量的姿态四元数（可选）
        position_weight: 位置误差的权重（默认1.0）
        quaternion_weight: 四元数角度误差的权重（默认0.5）
        
    Returns:
        errors: 位置误差和姿态角度误差
    """
    n_samples = joint_angles.shape[0]
    
    # 计算误差向量的长度
    pos_error_length = n_samples * 3
    quat_error_length = n_samples * 3 if measured_quaternions is not None else 0  # 使用3维表示姿态误差
    total_error_length = pos_error_length + quat_error_length
    
    # 初始化误差数组
    errors = np.zeros(total_error_length)
    
    for i in range(n_samples):
        # 使用当前DH参数计算预测位置和姿态
        predicted_result = forward_kinematics_with_params(joint_angles[i], dh_params)
        
        # 提取预测的位置
        predicted_pos = predicted_result[:3]
        
        # 计算位置误差并加权
        pos_errors = (predicted_pos - measured_positions[i]) * position_weight
        errors[i*3:(i+1)*3] = pos_errors
        
        # 如果有四元数数据，计算姿态角度误差
        if measured_quaternions is not None:
            # 提取预测的四元数并归一化
            predicted_quat = normalize_quaternion(predicted_result[3:7])
            
            # 获取测量的四元数并归一化
            measured_quat = normalize_quaternion(measured_quaternions[i])
            
            # 计算四元数在各轴上的角度误差（弧度）
            axis_errors = quaternion_angular_error_axes(predicted_quat, measured_quat)
            
            # 加权并存储角度误差（3个值，对应roll, pitch, yaw）
            quat_error_vector = axis_errors * quaternion_weight
            
            # 保存姿态角度误差（3个维度）
            errors[pos_error_length + i*3:pos_error_length + (i+1)*3] = quat_error_vector
    
    return errors