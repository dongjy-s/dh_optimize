import numpy as np
from .forward_kinematics import RokaeRobot

def forward_kinematics_with_params(joint_angles, dh_params):
    """使用指定的DH参数计算前向运动学"""
    robot = RokaeRobot()
    
    # 更新DH参数
    for i in range(6):
        robot.modified_dh_params[i][0] = dh_params[i*4]      # theta_offset
        robot.modified_dh_params[i][1] = dh_params[i*4+1]    # d
        robot.modified_dh_params[i][2] = dh_params[i*4+2]    # alpha
        robot.modified_dh_params[i][3] = dh_params[i*4+3]    # a
    
    # 计算前向运动学
    result = robot.forward_kinematics(joint_angles, verbose=False)
    
    if result['valid']:
        return result['position']
    else:
        # 如果关节角度无效，返回一个大误差
        return np.array([1e6, 1e6, 1e6])

def error_function(dh_params, joint_angles, measured_positions):
    """计算预测位置与测量位置之间的误差"""
    n_samples = joint_angles.shape[0]
    errors = np.zeros(n_samples * 3)
    
    for i in range(n_samples):
        # 使用当前DH参数计算预测位置
        predicted_pos = forward_kinematics_with_params(joint_angles[i], dh_params)
        
        # 计算误差
        errors[i*3:(i+1)*3] = predicted_pos - measured_positions[i]
    
    return errors