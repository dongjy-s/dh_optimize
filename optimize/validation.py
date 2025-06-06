import numpy as np
from .kinematics import forward_kinematics_with_params, normalize_quaternion
from .data_utils import save_error_comparison

def quaternion_angular_error(q1, q2):
    """
    计算两个四元数之间的角度误差（以弧度为单位）
    
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

def validate_optimization(joint_angles, measured_positions, initial_params, optimized_params, 
                         measured_quaternions=None):
    """验证优化效果"""
    # 计算每个点的位置误差
    initial_pos_errors = []
    optimized_pos_errors = []
    position_residuals = []  # 位置残差列表
    
    # 姿态误差 (角度)
    initial_ang_errors = []
    optimized_ang_errors = []
    
    for i in range(len(joint_angles)):
        # 使用初始参数计算位置和姿态
        initial_result = forward_kinematics_with_params(joint_angles[i], initial_params)
        initial_pos = initial_result[:3]
        initial_pos_err = np.linalg.norm(initial_pos - measured_positions[i])
        initial_pos_errors.append(initial_pos_err)
        
        # 使用优化后参数计算位置和姿态
        optimized_result = forward_kinematics_with_params(joint_angles[i], optimized_params)
        optimized_pos = optimized_result[:3]
        optimized_pos_err = np.linalg.norm(optimized_pos - measured_positions[i])
        optimized_pos_errors.append(optimized_pos_err)
        
        # 计算位置残差（优化后位置与测量位置的差值）
        position_residual = optimized_pos - measured_positions[i]
        position_residuals.append(position_residual)
        
        # 如果有四元数数据，计算姿态角度误差
        if measured_quaternions is not None:
            initial_quat = normalize_quaternion(initial_result[3:7])
            optimized_quat = normalize_quaternion(optimized_result[3:7])
            measured_quat = normalize_quaternion(measured_quaternions[i])
            
            # 计算初始和优化后的角度误差（弧度）
            initial_angle_err = quaternion_angular_error(initial_quat, measured_quat)
            optimized_angle_err = quaternion_angular_error(optimized_quat, measured_quat)
            
            # 转换为度
            initial_ang_errors.append(np.degrees(initial_angle_err))
            optimized_ang_errors.append(np.degrees(optimized_angle_err))
    
    # 统计结果
    print("\n验证优化效果:")
    print(f"初始参数 - 平均位置误差: {np.mean(initial_pos_errors):.4f} mm, 最大位置误差: {np.max(initial_pos_errors):.4f} mm")
    print(f"优化后参数 - 平均位置误差: {np.mean(optimized_pos_errors):.4f} mm, 最大位置误差: {np.max(optimized_pos_errors):.4f} mm")
    print(f"位置误差改进: {(1 - np.mean(optimized_pos_errors)/np.mean(initial_pos_errors))*100:.2f}%")
    
    # 输出位置残差统计信息
    position_residuals_array = np.array(position_residuals)
    print("\n位置残差统计(x,y,z分量):")
    print(f"平均残差: {np.mean(position_residuals_array, axis=0)} mm")
    print(f"最大残差: {np.max(np.abs(position_residuals_array), axis=0)} mm")
    print(f"残差标准差: {np.std(position_residuals_array, axis=0)} mm")
    
    # 如果有四元数数据，输出姿态角度误差统计信息
    if measured_quaternions is not None and len(initial_ang_errors) > 0:
        print("\n姿态误差统计(角度):")
        print(f"初始参数 - 平均角度误差: {np.mean(initial_ang_errors):.4f}°, 最大角度误差: {np.max(initial_ang_errors):.4f}°")
        print(f"优化后参数 - 平均角度误差: {np.mean(optimized_ang_errors):.4f}°, 最大角度误差: {np.max(optimized_ang_errors):.4f}°")
        print(f"姿态误差改进: {(1 - np.mean(optimized_ang_errors)/np.mean(initial_ang_errors))*100:.2f}%")
        
        # 添加详细的角度误差输出
        print("\n每个样本点的角度误差明细(度):")
        print("样本索引  初始误差  优化后误差  改进率(%)")
        print("-" * 40)
        for i in range(len(initial_ang_errors)):
            improvement = (1 - optimized_ang_errors[i]/initial_ang_errors[i]) * 100 if initial_ang_errors[i] > 0 else 0
            print(f"{i:^8}  {initial_ang_errors[i]:^8.2f}  {optimized_ang_errors[i]:^10.2f}  {improvement:^10.2f}")
            
        # 检查是否存在约90度的系统性偏差
        near_90_deg_errors = [err for err in initial_ang_errors if 85 <= err <= 95]
        if len(near_90_deg_errors) > len(initial_ang_errors) * 0.5:
            print("\n警告: 大部分误差接近90度，可能存在坐标系定义不一致或工具变换问题")
    
    # 保存误差数据
    save_error_comparison('result/error_comparison.txt', initial_pos_errors, optimized_pos_errors)
    
    # 返回结果
    return (initial_pos_errors, optimized_pos_errors, position_residuals, 
            (initial_ang_errors, optimized_ang_errors) if measured_quaternions is not None else None)