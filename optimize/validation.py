import numpy as np
from .kinematics import forward_kinematics_with_params
from .data_utils import save_error_comparison

def validate_optimization(joint_angles, measured_positions, initial_params, optimized_params):
    """验证优化效果"""
    # 计算每个点的位置误差
    initial_errors = []
    optimized_errors = []
    residuals = []  # 新增：残差列表
    
    for i in range(len(joint_angles)):
        # 使用初始参数计算位置
        initial_pos = forward_kinematics_with_params(joint_angles[i], initial_params)
        initial_err = np.linalg.norm(initial_pos - measured_positions[i])
        initial_errors.append(initial_err)
        
        # 使用优化后参数计算位置
        optimized_pos = forward_kinematics_with_params(joint_angles[i], optimized_params)
        optimized_err = np.linalg.norm(optimized_pos - measured_positions[i])
        optimized_errors.append(optimized_err)
        
        # 新增：计算残差（优化后位置与测量位置的差值）
        residual = optimized_pos - measured_positions[i]
        residuals.append(residual)
    
    # 统计结果
    print("\n验证优化效果:")
    print(f"初始参数 - 平均误差: {np.mean(initial_errors):.4f} mm, 最大误差: {np.max(initial_errors):.4f} mm")
    print(f"优化后参数 - 平均误差: {np.mean(optimized_errors):.4f} mm, 最大误差: {np.max(optimized_errors):.4f} mm")
    print(f"误差改进: {(1 - np.mean(optimized_errors)/np.mean(initial_errors))*100:.2f}%")
    
    # 新增：输出残差统计信息
    residuals_array = np.array(residuals)
    print("\n残差统计(x,y,z分量):")
    print(f"平均残差: {np.mean(residuals_array, axis=0)} mm")
    print(f"最大残差: {np.max(np.abs(residuals_array), axis=0)} mm")
    print(f"残差标准差: {np.std(residuals_array, axis=0)} mm")
    
    # 保存误差数据
    save_error_comparison('result/error_comparison.txt', initial_errors, optimized_errors)
    
    return initial_errors, optimized_errors, residuals  # 修改：返回残差数据