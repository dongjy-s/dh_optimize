import numpy as np
from .kinematics import forward_kinematics_with_params
from .data_utils import save_error_comparison

def validate_optimization(joint_angles, measured_positions, initial_params, optimized_params):
    """验证优化效果"""
    # 计算每个点的位置误差
    initial_errors = []
    optimized_errors = []
    
    for i in range(len(joint_angles)):
        # 使用初始参数计算位置
        initial_pos = forward_kinematics_with_params(joint_angles[i], initial_params)
        initial_err = np.linalg.norm(initial_pos - measured_positions[i])
        initial_errors.append(initial_err)
        
        # 使用优化后参数计算位置
        optimized_pos = forward_kinematics_with_params(joint_angles[i], optimized_params)
        optimized_err = np.linalg.norm(optimized_pos - measured_positions[i])
        optimized_errors.append(optimized_err)
    
    # 统计结果
    print("\n验证优化效果:")
    print(f"初始参数 - 平均误差: {np.mean(initial_errors):.4f} mm, 最大误差: {np.max(initial_errors):.4f} mm")
    print(f"优化后参数 - 平均误差: {np.mean(optimized_errors):.4f} mm, 最大误差: {np.max(optimized_errors):.4f} mm")
    print(f"误差改进: {(1 - np.mean(optimized_errors)/np.mean(initial_errors))*100:.2f}%")
      
    # 保存误差数据
    save_error_comparison('result/error_comparison.txt', initial_errors, optimized_errors)
    
    return initial_errors, optimized_errors