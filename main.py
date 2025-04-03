import os
import traceback
import numpy as np

from optimize.forward_kinematics import RokaeRobot
from optimize.utils import ensure_dir, print_dh_params, calculate_param_changes, setup_bounds, rmse
from optimize.data_utils import load_data, save_formatted_dh_params
from optimize.kinematics import error_function
from optimize.optimization import differential_evolution, optimize_with_lm
from optimize.validation import validate_optimization
from optimize.boundaries import (
    setup_adaptive_bounds, 
    adjust_bounds_dynamically, 
    DE_CONFIG, 
    INITIAL_SCALE, 
    BOUNDARY_ADJUSTMENT_INTERVALS,
    POSITION_WEIGHT,
    QUATERNION_WEIGHT_DE,
    QUATERNION_WEIGHT_LM
)


# 打印系统设置信息
print("使用参数优化配置:")
print(f"- 位置误差权重: {POSITION_WEIGHT}")
print(f"- 四元数误差权重 (DE): {QUATERNION_WEIGHT_DE}")
print(f"- 四元数误差权重 (LM): {QUATERNION_WEIGHT_LM}")
print(f"- 边界调整间隔: {BOUNDARY_ADJUSTMENT_INTERVALS} 次迭代")

def main():
    """
    主函数：执行机器人运动学参数优化的完整流程。
    
    返回:
        tuple: (优化后的DH参数, 最终RMSE误差) 或 (None, None)（如果发生错误）
    """
    # ====================== 数据文件配置 ======================
    data_file = 'data.txt'  # 数据文件路径
    result_dir = 'result'
    result_file = f'{result_dir}/optimized_dh_params.txt'
    
    # 是否使用动态边界调整
    dynamic_bounds = True
    # ==========================================================
    
    try:
        # 创建结果目录
        ensure_dir(result_dir)
        
        # 加载数据文件，增加更好的错误处理
        if not os.path.exists(data_file):
            print(f"错误: 找不到数据文件 '{data_file}'")
            print(f"当前工作目录: {os.getcwd()}")
            print("请确保数据文件(data.txt 或 data_local.txt)存在于当前目录")
            return None, None
        
        print(f"使用数据文件: {data_file}")
        
        # 加载测量数据
        try:
            data = load_data(data_file)
            if len(data) == 3:
                joint_angles, measured_positions, measured_quaternions = data
                print(f"加载了 {len(joint_angles)} 组样本数据，包含位置和姿态四元数")
                use_quaternions = True
                de_params = DE_CONFIG['with_quaternions']
            else:
                joint_angles, measured_positions = data
                measured_quaternions = None
                use_quaternions = False
                de_params = DE_CONFIG['position_only']
                print(f"加载了 {len(joint_angles)} 组样本数据（仅包含位置数据）")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            traceback.print_exc()
            return None, None
        
        # 获取初始DH参数
        robot = RokaeRobot()
        initial_dh_params = []
        for i in range(6):
            initial_dh_params.extend(robot.modified_dh_params[i])
        
        print_dh_params(initial_dh_params, "初始")
        
        # 计算初始误差
        if use_quaternions:
            initial_errors = error_function(
                initial_dh_params, 
                joint_angles, 
                measured_positions, 
                measured_quaternions=measured_quaternions,
                position_weight=POSITION_WEIGHT, 
                quaternion_weight=QUATERNION_WEIGHT_DE
            )
            # 添加检查
            if any(np.isnan(initial_errors)):
                print("警告: 初始误差计算中存在NaN值，可能影响优化结果")
        else:
            initial_errors = error_function(
                initial_dh_params, 
                joint_angles, 
                measured_positions
            )
        
        initial_rmse = rmse(initial_errors)
        print(f"初始RMSE: {initial_rmse:.6f}")
        
        # 设置参数优化边界
        bounds = setup_adaptive_bounds(initial_dh_params, INITIAL_SCALE)
        
        # 第一阶段：使用DE算法进行全局搜索
        print("\n开始DE全局优化...")
        if dynamic_bounds:
            # 使用自定义DE实现以支持动态边界调整
            print("启用动态边界调整策略")
            
            # 初始化历史记录
            de_history = {'fitness': [], 'best_x': []}
            
            # 定义动态优化过程的回调函数
            def de_callback(xk, convergence):
                """每次迭代后的回调函数"""
                nonlocal bounds, de_history, de_params, initial_rmse
                
                # 计算当前迭代次数
                iteration = len(de_history['fitness']) - 1 if 'fitness' in de_history and len(de_history['fitness']) > 0 else 0
                
                # 获取当前种群的最佳适应度
                current_best = de_history['fitness'][-1] if 'fitness' in de_history and de_history['fitness'] else float('inf')
                
                # 仅在特定迭代时调整边界，其他时候不做任何修改
                if iteration > 0 and iteration % BOUNDARY_ADJUSTMENT_INTERVALS == 0:
                    # 获取前一次的适应度值，确保索引有效
                    prev_best = de_history['fitness'][-2] if len(de_history['fitness']) >= 2 else initial_rmse
                    
                    # 确保不超过最大迭代次数
                    if iteration >= de_params['maxiter']:
                        print(f"已达到最大迭代次数 {de_params['maxiter']}，终止优化")
                        return False, bounds  # 返回False阻止继续优化
                    
                    # 动态调整边界
                    print(f"\n迭代 {iteration}/{de_params['maxiter']}, 当前适应度: {current_best:.6f}")
                    
                    # 使用专门的边界调整函数，确保正确传递前一次和当前的适应度值
                    if abs(prev_best - current_best) < 1e-10:
                        # 如果适应度没有变化，添加一个微小差值以避免改善率为0
                        new_bounds = adjust_bounds_dynamically(xk, prev_best, prev_best * 0.9999, bounds)
                    else:
                        new_bounds = adjust_bounds_dynamically(xk, prev_best, current_best, bounds)
                    
                    bounds = new_bounds
                    
                    # 更新DE算法使用的边界
                    return True, bounds
                
                # 大多数情况下返回None, bounds以保持当前边界不变
                return None, bounds
            
            # 调用支持动态边界的DE优化
            if use_quaternions:
                de_optimized_params, de_fitness, de_history = differential_evolution(
                    error_function, 
                    bounds, 
                    joint_angles, 
                    measured_positions, 
                    popsize=de_params['popsize'],
                    maxiter=de_params['maxiter'],
                    F=de_params['F'],
                    CR=de_params['CR'],
                    measured_quaternions=measured_quaternions,
                    position_weight=POSITION_WEIGHT,
                    quaternion_weight=QUATERNION_WEIGHT_DE,
                    callback=de_callback,
                    history=de_history
                )
            else:
                de_optimized_params, de_fitness, de_history = differential_evolution(
                    error_function, 
                    bounds, 
                    joint_angles, 
                    measured_positions, 
                    popsize=de_params['popsize'], 
                    maxiter=de_params['maxiter'], 
                    F=de_params['F'], 
                    CR=de_params['CR'],
                    callback=de_callback,
                    history=de_history
                )
        else:
            # 使用标准DE实现（不支持动态边界调整）
            if use_quaternions:
                de_optimized_params, de_fitness, de_history = differential_evolution(
                    error_function, 
                    bounds, 
                    joint_angles, 
                    measured_positions, 
                    popsize=de_params['popsize'],
                    maxiter=de_params['maxiter'],
                    F=de_params['F'],
                    CR=de_params['CR'],
                    measured_quaternions=measured_quaternions,
                    position_weight=POSITION_WEIGHT,
                    quaternion_weight=QUATERNION_WEIGHT_DE
                )
            else:
                de_optimized_params, de_fitness, de_history = differential_evolution(
                    error_function, 
                    bounds, 
                    joint_angles, 
                    measured_positions, 
                    popsize=de_params['popsize'], 
                    maxiter=de_params['maxiter'], 
                    F=de_params['F'], 
                    CR=de_params['CR']
                )
        
        print_dh_params(de_optimized_params, "DE优化")
        print(f"DE优化后RMSE: {de_fitness:.6f} mm")
        
        # 检查DE优化结果
        if de_fitness >= initial_rmse * 0.999:  # 允许微小的改善
            print("警告: DE优化没有显著改善结果，将使用初始参数进行LM优化")
            de_optimized_params = initial_dh_params.copy()  # 使用copy()避免引用问题
        
        # 为LM优化阶段设置边界
        if np.array_equal(de_optimized_params, initial_dh_params):  # 使用值比较代替身份比较
            # 如果回退到初始参数，边界也应基于初始参数
            local_bounds = setup_adaptive_bounds(initial_dh_params)
        else:
            local_bounds = setup_adaptive_bounds(de_optimized_params)
        
        # 第二阶段：使用LM算法进行局部优化
        try:
            if use_quaternions:
                final_params, final_rmse = optimize_with_lm(
                    de_optimized_params, 
                    joint_angles, 
                    measured_positions, 
                    error_function, 
                    local_bounds,  # 使用针对局部优化的更窄边界
                    measured_quaternions=measured_quaternions,
                    position_weight=POSITION_WEIGHT,
                    quaternion_weight=QUATERNION_WEIGHT_LM
                )
            else:
                final_params, final_rmse = optimize_with_lm(
                    de_optimized_params, 
                    joint_angles, 
                    measured_positions, 
                    error_function, 
                    local_bounds  # 使用针对局部优化的更窄边界
                )
        except Exception as e:
            print(f"LM优化时出错: {e}")
            traceback.print_exc()
            print("将使用DE优化结果作为最终结果")
            final_params, final_rmse = de_optimized_params, de_fitness

        print_dh_params(final_params, "最终优化")
        print(f"最终优化后RMSE: {final_rmse:.6f} mm")
        
        # 计算参数变化并输出
        calculate_param_changes(initial_dh_params, final_params)
        
        # 验证优化效果
        validate_kwargs = {}
        if use_quaternions:
            validate_kwargs['measured_quaternions'] = measured_quaternions
            
        validate_optimization(
            joint_angles, 
            measured_positions, 
            initial_dh_params, 
            final_params,
            **validate_kwargs
        )

        # 保存优化后的参数
        save_formatted_dh_params(result_file, final_params, initial_dh_params)
        print(f"优化后的参数已保存到 {result_file}")
        
        # 对最终优化结果的有效性检查
        if final_rmse > initial_rmse:
            print("警告: 最终优化结果比初始结果更差，建议检查优化过程")
            # 可以考虑在这种情况下回退到初始参数
            # final_params = initial_dh_params
            # final_rmse = initial_rmse
        
        return final_params, final_rmse
    
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()