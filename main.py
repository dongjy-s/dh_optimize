import os
import traceback
import numpy as np  # 添加缺失的导入

from optimize.forward_kinematics import RokaeRobot
from optimize.utils import ensure_dir, print_dh_params, calculate_param_changes, setup_bounds, rmse
from optimize.data_utils import load_data, save_formatted_dh_params
from optimize.kinematics import error_function
from optimize.optimization import differential_evolution, optimize_with_lm
from optimize.validation import validate_optimization


def setup_adaptive_bounds(initial_params):
    """
    设置DH参数的优化边界
    
    参数:
        initial_params: 初始DH参数
    
    返回:
        bounds: 参数边界列表
    """
    bounds = []
    
    # 为每个连杆分别设置DH参数边界范围
    
    # 连杆1 (基座旋转)
    link1_theta_range = 1.0   # 关节角偏移范围 (度)
    link1_d_range = 1.0       # 连杆偏移范围 (mm)
    link1_alpha_range = 1.0   # 扭转角范围 (度)
    link1_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 连杆2 (肩部关节)
    link2_theta_range = 1.0   # 关节角偏移范围 (度)
    link2_d_range = 1.0       # 连杆偏移范围 (mm)
    link2_alpha_range = 1.0   # 扭转角范围 (度)
    link2_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 连杆3 (上臂)
    link3_theta_range = 1.0   # 关节角偏移范围 (度)
    link3_d_range = 1.0       # 连杆偏移范围 (mm)
    link3_alpha_range = 1.0   # 扭转角范围 (度)
    link3_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 连杆4 (肘部关节)
    link4_theta_range = 1.0   # 关节角偏移范围 (度)
    link4_d_range = 1.0       # 连杆偏移范围 (mm)
    link4_alpha_range = 1.0   # 扭转角范围 (度)
    link4_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 连杆5 (腕部关节)
    link5_theta_range = 1.0   # 关节角偏移范围 (度)
    link5_d_range = 1.0       # 连杆偏移范围 (mm)
    link5_alpha_range = 1.0   # 扭转角范围 (度)
    link5_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 连杆6 (末端关节) - 设置为零以便不优化
    link6_theta_range = 1.0   # 关节角偏移范围 (度)
    link6_d_range = 1.0       # 连杆偏移范围 (mm)
    link6_alpha_range = 1.0   # 扭转角范围 (度)
    link6_a_range = 1.0       # 连杆长度范围 (mm)
    
    # 为每个连杆的参数设置固定边界
    for i in range(0, len(initial_params), 4):
        link_index = i // 4 + 1
        
        # 获取当前连杆参数
        theta_offset = initial_params[i]
        d = initial_params[i+1]
        alpha = initial_params[i+2]
        a = initial_params[i+3]
        
        # 为不同连杆和不同参数类型设置特定边界
        if link_index == 1:  # 连杆1 (基座旋转)
            bounds.append((theta_offset - link1_theta_range, theta_offset + link1_theta_range))
            bounds.append((d - link1_d_range, d + link1_d_range))
            bounds.append((alpha - link1_alpha_range, alpha + link1_alpha_range))
            bounds.append((a - link1_a_range, a + link1_a_range))
        elif link_index == 2:  # 连杆2 (肩部关节)
            bounds.append((theta_offset - link2_theta_range, theta_offset + link2_theta_range))
            bounds.append((d - link2_d_range, d + link2_d_range))
            bounds.append((alpha - link2_alpha_range, alpha + link2_alpha_range))
            bounds.append((a - link2_a_range, a + link2_a_range))
        elif link_index == 3:  # 连杆3 (上臂)
            bounds.append((theta_offset - link3_theta_range, theta_offset + link3_theta_range))
            bounds.append((d - link3_d_range, d + link3_d_range))
            bounds.append((alpha - link3_alpha_range, alpha + link3_alpha_range))
            bounds.append((a - link3_a_range, a + link3_a_range))
        elif link_index == 4:  # 连杆4 (肘部关节)
            bounds.append((theta_offset - link4_theta_range, theta_offset + link4_theta_range))
            bounds.append((d - link4_d_range, d + link4_d_range))
            bounds.append((alpha - link4_alpha_range, alpha + link4_alpha_range))
            bounds.append((a - link4_a_range, a + link4_a_range))
        elif link_index == 5:  # 连杆5 (腕部关节)
            bounds.append((theta_offset - link5_theta_range, theta_offset + link5_theta_range))
            bounds.append((d - link5_d_range, d + link5_d_range))
            bounds.append((alpha - link5_alpha_range, alpha + link5_alpha_range))
            bounds.append((a - link5_a_range, a + link5_a_range))
        elif link_index == 6:  # 连杆6 (末端关节) - 不进行优化
            bounds.append((theta_offset - link6_theta_range, theta_offset + link6_theta_range))
            bounds.append((d - link6_d_range, d + link6_d_range))
            bounds.append((alpha - link6_alpha_range, alpha + link6_alpha_range))
            bounds.append((a - link6_a_range, a + link6_a_range))
    
    # 保留边界值有效性检查
    for i, (lower, upper) in enumerate(bounds):
        if lower >= upper:
            print(f"警告: 参数 {i} 的边界设置无效 [{lower}, {upper}]，将调整为有效边界")
            # 确保上下边界有效
            mean_val = (lower + upper) / 2
            bounds[i] = (mean_val - 1e-10, mean_val + 1e-10)
    
    # 打印边界设置以便调试
    print("\n参数优化边界:")
    for i in range(len(initial_params) // 4):
        print(f"Link {i+1}: theta_offset={bounds[i*4]}, d={bounds[i*4+1]}, "
              f"alpha={bounds[i*4+2]}, a={bounds[i*4+3]}")
    
    return bounds


def main():
    """
    主函数：执行机器人运动学参数优化的完整流程。
    
    返回:
        tuple: (优化后的DH参数, 最终RMSE误差) 或 (None, None)（如果发生错误）
    """
    # ====================== 可调参数配置 ======================
    # 数据文件配置
    data_file = 'data.txt'
    result_dir = 'result'
    result_file = f'{result_dir}/optimized_dh_params.txt'
    
    # 权重配置
    position_weight = 1.0      # 位置误差权重
    quaternion_weight = 0.0    # 姿态误差权重（DE优化阶段）
    quaternion_weight_lm = 0.0 # 姿态误差权重（LM优化阶段）
    
    # DE优化器配置
    de_config = {
        'with_quaternions': {  # 包含四元数数据时DE优化器参数
            'popsize': 50,     # 种群大小
            'maxiter': 120,    # 最大迭代次数
            'F': 0.5,          # 变异因子
            'CR': 0.7,         # 交叉概率
        },
        'position_only': {     # 仅包含位置数据时DE优化器参数
            'popsize': 30,     # 种群大小
            'maxiter': 100,    # 最大迭代次数
            'F': 0.5,          # 变异因子
            'CR': 0.9,         # 交叉概率
        }
    }
    # ==========================================================
    
    try:
        # 创建结果目录
        ensure_dir(result_dir)
        
        # 加载数据文件
        if not os.path.exists(data_file):
            print(f"错误: 找不到数据文件 '{data_file}'")
            print(f"当前工作目录: {os.getcwd()}")
            print("请确保从正确的目录运行程序或提供正确的文件路径")
            return None, None
        
        # 加载测量数据
        try:
            data = load_data(data_file)
            if len(data) == 3:
                joint_angles, measured_positions, measured_quaternions = data
                print(f"加载了 {len(joint_angles)} 组样本数据，包含位置和姿态四元数")
                use_quaternions = True
                de_params = de_config['with_quaternions']
            else:
                joint_angles, measured_positions = data
                measured_quaternions = None
                use_quaternions = False
                de_params = de_config['position_only']
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
                position_weight=position_weight, 
                quaternion_weight=quaternion_weight
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
        bounds = setup_adaptive_bounds(initial_dh_params)
        
        # 第一阶段：使用DE算法进行全局搜索
        print("\n开始DE全局优化...")
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
                position_weight=position_weight,
                quaternion_weight=quaternion_weight
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
                    position_weight=position_weight,
                    quaternion_weight=quaternion_weight_lm
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