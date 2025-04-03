import os
import traceback
import numpy as np

from optimize.forward_kinematics import RokaeRobot
from optimize.utils import ensure_dir, print_dh_params, calculate_param_changes, setup_bounds, rmse
from optimize.data_utils import load_data, save_formatted_dh_params
from optimize.kinematics import error_function
from optimize.optimization import differential_evolution, optimize_with_lm
from optimize.validation import validate_optimization


 # 使用默认配置
print("默认参数边界配置：")
PARAM_RANGES = {
    # 连杆索引: [theta_range, d_range, alpha_range, a_range]
    # 注意: 参数范围为0.0表示该参数固定不变，大于0表示可以在范围内优化
    1: [1.0, 1.0, 1.0, 1.0],  # 基座旋转 
    2: [1.0, 1.0, 1.0, 1.0],  # 肩部关节 
    3: [1.0, 1.0, 1.0, 1.0],  # 上臂 
    4: [1.0, 1.0, 1.0, 1.0],  # 肘部关节 
    5: [1.0, 1.0, 1.0, 1.0],  # 腕部关节 
    6: [1.0, 1.0, 1.0, 1.0],  # 末端关节 
}
INITIAL_SCALE = 1.0
MIN_SCALE = 0.1
ADJUSTMENT_RATE = 0.8
BOUNDARY_ADJUSTMENT_INTERVALS = 20
DE_CONFIG = {
    'with_quaternions': {
        'popsize': 50, 'maxiter': 120, 'F': 0.5, 'CR': 0.7,
    },
    'position_only': {
        'popsize': 30, 'maxiter': 100, 'F': 0.5, 'CR': 0.9,
    }
}

def setup_adaptive_bounds(initial_params, scale_factor=1.0, param_ranges=None):
    """
    设置DH参数的优化边界，支持动态调整边界范围
    
    参数:
        initial_params: 初始DH参数
        scale_factor: 边界范围缩放因子(0-1之间)，用于动态调整边界
        param_ranges: 参数范围配置字典
    
    返回:
        bounds: 参数边界列表
    """
    bounds = []
    
    # 使用配置文件中的参数范围或默认范围
    base_ranges = param_ranges if param_ranges is not None else PARAM_RANGES
    
    # 应用缩放因子
    for i in range(0, len(initial_params), 4):
        link_index = i // 4 + 1
        
        # 获取当前连杆参数
        theta_offset = initial_params[i]
        d = initial_params[i+1]
        alpha = initial_params[i+2]
        a = initial_params[i+3]
        
        # 获取并应用缩放后的范围
        if link_index in base_ranges:
            ranges = base_ranges[link_index]
            
            # 检查原始范围配置是否为0，确保只有明确设置为0的参数才被固定
            # 即使应用缩放因子后范围很小，只要原始配置不是0，也应该允许参数优化
            theta_is_fixed = abs(ranges[0]) < 1e-10  # 判断原始配置是否为0
            d_is_fixed = abs(ranges[1]) < 1e-10
            alpha_is_fixed = abs(ranges[2]) < 1e-10
            a_is_fixed = abs(ranges[3]) < 1e-10
            
            # 应用缩放因子计算实际范围
            theta_range = ranges[0] * scale_factor
            d_range = ranges[1] * scale_factor
            alpha_range = ranges[2] * scale_factor
            a_range = ranges[3] * scale_factor
            
            # 根据原始配置决定是否固定参数，而不是根据缩放后的范围
            if theta_is_fixed:
                bounds.append((theta_offset, theta_offset))  # 固定参数
            else:
                # 确保范围有效，即使很小也保持上下界不同
                min_range = 1e-6  # 最小有效范围，避免数值精度问题
                actual_range = max(theta_range, min_range)
                bounds.append((theta_offset - actual_range, theta_offset + actual_range))
                
            if d_is_fixed:
                bounds.append((d, d))  # 固定参数
            else:
                actual_range = max(d_range, min_range)
                bounds.append((d - actual_range, d + actual_range))
                
            if alpha_is_fixed:
                bounds.append((alpha, alpha))  # 固定参数
            else:
                actual_range = max(alpha_range, min_range)
                bounds.append((alpha - actual_range, alpha + actual_range))
                
            if a_is_fixed:
                bounds.append((a, a))  # 固定参数
            else:
                actual_range = max(a_range, min_range)
                bounds.append((a - actual_range, a + actual_range))
        else:
            # 如果没有指定范围，使用默认参数（固定不变）
            bounds.append((theta_offset, theta_offset))
            bounds.append((d, d))
            bounds.append((alpha, alpha))
            bounds.append((a, a))
    
    # 保留边界值有效性检查 - 只有当上下界不相等时才检查
    for i, (lower, upper) in enumerate(bounds):
        if lower != upper and lower >= upper:  # 仅在非固定参数且边界无效时调整
            print(f"警告: 参数 {i} 的边界设置无效 [{lower}, {upper}]，将调整为有效边界")
            # 确保上下边界有效
            mean_val = (lower + upper) / 2
            bounds[i] = (mean_val - 1e-5, mean_val + 1e-5)
    
    # 打印边界设置以便调试，改进输出格式
    print(f"\n{'='*60}")
    print(f"参数优化边界 (缩放因子={scale_factor:.3f})")
    print(f"{'='*60}")
    
    # 检查并显示每个参数的固定状态
    fixed_params = []
    for i, (lower, upper) in enumerate(bounds):
        if abs(upper - lower) < 1e-10:  # 使用小阈值判断是否固定
            fixed_params.append(i)
    
    # 按连杆组织固定参数信息
    link_param_status = {}
    for i in range(len(initial_params) // 4):
        link_index = i + 1
        param_names = ["theta", "d", "alpha", "a"]
        status = []
        
        for j in range(4):
            param_idx = i * 4 + j
            is_fixed = param_idx in fixed_params
            status.append(f"{param_names[j]}=" + ("固定" if is_fixed else "可优化"))
        
        link_param_status[link_index] = status
    
    # 打印每个连杆的参数状态，使用更美观的格式
    for link_idx, status in sorted(link_param_status.items()):
        print(f"\n【连杆 {link_idx}】 {', '.join(status)}")
        
        # 获取当前连杆的四个参数边界
        theta_bound = bounds[link_idx*4-4]
        d_bound = bounds[link_idx*4-3]
        alpha_bound = bounds[link_idx*4-2]
        a_bound = bounds[link_idx*4-1]
        
        # 使用表格样式格式化输出
        print(f"  {'参数':<10} {'下界':<12} {'上界':<12}")
        print(f"  {'-'*34}")
        
        # 根据参数是否固定使用不同的格式显示边界
        # theta参数
        if abs(theta_bound[1] - theta_bound[0]) < 1e-10:  # 固定参数
            print(f"  {'theta':<10} {theta_bound[0]:<12.3f} {'(固定)':<12}")
        else:
            print(f"  {'theta':<10} {theta_bound[0]:<12.3f} {theta_bound[1]:<12.3f}")
        
        # d参数
        if abs(d_bound[1] - d_bound[0]) < 1e-10:  # 固定参数
            print(f"  {'d':<10} {d_bound[0]:<12.3f} {'(固定)':<12}")
        else:
            print(f"  {'d':<10} {d_bound[0]:<12.3f} {d_bound[1]:<12.3f}")
        
        # alpha参数
        if abs(alpha_bound[1] - alpha_bound[0]) < 1e-10:  # 固定参数
            print(f"  {'alpha':<10} {alpha_bound[0]:<12.3f} {'(固定)':<12}")
        else:
            print(f"  {'alpha':<10} {alpha_bound[0]:<12.3f} {alpha_bound[1]:<12.3f}")
        
        # a参数
        if abs(a_bound[1] - a_bound[0]) < 1e-10:  # 固定参数
            print(f"  {'a':<10} {a_bound[0]:<12.3f} {'(固定)':<12}")
        else:
            print(f"  {'a':<10} {a_bound[0]:<12.3f} {a_bound[1]:<12.3f}")
    
    print(f"\n{'='*60}")
    
    return bounds


def adjust_bounds_dynamically(params, prev_rmse, curr_rmse, bounds, min_scale=0.1, adjustment_rate=0.8):
    """
    根据优化进度动态调整参数边界
    
    参数:
        params: 当前参数值
        prev_rmse: 前一次迭代的RMSE误差
        curr_rmse: 当前迭代的RMSE误差
        bounds: 当前参数边界
        min_scale: 边界缩小的最小比例
        adjustment_rate: 边界调整率
        
    返回:
        new_bounds: 调整后的边界
    """
    # 复制当前边界，避免修改原始边界
    new_bounds = bounds.copy()
    
    # 计算误差改善比例
    improvement_ratio = (prev_rmse - curr_rmse) / prev_rmse if prev_rmse > 0 else 0
    
    # 格式化改善率为百分比并输出，确保数值显示正确
    improvement_percent = improvement_ratio * 100
    
    # 如果误差有显著改善，保持当前边界
    if improvement_ratio > 0.05:  # 误差改善超过5%
        print(f"优化进展良好 (改善率: {improvement_percent:.2f}%)，保持当前搜索边界")
        return new_bounds
    
    # 如果误差改善不明显，收缩边界以精细化搜索
    scale_factor = max(min_scale, adjustment_rate)
    print(f"优化改善缓慢 (改善率: {improvement_percent:.2f}%)，收缩搜索边界，缩放因子: {scale_factor:.2f}")
    
    # 对每个参数调整边界，向当前参数值收缩
    for i, (lower, upper) in enumerate(bounds):
        # 只调整非固定参数的边界
        if abs(upper - lower) > 1e-10:  # 非固定参数
            param_value = params[i]
            # 计算新边界，向当前参数值收缩
            new_range = (upper - lower) * scale_factor / 2
            new_bounds[i] = (param_value - new_range, param_value + new_range)
            
            # 确保边界有效
            if new_bounds[i][0] >= new_bounds[i][1]:
                new_bounds[i] = (param_value - 1e-5, param_value + 1e-5)
        # 固定参数保持不变
    
    return new_bounds


def main():
    """
    主函数：执行机器人运动学参数优化的完整流程。
    
    返回:
        tuple: (优化后的DH参数, 最终RMSE误差) 或 (None, None)（如果发生错误）
    """
    # ====================== 可调参数配置 ======================
    # 数据文件配置
    # 检查可用的数据文件并提供选择
    available_files = []
    default_files = ['data.txt', 'dat_local.txt']
    
    # 检查默认文件是否存在
    for file in default_files:
        if os.path.exists(file):
            available_files.append(file)
    
    # 检查当前目录下的所有txt文件
    for file in os.listdir('.'):
        if file.endswith('.txt') and file not in available_files and file != 'optimized_dh_params.txt':
            available_files.append(file)
    
    if not available_files:
        print("错误: 未找到任何数据文件")
        return None, None
    
    # 用户选择数据文件
    if len(available_files) == 1:
        data_file = available_files[0]
        print(f"找到唯一数据文件: {data_file}")
    else:
        print("\n可用的数据文件:")
        for i, file in enumerate(available_files):
            print(f"{i+1}. {file}")
        
        try:
            choice = int(input("\n请选择要使用的数据文件 (输入序号): "))
            if 1 <= choice <= len(available_files):
                data_file = available_files[choice-1]
            else:
                print(f"无效选择，使用默认文件: {available_files[0]}")
                data_file = available_files[0]
        except ValueError:
            print(f"无效输入，使用默认文件: {available_files[0]}")
            data_file = available_files[0]
    
    result_dir = 'result'
    result_file = f'{result_dir}/optimized_dh_params.txt'
    
    # 权重配置
    position_weight = 1.0      # 位置误差权重
    quaternion_weight = 0.0    # 姿态误差权重（DE优化阶段）
    quaternion_weight_lm = 0.0 # 姿态误差权重（LM优化阶段）
    
    # 使用配置文件中的边界调整配置
    initial_scale = INITIAL_SCALE        # 初始边界缩放因子
    min_scale = MIN_SCALE                # 最小边界缩放因子
    adjustment_rate = ADJUSTMENT_RATE    # 边界调整率
    dynamic_bounds = True                # 是否使用动态边界调整
    boundary_adjustment_intervals = BOUNDARY_ADJUSTMENT_INTERVALS
    
    # 使用配置文件中的DE优化器配置
    de_config = DE_CONFIG
    # ==========================================================
    
    try:
        # 创建结果目录
        ensure_dir(result_dir)
        
        # 加载数据文件，增加更好的错误处理
        if not os.path.exists(data_file):
            print(f"错误: 找不到数据文件 '{data_file}'")
            print(f"当前工作目录: {os.getcwd()}")
            print("请确保数据文件(data.txt 或 dat_local.txt)存在于当前目录")
            return None, None
        
        print(f"使用数据文件: {data_file}")
        
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
        bounds = setup_adaptive_bounds(initial_dh_params, initial_scale, PARAM_RANGES)
        
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
                nonlocal bounds, de_history, de_params, initial_rmse, boundary_adjustment_intervals, min_scale
                
                # 计算当前迭代次数
                iteration = len(de_history['fitness']) - 1 if 'fitness' in de_history and len(de_history['fitness']) > 0 else 0
                
                # 获取当前种群的最佳适应度
                current_best = de_history['fitness'][-1] if 'fitness' in de_history and de_history['fitness'] else float('inf')
                
                # 仅在特定迭代时调整边界，其他时候不做任何修改
                if iteration > 0 and iteration % boundary_adjustment_intervals == 0:
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
                        new_bounds = adjust_bounds_dynamically(xk, prev_best, prev_best * 0.9999, bounds, min_scale)
                    else:
                        new_bounds = adjust_bounds_dynamically(xk, prev_best, current_best, bounds, min_scale)
                    
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
                    position_weight=position_weight,
                    quaternion_weight=quaternion_weight,
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