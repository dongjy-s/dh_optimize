"""
优化边界设置与动态调整模块

提供 DH 参数边界设置与动态调整的各种函数，支持在优化过程中根据收敛情况自动调整边界范围。
"""


# 全局参数配置
# 连杆参数范围配置 - 用于确定哪些参数可以优化，哪些参数固定不变
PARAM_RANGES = {
    # 连杆索引: [theta_range, d_range, alpha_range, a_range]
    # 注意: 参数范围为0.0表示该参数固定不变，大于0表示可以在范围内优化
    1: [0.0, 0.0, 0.0, 0.0],  # 基座旋转 - 所有参数固定
    2: [1.0, 1.0, 0.0, 1.0],  # 肩部关节 - alpha固定，其他可优化
    3: [1.0, 1.0, 0.0, 1.0],  # 上臂 - alpha固定，其他可优化
    4: [1.0, 1.0, 0.0, 1.0],  # 肘部关节 - alpha固定，其他可优化
    5: [1.0, 1.0, 0.0, 1.0],  # 腕部关节 - alpha固定，其他可优化
    6: [0.0, 0.0, 0.0, 0.0],  # 末端关节 - 所有参数固定
}

# 优化配置
# 边界设置配置
INITIAL_SCALE = 1            # 初始边界缩放因子
MIN_SCALE = 0.1                # 最小边界缩放因子
ADJUSTMENT_RATE = 0.7          # 边界调整率
BOUNDARY_ADJUSTMENT_INTERVALS = 20  # 边界调整间隔（迭代次数）

# 误差权重
POSITION_WEIGHT = 1.3          # 位置误差权重
QUATERNION_WEIGHT_DE = 0.5     # 姿态误差权重（DE优化阶段）
QUATERNION_WEIGHT_LM = 0.0     # 姿态误差权重（LM优化阶段）

# 差分进化配置
DE_CONFIG = {
    'with_quaternions': {
        'popsize': 50, 'maxiter': 100, 'F': 0.5, 'CR': 0.9,
    },
    'position_only': {
        'popsize': 50, 'maxiter': 100, 'F': 0.5, 'CR': 0.9,
    }
}

def setup_adaptive_bounds(initial_params, scale_factor=INITIAL_SCALE):
    """
    设置DH参数的优化边界，支持动态调整边界范围
    
    参数:
        initial_params: 初始DH参数
        scale_factor: 边界范围缩放因子(0-1之间)，用于动态调整边界
    
    返回:
        bounds: 参数边界列表
    """
    bounds = []
    
    # 应用缩放因子
    for i in range(0, len(initial_params), 4):
        link_index = i // 4 + 1
        
        # 获取当前连杆参数
        theta_offset = initial_params[i]
        d = initial_params[i+1]
        alpha = initial_params[i+2]
        a = initial_params[i+3]
        
        # 获取并应用缩放后的范围
        if link_index in PARAM_RANGES:
            ranges = PARAM_RANGES[link_index]
            
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


def adjust_bounds_dynamically(params, prev_rmse, curr_rmse, bounds, min_scale=MIN_SCALE, adjustment_rate=ADJUSTMENT_RATE):
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
