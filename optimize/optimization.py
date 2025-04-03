import numpy as np
import random
from scipy.optimize import least_squares, minimize
from .utils import rmse

def optimize_with_lm(initial_params, joint_angles, measured_positions, error_func, bounds=None,
                    measured_quaternions=None, position_weight=1.0, quaternion_weight=0.5):
    """使用Levenberg-Marquardt算法优化DH参数，添加参数约束
    
    Args:
        initial_params: 初始DH参数
        joint_angles: 关节角度数据
        measured_positions: 测量位置数据
        error_func: 误差函数
        bounds: 参数边界，格式为[(min1, max1), (min2, max2), ...] (可选)
        measured_quaternions: 测量姿态四元数数据(可选)
        position_weight: 位置误差权重(默认1.0)
        quaternion_weight: 四元数误差权重(默认0.5)
    
    Returns:
        optimized_params: 优化后的参数
        final_rmse: 最终RMSE值
    """
    import numpy as np
    from scipy.optimize import least_squares
    
    print("\n开始LM优化...")
    
    # 定义目标函数
    def objective(params):
        # 计算常规误差
        if measured_quaternions is not None:
            errors = error_func(params, joint_angles, measured_positions, 
                               measured_quaternions=measured_quaternions,
                               position_weight=position_weight, 
                               quaternion_weight=quaternion_weight)
        else:
            errors = error_func(params, joint_angles, measured_positions)
        
        # 如果提供了边界，添加边界惩罚
        if bounds is not None:
            penalty = 0
            for i, (param, (lower, upper)) in enumerate(zip(params, bounds)):
                if param < lower or param > upper:
                    # 对超出边界的参数添加惩罚
                    penalty += abs(param - (lower if param < lower else upper)) * 10
            
            if penalty > 0:
                # 添加惩罚到误差向量
                errors = np.append(errors, np.ones(10) * penalty)
        
        return errors
    
    # 设置LM优化的边界约束
    bounds_lower = []
    bounds_upper = []
    
    if bounds is not None:
        # 使用提供的边界
        for lower, upper in bounds:
            bounds_lower.append(lower)
            bounds_upper.append(upper)
    else:
        # 使用默认边界设置
        for i in range(len(initial_params) // 4):
            idx = i * 4
            for j in range(4):
                param = initial_params[idx + j]
                
                # 特殊处理连杆2和3的d参数
                if (i == 1 or i == 2) and j == 1:
                    bounds_lower.append(-20)
                    bounds_upper.append(20)
                elif abs(param) < 1e-6:  # 接近0的参数
                    bounds_lower.append(-10)
                    bounds_upper.append(10)
                else:
                    # 允许±10%的变化
                    lower = min(param * 0.9, param * 1.1)
                    upper = max(param * 0.9, param * 1.1)
                    bounds_lower.append(lower)
                    bounds_upper.append(upper)
    
    try:
        # 使用Trust Region Reflective算法进行有约束优化
        result = least_squares(
            objective, 
            initial_params, 
            method='trf',  # Trust Region Reflective算法支持边界约束
            bounds=(bounds_lower, bounds_upper),
            ftol=1e-8, 
            xtol=1e-8, 
            gtol=1e-8, 
            max_nfev=1000, 
            verbose=1
        )
        
        # 获取优化结果
        optimized_params = result.x
        
        # 计算最终误差，使用相同的方式
        if measured_quaternions is not None:
            final_errors = error_func(optimized_params, joint_angles, measured_positions, 
                                     measured_quaternions=measured_quaternions,
                                     position_weight=position_weight, 
                                     quaternion_weight=quaternion_weight)
        else:
            final_errors = error_func(optimized_params, joint_angles, measured_positions)
            
        final_rmse = np.sqrt(np.mean(np.square(final_errors)))
        
        print(f"LM优化完成，最终RMSE: {final_rmse:.6f}\n")
        
        return optimized_params, final_rmse
    except Exception as e:
        print(f"LM优化过程中出错: {e}")
        print("返回初始参数")
        
        # 计算初始参数的RMSE作为返回值
        if measured_quaternions is not None:
            initial_errors = error_func(initial_params, joint_angles, measured_positions, 
                                       measured_quaternions=measured_quaternions,
                                       position_weight=position_weight, 
                                       quaternion_weight=quaternion_weight)
        else:
            initial_errors = error_func(initial_params, joint_angles, measured_positions)
            
        initial_rmse = np.sqrt(np.mean(np.square(initial_errors)))
        
        return initial_params, initial_rmse

# 保留这个版本的differential_evolution函数，它支持动态边界调整
def differential_evolution(func, bounds, *args, popsize=15, maxiter=100, F=0.5, CR=0.7, 
                           seed=42, callback=None, measured_quaternions=None, 
                           position_weight=1.0, quaternion_weight=0.0, history=None):
    """
    自定义的差分进化算法实现，支持动态调整参数边界
    
    参数:
        func: 目标函数
        bounds: 参数边界，形式为 [(min1, max1), (min2, max2), ...]
        *args: 传递给目标函数的额外参数
        popsize: 种群大小
        maxiter: 最大迭代次数
        F: 变异因子
        CR: 交叉概率
        seed: 随机种子，默认为42以确保结果可重复
        callback: 回调函数，接收最佳个体和收敛值
        measured_quaternions: 测量的四元数数据
        position_weight: 位置误差权重
        quaternion_weight: 四元数误差权重
        history: 历史记录字典
        
    返回:
        tuple: (最优参数, 最优适应度, 优化历史)
    """
    # 解包args参数
    joint_angles, measured_positions = args[0], args[1]
    
    # 初始化历史记录，如果未提供
    if history is None:
        history = {'fitness': [], 'best_x': []}
    
    # 设置随机种子以确保结果可重复
    if seed is not None:
        print(f"设置随机种子: {seed}")
        np.random.seed(seed)
        random.seed(seed)
    
    # 参数维度
    dimensions = len(bounds)
    
    # 确定哪些参数是固定的（上下界完全相同）
    fixed_params = []
    for i, (lower, upper) in enumerate(bounds):
        # 使用严格相等判断固定参数
        if abs(upper - lower) < 1e-10:  # 使用小阈值处理浮点误差
            fixed_params.append(i)
    
    if fixed_params:
        print(f"固定参数索引: {fixed_params}")
        
        # 按连杆组织固定参数信息，更直观地显示
        link_fixed_params = {}
        for idx in fixed_params:
            link_idx = idx // 4 + 1  # 计算连杆索引 (1-6)
            param_type = idx % 4     # 参数类型 (0=theta, 1=d, 2=alpha, 3=a)
            param_names = ["theta", "d", "alpha", "a"]
            
            if link_idx not in link_fixed_params:
                link_fixed_params[link_idx] = []
            link_fixed_params[link_idx].append(param_names[param_type])
        
        print("固定参数详情:")
        for link, params in sorted(link_fixed_params.items()):
            params_str = ", ".join(params)
            print(f"  连杆 {link}: {params_str}")
    
    # 初始化种群时确保固定参数准确设置
    population = []
    for i in range(popsize):
        individual = []
        for j in range(dimensions):
            if j in fixed_params:
                # 对于固定参数，始终使用边界值（上下界相同）
                individual.append(bounds[j][0])
            else:
                # 对于可变参数，在边界范围内随机生成
                individual.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(individual)
    
    # 计算初始种群的适应度
    fitness = []
    for ind in population:
        if measured_quaternions is not None:
            errors = func(ind, joint_angles, measured_positions, 
                          measured_quaternions=measured_quaternions,
                          position_weight=position_weight, 
                          quaternion_weight=quaternion_weight)
        else:
            errors = func(ind, joint_angles, measured_positions)
        fitness.append(rmse(errors))
    
    # 找到最佳个体
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # 记录初始最佳适应度
    if len(history['fitness']) == 0:  # 避免重复添加
        history['fitness'].append(best_fitness)
        history['best_x'].append(best_x.copy())
    
    # 开始迭代进化
    iteration_counter = 0
    for gen in range(maxiter):
        iteration_counter = gen + 1
        
        # 进化种群
        for i in range(popsize):
            # 变异: 从种群中选择3个不同的个体
            candidates = list(range(popsize))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)
            
            # 创建试验向量，确保固定参数不变
            trial = []
            for j in range(dimensions):
                if j in fixed_params:
                    # 对于固定参数，保持不变
                    trial.append(population[i][j])
                else:
                    # 变异
                    mutant = population[a][j] + F * (population[b][j] - population[c][j])
                    
                    # 边界检查
                    if mutant < bounds[j][0]:
                        mutant = bounds[j][0]
                    elif mutant > bounds[j][1]:
                        mutant = bounds[j][1]
                    
                    # 交叉
                    if random.random() < CR or j == random.randrange(dimensions):
                        trial.append(mutant)
                    else:
                        trial.append(population[i][j])
            
            # 评估试验向量
            if measured_quaternions is not None:
                trial_errors = func(trial, joint_angles, measured_positions, 
                                   measured_quaternions=measured_quaternions,
                                   position_weight=position_weight, 
                                   quaternion_weight=quaternion_weight)
            else:
                trial_errors = func(trial, joint_angles, measured_positions)
            
            trial_fitness = rmse(trial_errors)
            
            # 选择: 如果试验向量更好，则替换当前个体
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # 更新全局最优
                if trial_fitness < best_fitness:
                    best_x = trial.copy()
                    best_fitness = trial_fitness
        
        # 记录当前代的最佳适应度
        history['fitness'].append(best_fitness)
        history['best_x'].append(best_x.copy())
        
        # 计算收敛度量
        convergence = np.std(fitness) / np.mean(fitness) if np.mean(fitness) > 0 else 0
        
        # 使用回调函数，传递当前迭代次数确保回调函数知道当前进度
        if callback is not None:
            should_update, new_bounds = callback(best_x, convergence)
            # 修改处理回调返回值的逻辑
            if should_update is True:  # 仅在显式返回True时更新边界
                bounds = new_bounds
                # 检查并调整超出新边界的个体
                for i in range(popsize):
                    for j in range(dimensions):
                        if population[i][j] < bounds[j][0]:
                            population[i][j] = bounds[j][0]
                        elif population[i][j] > bounds[j][1]:
                            population[i][j] = bounds[j][1]
            
            # 仅在回调明确返回False时提前终止
            elif should_update is False:  # 显式检查False
                print(f"根据回调函数指示，提前终止DE优化，已完成{gen+1}代")
                break
        
        # 每10代打印进度，显示正确的迭代计数
        if (gen + 1) % 10 == 0:
            print(f"DE 迭代 {gen + 1}/{maxiter}, 最佳RMSE: {best_fitness:.6f}")
    
    print(f"DE优化完成，共迭代 {iteration_counter} 代，最终RMSE: {best_fitness:.6f}")
    return best_x, best_fitness, history
