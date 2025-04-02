import numpy as np
import random
from scipy.optimize import least_squares, minimize
from .utils import rmse

def differential_evolution(func, bounds, joint_angles, measured_positions, 
                           popsize=20, maxiter=100, F=0.8, CR=0.9,
                           measured_quaternions=None, position_weight=1.0, quaternion_weight=0.5):
    """差分进化算法
    参数:
        func: 目标函数，计算个体适应度
        bounds: 参数边界列表，每个元素为(min,max)元组
        joint_angles: 关节角度数据
        measured_positions: 测量位置数据
        popsize: 种群大小(默认20)
        maxiter: 最大迭代次数(默认100)
        F: 变异因子(默认0.8)
        CR: 交叉概率(默认0.9)
        measured_quaternions: 测量姿态四元数数据(可选)
        position_weight: 位置误差权重(默认1.0)
        quaternion_weight: 四元数误差权重(默认0.5)
    返回:
        best_solution: 最优参数解
        best_fitness: 最优适应度值
        history: 收敛历史记录
    """
    # 参数维度
    dimensions = len(bounds)  # 获取参数维度数
    
    # 初始化种群
    population = _initialize_population(popsize, dimensions, bounds)  # 随机初始化种群
    
    # 计算初始种群的适应度
    if measured_quaternions is not None:
        fitness = [rmse(func(ind, joint_angles, measured_positions, 
                            measured_quaternions=measured_quaternions,
                            position_weight=position_weight, 
                            quaternion_weight=quaternion_weight)) 
                  for ind in population]
    else:
        fitness = [rmse(func(ind, joint_angles, measured_positions)) for ind in population]
    
    # 找到最佳个体
    best_idx = np.argmin(fitness)  # 获取最优个体索引
    best_solution = population[best_idx].copy()  # 深拷贝最优个体
    best_fitness = fitness[best_idx]  # 记录最优适应度
    
    # 记录收敛历史
    history = [best_fitness]  # 初始化历史记录
    
    # 开始迭代
    for generation in range(maxiter):
        # 进化一代种群
        population, fitness, best_solution, best_fitness = _evolve_population(
            population, fitness, func, bounds, dimensions, 
            joint_angles, measured_positions, F, CR, best_solution, best_fitness,
            measured_quaternions, position_weight, quaternion_weight
        )
        
        # 记录当前代的最佳适应度
        history.append(best_fitness)  # 添加到历史记录
        
        # 每10代打印一次进度
        if (generation + 1) % 10 == 0:
            print(f"DE 迭代 {generation + 1}/{maxiter}, 最佳RMSE: {best_fitness:.6f}")
    
    return best_solution, best_fitness, history  # 返回最终结果

def _initialize_population(popsize, dimensions, bounds):
    """初始化DE种群"""
    population = []  # 创建空列表用于存储种群
    for i in range(popsize):  # 遍历种群中的每个个体
        individual = []  # 创建空列表用于存储个体参数
        for j in range(dimensions):  # 遍历每个参数维度
            # 在参数边界范围内生成随机值
            param = random.uniform(bounds[j][0], bounds[j][1])
            individual.append(param)  # 将参数添加到个体中
        population.append(individual)  # 将个体添加到种群中
    return population  # 返回初始化完成的种群

def _evolve_population(population, fitness, func, bounds, dimensions, 
                      joint_angles, measured_positions, F, CR, 
                      best_solution, best_fitness,
                      measured_quaternions=None, position_weight=1.0, quaternion_weight=0.5):
    """进化DE种群一代"""
    popsize = len(population)  # 获取当前种群大小
    
    for i in range(popsize):
        # 1. 选择阶段：从种群中选择3个不同的候选个体
        candidates = list(range(popsize))  # 创建候选索引列表
        candidates.remove(i)  # 排除当前个体自身
        a, b, c = random.sample(candidates, 3)  # 随机选择3个不同个体
        
        # 2. 变异和交叉阶段：生成试验向量
        trial = _mutation_crossover(population, i, a, b, c, dimensions, F, CR, bounds)
        
        # 3. 选择阶段：评估试验向量的适应度
        if measured_quaternions is not None:
            trial_fitness = rmse(func(trial, joint_angles, measured_positions, 
                                     measured_quaternions=measured_quaternions,
                                     position_weight=position_weight, 
                                     quaternion_weight=quaternion_weight))
        else:
            trial_fitness = rmse(func(trial, joint_angles, measured_positions))
        
        # 4. 贪婪选择：如果试验向量更好则替换当前个体
        if trial_fitness < fitness[i]:
            population[i] = trial  # 更新种群中的个体
            fitness[i] = trial_fitness  # 更新适应度值
            
            # 5. 更新全局最优解
            if trial_fitness < best_fitness:
                best_solution = trial.copy()  # 深拷贝当前最优解
                best_fitness = trial_fitness  # 更新最优适应度
    
    # 返回更新后的种群、适应度、最优解和最优适应度
    return population, fitness, best_solution, best_fitness

def _mutation_crossover(population, i, a, b, c, dimensions, F, CR, bounds):
    """DE变异和交叉操作"""
    # 变异
    mutant = [population[a][j] + F * (population[b][j] - population[c][j]) 
             for j in range(dimensions)]
    
    # 边界处理
    for j in range(dimensions):
        if mutant[j] < bounds[j][0]:
            mutant[j] = bounds[j][0]
        if mutant[j] > bounds[j][1]:
            mutant[j] = bounds[j][1]
    
    # 交叉
    trial = []
    for j in range(dimensions):
        if random.random() < CR or j == random.randrange(dimensions):
            trial.append(mutant[j])
        else:
            trial.append(population[i][j])
    
    return trial

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
