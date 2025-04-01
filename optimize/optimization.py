import numpy as np
import random
from scipy.optimize import least_squares, minimize
from .utils import rmse

def differential_evolution(func, bounds, joint_angles, measured_positions, 
                           popsize=20, maxiter=100, F=0.8, CR=0.9):
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
    fitness = [rmse(func(ind, joint_angles, measured_positions)) for ind in population]  # 计算每个个体的RMSE
    
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
            joint_angles, measured_positions, F, CR, best_solution, best_fitness
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
                      best_solution, best_fitness):
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

def optimize_with_lm(initial_params, joint_angles, measured_positions, func):
    """使用LM算法进一步优化DH参数"""
    print("开始LM优化...")
    
    # 使用scipy的最小二乘优化
    result = least_squares(
        func,                                    # 误差函数，输入是参数向量，返回误差向量
        initial_params,                          # 初始参数估计值，通常是之前DE算法得到的结果
        args=(joint_angles, measured_positions), # 额外传递给误差函数的参数(关节角和测量位置)
        method='lm',                            # 使用Levenberg-Marquardt算法
        ftol=1e-12,                             # 函数值收敛容差，值越小，收敛要求越严格
        xtol=1e-12,                             # 参数值收敛容差，值越小，收敛要求越严格
        verbose=1                               # 打印优化过程中的信息，1表示简略信息
    )
    
    print(f"LM优化完成，最终RMSE: {rmse(result.fun):.6f}")
    return result.x, rmse(result.fun)

def optimize_with_local(initial_params, joint_angles, measured_positions, func, 
                        method='lm', ftol=1e-12, xtol=1e-12):
    """使用局部优化算法进一步优化DH参数

    参数:
        initial_params: 初始参数向量
        joint_angles: 关节角数据
        measured_positions: 测量得到的末端位置数据
        func: 误差函数
        method: 使用的 least_squares 方法，可选 'lm'、'trf'、'dogbox'
        ftol: 目标函数收敛公差
        xtol: 参数更新收敛公差

    返回:
        (优化后的参数, 最终RMSE)
    """
    print(f"开始使用 {method} 方法进行局部优化...")
    
    from scipy.optimize import least_squares  # 确保导入在函数内或在文件顶部
    
    result = least_squares(
        func,
        initial_params,
        args=(joint_angles, measured_positions),
        method=method,
        ftol=ftol,
        xtol=xtol,
        verbose=1
    )
    
    final_rmse = rmse(result.fun)
    print(f"{method} 优化完成，最终RMSE: {final_rmse:.6f}")
    return result.x, final_rmse

def optimize_with_minimize(initial_params, joint_angles, measured_positions, func, bounds, method='L-BFGS-B'):
    """
    使用 SciPy 的 minimize 函数进行局部优化
    参数:
        initial_params: 初始参数向量
        joint_angles: 关节角数据（数组或矩阵）
        measured_positions: 测量得到的末端位置数据（数组或矩阵）
        func: 误差函数，输入参数为参数向量、joint_angles、measured_positions，返回误差向量
        bounds: 参数边界，格式为 [(lower, upper), ...]
        method: 优化方法，例如 'L-BFGS-B' 或 'SLSQP'
    返回:
        (优化后的参数, 最终RMSE)
    """
    # 将误差函数包装为标量目标函数（RMSE）
    def objective(params):
        errors = func(params, joint_angles, measured_positions)
        return rmse(errors)
    
    res = minimize(objective, initial_params, method=method, bounds=bounds)
    final_params = res.x
    final_rmse = objective(final_params)
    print(f"{method} 优化完成，最终RMSE: {final_rmse:.6f}")
    return final_params, final_rmse