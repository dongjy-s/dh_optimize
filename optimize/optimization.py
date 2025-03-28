import numpy as np
import random
from scipy.optimize import least_squares
from .utils import rmse

def differential_evolution(func, bounds, joint_angles, measured_positions, 
                           popsize=20, maxiter=100, F=0.8, CR=0.5):
    """差分进化算法"""
    # 参数维度
    dimensions = len(bounds)
    
    # 初始化种群
    population = _initialize_population(popsize, dimensions, bounds)
    
    # 计算初始种群的适应度
    fitness = [rmse(func(ind, joint_angles, measured_positions)) for ind in population]
    
    # 找到最佳个体
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # 记录收敛历史
    history = [best_fitness]
    
    # 开始迭代
    for generation in range(maxiter):
        population, fitness, best_solution, best_fitness = _evolve_population(
            population, fitness, func, bounds, dimensions, 
            joint_angles, measured_positions, F, CR, best_solution, best_fitness
        )
        
        # 记录当前代的最佳适应度
        history.append(best_fitness)
        
        # 打印进度
        if (generation + 1) % 10 == 0:
            print(f"DE 迭代 {generation + 1}/{maxiter}, 最佳RMSE: {best_fitness:.6f}")
    
    return best_solution, best_fitness, history

def _initialize_population(popsize, dimensions, bounds):
    """初始化DE种群"""
    population = []
    for i in range(popsize):
        individual = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensions)]
        population.append(individual)
    return population

def _evolve_population(population, fitness, func, bounds, dimensions, 
                      joint_angles, measured_positions, F, CR, 
                      best_solution, best_fitness):
    """进化DE种群一代"""
    popsize = len(population)
    
    for i in range(popsize):
        # 选择三个不同的个体，且都不是当前个体
        candidates = list(range(popsize))
        candidates.remove(i)
        a, b, c = random.sample(candidates, 3)
        
        # 变异和交叉
        trial = _mutation_crossover(population, i, a, b, c, dimensions, F, CR, bounds)
        
        # 选择
        trial_fitness = rmse(func(trial, joint_angles, measured_positions))
        if trial_fitness < fitness[i]:
            population[i] = trial
            fitness[i] = trial_fitness
            
            # 更新全局最优
            if trial_fitness < best_fitness:
                best_solution = trial.copy()
                best_fitness = trial_fitness
    
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
        func,
        initial_params,
        args=(joint_angles, measured_positions),
        method='lm',
        ftol=1e-8,
        xtol=1e-8,
        verbose=1
    )
    
    print(f"LM优化完成，最终RMSE: {rmse(result.fun):.6f}")
    return result.x, rmse(result.fun)