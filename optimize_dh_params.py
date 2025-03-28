import numpy as np
import pandas as pd
import random
import os
from scipy.optimize import least_squares
from forward_kinematics import RokaeRobot
import matplotlib.pyplot as plt

# 创建目录
def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# 初始化目录
ensure_dir('graph')  # 图像保存目录
ensure_dir('result') # 结果保存目录

def save_formatted_dh_params(filename, params, initial_params=None):
    """
    将优化后的DH参数保存为易读的格式
    
    Args:
        filename: 输出文件名
        params: DH参数列表 [theta_offset_1, d_1, alpha_1, a_1, ..., theta_offset_6, d_6, alpha_6, a_6]
        initial_params: 可选，初始DH参数，用于对比显示
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 优化后的DH参数\n")
        f.write("# 格式: [theta_offset, d, alpha, a] (单位: 度/mm)\n\n")
        
        # 为每个连杆写入参数
        for i in range(6):
            start_idx = i * 4
            theta_offset = params[start_idx]
            d = params[start_idx + 1]
            alpha = params[start_idx + 2]
            a = params[start_idx + 3]
            
            f.write(f"# 连杆 {i+1}\n")
            f.write(f"theta_offset = {theta_offset:.6f} 度\n")
            f.write(f"d           = {d:.6f} mm\n")
            f.write(f"alpha       = {alpha:.6f} 度\n")
            f.write(f"a           = {a:.6f} mm\n")
            
            # 如果提供了初始参数，显示变化量和百分比
            if initial_params is not None:
                init_theta = initial_params[start_idx]
                init_d = initial_params[start_idx + 1]
                init_alpha = initial_params[start_idx + 2]
                init_a = initial_params[start_idx + 3]
                
                # 计算变化
                theta_change = theta_offset - init_theta
                d_change = d - init_d
                alpha_change = alpha - init_alpha
                a_change = a - init_a
                
                # 计算百分比变化
                theta_pct = (theta_change / init_theta * 100) if init_theta != 0 else float('inf')
                d_pct = (d_change / init_d * 100) if init_d != 0 else float('inf')
                alpha_pct = (alpha_change / init_alpha * 100) if init_alpha != 0 else float('inf')
                a_pct = (a_change / init_a * 100) if init_a != 0 else float('inf')
                
                # 写入变化量
                f.write(f"# 变化量:\n")
                f.write(f"# theta_offset: {theta_change:.6f} 度 ({theta_pct:.2f}% 变化)\n")
                f.write(f"# d           : {d_change:.6f} mm ({d_pct:.2f}% 变化)\n") 
                f.write(f"# alpha       : {alpha_change:.6f} 度 ({alpha_pct:.2f}% 变化)\n")
                f.write(f"# a           : {a_change:.6f} mm ({a_pct:.2f}% 变化)\n")
            
            f.write("\n")
        
        # 添加表格形式的摘要
        f.write("# 参数摘要表\n")
        f.write("# -----------------------------------------------------------------------------\n")
        f.write("# 连杆 | theta_offset(度) |    d(mm)    |  alpha(度)   |    a(mm)     |\n")
        f.write("# -----------------------------------------------------------------------------\n")
        for i in range(6):
            start_idx = i * 4
            f.write(f"#  {i+1}   | {params[start_idx]:15.6f} | {params[start_idx+1]:12.6f} | {params[start_idx+2]:13.6f} | {params[start_idx+3]:12.6f} |\n")
        f.write("# -----------------------------------------------------------------------------\n\n")
        
        # 最后也保存为原始格式，便于程序读取
        f.write("# 原始格式数据（每行一个参数值）:\n")
        for param in params:
            f.write(f"{param:.6f}\n")

def load_data(filename):
    """
    加载测量数据
    
    Returns:
        joint_angles: 关节角度数组 [n_samples, 6]
        measured_positions: 测量得到的末端位置 [n_samples, 3]
    """
    data = []
    joint_angles = []
    measured_positions = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        # 跳过前两行标题
        next(f)
        next(f)
        
        for line in f:
            # 解析每行数据
            parts = line.strip().split('|')
            joints = list(map(float, parts[0].strip().split()))
            positions = list(map(float, parts[1].strip().split()))
            
            joint_angles.append(joints)
            measured_positions.append(positions)
    
    return np.array(joint_angles), np.array(measured_positions)

def forward_kinematics_with_params(joint_angles, dh_params):
    """
    使用指定的DH参数计算前向运动学
    
    Args:
        joint_angles: 关节角度 [6,]
        dh_params: 优化的DH参数，格式：[theta_offset_1, d_1, alpha_1, a_1, ..., theta_offset_6, d_6, alpha_6, a_6]
    
    Returns:
        末端位置 [3,]
    """
    # 创建修改过DH参数的机器人对象
    robot = RokaeRobot()
    
    # 更新DH参数
    for i in range(6):
        robot.modified_dh_params[i][0] = dh_params[i*4]      # theta_offset
        robot.modified_dh_params[i][1] = dh_params[i*4+1]    # d
        robot.modified_dh_params[i][2] = dh_params[i*4+2]    # alpha
        robot.modified_dh_params[i][3] = dh_params[i*4+3]    # a
    
    # 计算前向运动学
    result = robot.forward_kinematics(joint_angles, verbose=False)
    
    if result['valid']:
        return result['position']
    else:
        # 如果关节角度无效，返回一个大误差
        return np.array([1e6, 1e6, 1e6])
    
def error_function(dh_params, joint_angles, measured_positions):
    """
    计算预测位置与测量位置之间的误差
    
    Args:
        dh_params: 待优化的DH参数
        joint_angles: 所有样本的关节角度 [n_samples, 6]
        measured_positions: 测量得到的末端位置 [n_samples, 3]
    
    Returns:
        所有样本点的误差向量 [n_samples * 3,]
    """
    n_samples = joint_angles.shape[0]
    errors = np.zeros(n_samples * 3)
    
    for i in range(n_samples):
        # 使用当前DH参数计算预测位置
        predicted_pos = forward_kinematics_with_params(joint_angles[i], dh_params)
        
        # 计算误差
        errors[i*3:(i+1)*3] = predicted_pos - measured_positions[i]
    
    return errors

def rmse(errors):
    """计算均方根误差"""
    return np.sqrt(np.mean(np.square(errors)))

def differential_evolution(func, bounds, joint_angles, measured_positions, 
                           popsize=20, maxiter=100, F=0.8, CR=0.5):
    """
    差分进化算法
    
    Args:
        func: 目标函数
        bounds: 参数边界 [(min1, max1), (min2, max2), ...]
        joint_angles: 关节角度数据
        measured_positions: 测量位置数据
        popsize: 种群大小
        maxiter: 最大迭代次数
        F: 变异因子
        CR: 交叉概率
    
    Returns:
        最优参数和最优适应度值
    """
    # 参数维度
    dimensions = len(bounds)
    
    # 初始化种群
    population = []
    for i in range(popsize):
        individual = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensions)]
        population.append(individual)
    
    # 计算初始种群的适应度
    fitness = [rmse(func(ind, joint_angles, measured_positions)) for ind in population]
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # 记录收敛历史
    history = [best_fitness]
    
    # 开始迭代
    for generation in range(maxiter):
        for i in range(popsize):
            # 选择三个不同的个体，且都不是当前个体
            candidates = list(range(popsize))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)
            
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
            
            # 选择
            trial_fitness = rmse(func(trial, joint_angles, measured_positions))
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # 更新全局最优
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
        
        # 记录当前代的最佳适应度
        history.append(best_fitness)
        
        # 打印进度
        if (generation + 1) % 10 == 0:
            print(f"DE 迭代 {generation + 1}/{maxiter}, 最佳RMSE: {best_fitness:.6f}")
    
    return best_solution, best_fitness, history

def optimize_with_lm(initial_params, joint_angles, measured_positions):
    """
    使用LM算法进一步优化DH参数
    
    Args:
        initial_params: 初始参数（来自DE算法）
        joint_angles: 关节角度数据
        measured_positions: 测量位置数据
    
    Returns:
        优化后的参数
    """
    print("开始LM优化...")
    
    # 使用scipy的最小二乘优化
    result = least_squares(
        error_function,
        initial_params,
        args=(joint_angles, measured_positions),
        method='lm',
        ftol=1e-8,
        xtol=1e-8,
        verbose=1
    )
    
    print(f"LM优化完成，最终RMSE: {rmse(result.fun):.6f}")
    return result.x, rmse(result.fun)

def main():
    # 加载测量数据
    joint_angles, measured_positions = load_data('Pos_real.txt')
    print(f"加载了 {len(joint_angles)} 组样本数据")
    
    # 获取初始DH参数
    robot = RokaeRobot()
    initial_dh_params = []
    for i in range(6):
        initial_dh_params.extend(robot.modified_dh_params[i])
    
    print("初始DH参数:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={initial_dh_params[i*4]:.2f}, d={initial_dh_params[i*4+1]:.2f}, "
              f"alpha={initial_dh_params[i*4+2]:.2f}, a={initial_dh_params[i*4+3]:.2f}")
    
    # 计算初始误差
    initial_errors = error_function(initial_dh_params, joint_angles, measured_positions)
    initial_rmse = rmse(initial_errors)
    print(f"初始RMSE: {initial_rmse:.6f} mm")
    
    # 设置参数边界 (允许参数在初始值附近±10%变化)
    bounds = []
    for param in initial_dh_params:
        # 角度参数(theta_offset和alpha)的边界范围大一些
        if param == 0:
            bounds.append((-5, 5))
        else:
            # 角度参数
            if abs(param) >= 90:
                bounds.append((param * 0.9, param * 1.1))
            # 长度参数
            else:
                bounds.append((param * 0.9, param * 1.1))
    
    # 第一阶段：使用DE算法进行全局搜索
    print("\n开始DE全局优化...")
    de_optimized_params, de_fitness, de_history = differential_evolution(
        error_function, bounds, joint_angles, measured_positions, 
        popsize=30, maxiter=100, F=0.8, CR=0.7
    )
    
    print("\nDE优化结果:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={de_optimized_params[i*4]:.2f}, d={de_optimized_params[i*4+1]:.2f}, "
              f"alpha={de_optimized_params[i*4+2]:.2f}, a={de_optimized_params[i*4+3]:.2f}")
    print(f"DE优化后RMSE: {de_fitness:.6f} mm")
    
    # 第二阶段：使用LM算法进行局部优化
    final_params, final_rmse = optimize_with_lm(de_optimized_params, joint_angles, measured_positions)
    
    print("\n最终优化结果:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={final_params[i*4]:.2f}, d={final_params[i*4+1]:.2f}, "
              f"alpha={final_params[i*4+2]:.2f}, a={final_params[i*4+3]:.2f}")
    print(f"最终优化后RMSE: {final_rmse:.6f} mm")
    
    # 计算参数变化
    param_changes = []
    for i in range(len(initial_dh_params)):
        change = ((final_params[i] - initial_dh_params[i]) / initial_dh_params[i] * 100) if initial_dh_params[i] != 0 else final_params[i]
        param_changes.append(change)
    
    print("\n参数变化百分比:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={param_changes[i*4]:.2f}%, d={param_changes[i*4+1]:.2f}%, "
              f"alpha={param_changes[i*4+2]:.2f}%, a={param_changes[i*4+3]:.2f}%")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(de_history)), de_history, label='DE', marker='o', markersize=3)
    plt.axhline(y=final_rmse, color='r', linestyle='--', label=f'LM Final: {final_rmse:.6f}')
    plt.axhline(y=initial_rmse, color='g', linestyle='--', label=f'Initial: {initial_rmse:.6f}')
    plt.xlabel('Generation')
    plt.ylabel('RMSE (mm)')
    plt.title('DH Parameter Optimization Convergence')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('graph/optimization_convergence.png')
    plt.show()
    
    # 保存优化后的参数 - 使用格式化的方式
    result_file = 'result/optimized_dh_params.txt'
    save_formatted_dh_params(result_file, final_params, initial_dh_params)
    print(f"优化后的参数已保存到 {result_file}")
    
    # 验证优化效果
    validate_optimization(joint_angles, measured_positions, initial_dh_params, final_params)

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
    
    # 绘制误差比较
    plt.figure(figsize=(12, 6))
    plt.plot(initial_errors, label='Initial Parameters', marker='o', markersize=3, alpha=0.7)
    plt.plot(optimized_errors, label='Optimized Parameters', marker='x', markersize=3, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Position Error (mm)')
    plt.title('Position Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph/error_comparison.png')
    plt.show()
    
    # 保存误差数据到结果文件
    error_file = 'result/error_comparison.txt'
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("# 优化前后误差比较\n")
        f.write("# 格式: 样本索引, 初始误差(mm), 优化后误差(mm), 改进(%)\n\n")
        
        for i in range(len(initial_errors)):
            improvement = (1 - optimized_errors[i] / initial_errors[i]) * 100
            f.write(f"{i}, {initial_errors[i]:.6f}, {optimized_errors[i]:.6f}, {improvement:.2f}\n")
        
        # 添加统计摘要
        f.write("\n# 统计摘要\n")
        f.write(f"平均初始误差: {np.mean(initial_errors):.6f} mm\n")
        f.write(f"平均优化后误差: {np.mean(optimized_errors):.6f} mm\n")
        f.write(f"最大初始误差: {np.max(initial_errors):.6f} mm\n")
        f.write(f"最大优化后误差: {np.max(optimized_errors):.6f} mm\n")
        f.write(f"平均误差改进: {(1 - np.mean(optimized_errors) / np.mean(initial_errors)) * 100:.2f}%\n")
    
    print(f"误差比较数据已保存到 {error_file}")

if __name__ == "__main__":
    main()