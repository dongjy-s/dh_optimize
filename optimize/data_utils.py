import numpy as np

def load_data(filename):
    """加载测量数据"""
    joint_angles = []
    measured_positions = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        # 跳过前两行标题
        next(f)
        next(f)
        
        for line in f:
            parts = line.strip().split('|')
            joints = list(map(float, parts[0].strip().split()))
            positions = list(map(float, parts[1].strip().split()))
            
            joint_angles.append(joints)
            measured_positions.append(positions)
    
    return np.array(joint_angles), np.array(measured_positions)

def save_formatted_dh_params(filename, params, initial_params=None):
    """将优化后的DH参数保存为易读的格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 优化后的DH参数\n")
        f.write("# 格式: [theta_offset, d, alpha, a] (单位: 度/mm)\n\n")
        
        # 为每个连杆写入参数
        for i in range(6):
            _save_link_params(f, i, params, initial_params)
        
        # 添加表格形式的摘要
        _add_params_summary_table(f, params)
        
        # 保存原始格式数据
        f.write("# 原始格式数据（每行一个参数值）:\n")
        for param in params:
            f.write(f"{param:.6f}\n")

def _save_link_params(file, link_idx, params, initial_params=None):
    """保存单个连杆的参数"""
    start_idx = link_idx * 4
    param_names = ["theta_offset", "d", "alpha", "a"]
    param_values = params[start_idx:start_idx+4]
    
    file.write(f"# 连杆 {link_idx+1}\n")
    
    for i, (name, value) in enumerate(zip(param_names, param_values)):
        file.write(f"{name:<13} = {value:.6f} {'度' if i in [0, 2] else 'mm'}\n")
    
    # 如果提供了初始参数，显示变化量和百分比
    if initial_params is not None:
        _add_param_changes(file, param_names, param_values, initial_params[start_idx:start_idx+4])
    
    file.write("\n")

def _add_param_changes(file, param_names, current_values, initial_values):
    """添加参数变化信息"""
    file.write("# 变化量:\n")
    
    for i, (name, current, initial) in enumerate(zip(param_names, current_values, initial_values)):
        change = current - initial
        pct = (change / initial * 100) if initial != 0 else float('inf')
        units = '度' if i in [0, 2] else 'mm'
        
        file.write(f"# {name:<13}: {change:.6f} {units} ({pct:.2f}% 变化)\n")

def _add_params_summary_table(file, params):
    """添加参数摘要表"""
    file.write("# 参数摘要表\n")
    file.write("# -----------------------------------------------------------------------------\n")
    file.write("# 连杆 | theta_offset(度) |    d(mm)    |  alpha(度)   |    a(mm)     |\n")
    file.write("# -----------------------------------------------------------------------------\n")
    
    for i in range(6):
        start_idx = i * 4
        file.write(f"#  {i+1}   | {params[start_idx]:15.6f} | {params[start_idx+1]:12.6f} | " +
                  f"{params[start_idx+2]:13.6f} | {params[start_idx+3]:12.6f} |\n")
                  
    file.write("# -----------------------------------------------------------------------------\n\n")

def save_error_comparison(filename, initial_errors, optimized_errors):
    """保存误差比较结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 优化前后误差比较\n")
        f.write("# 格式: 样本索引, 初始误差(mm), 优化后误差(mm), 改进(%)\n\n")
        
        for i in range(len(initial_errors)):
            improvement = (1 - optimized_errors[i] / initial_errors[i]) * 100
            f.write(f"{i}, {initial_errors[i]:.6f}, {optimized_errors[i]:.6f}, {improvement:.2f}\n")
        
        # 添加统计摘要
        _add_error_summary(f, initial_errors, optimized_errors)
    
    print(f"误差比较数据已保存到 {filename}")

def _add_error_summary(file, initial_errors, optimized_errors):
    """添加误差统计摘要"""
    file.write("\n# 统计摘要\n")
    file.write(f"平均初始误差: {np.mean(initial_errors):.6f} mm\n")
    file.write(f"平均优化后误差: {np.mean(optimized_errors):.6f} mm\n")
    file.write(f"最大初始误差: {np.max(initial_errors):.6f} mm\n")
    file.write(f"最大优化后误差: {np.max(optimized_errors):.6f} mm\n")
    file.write(f"平均误差改进: {(1 - np.mean(optimized_errors) / np.mean(initial_errors)) * 100:.2f}%\n")

