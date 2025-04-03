import numpy as np

def load_data(filename):
    """加载测量数据
    
    从data.txt文件中加载关节角度、末端位置数据和姿态四元数数据，
    文件格式包含关节角度部分、位置数据部分和姿态四元数部分。
    
    Args:
        filename (str): 数据文件路径
    
    Returns:
        tuple: 包含三个numpy数组，关节角度、对应的末端位置和姿态四元数
    """
    joint_angles = []
    measured_positions = []
    measured_quaternions = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 查找两个分隔符的位置
        separator_indices = []
        for i, line in enumerate(lines):
            if line.strip() == "--------":
                separator_indices.append(i)
        
        if len(separator_indices) < 1:
            raise ValueError("无法在文件中找到分隔符 '--------'")
        
        # 提取关节角度（跳过第一行标题）
        for i in range(1, separator_indices[0]):
            line = lines[i].strip()
            if line and not line.startswith("#"):
                values = [float(val.strip()) for val in line.split(',') if val.strip()]
                if len(values) >= 6:  # 确保至少有6个值
                    joint_angles.append(values[:6])  # 只取前6个值（关节角度）
        
        # 提取位置数据（跳过位置标题行）
        for i in range(separator_indices[0] + 2, separator_indices[1] if len(separator_indices) > 1 else len(lines)):
            line = lines[i].strip()
            if line and not line.startswith("#"):
                values = [float(val.strip()) for val in line.split(',') if val.strip()]
                if len(values) >= 3:  # 确保至少有3个值
                    measured_positions.append(values[:3])  # 只取前3个值（x,y,z坐标）
        
        # 提取姿态四元数数据（如果存在）
        if len(separator_indices) > 1:
            for i in range(separator_indices[1] + 2, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith("#"):
                    values = [float(val.strip()) for val in line.split(',') if val.strip()]
                    if len(values) >= 4:  # 确保至少有4个值
                        measured_quaternions.append(values[:4])  # 只取前4个值（qx,qy,qz,qw）
    
    # 检查是否有相同数量的关节角度和位置数据
    if len(joint_angles) != len(measured_positions):
        print(f"警告: 关节角度数量({len(joint_angles)})与位置数据数量({len(measured_positions)})不匹配")
    
    # 检查是否有姿态四元数数据
    if len(measured_quaternions) > 0 and len(joint_angles) != len(measured_quaternions):
        print(f"警告: 关节角度数量({len(joint_angles)})与姿态四元数数量({len(measured_quaternions)})不匹配")

    print(f"加载了{len(joint_angles)}组关节角度数据")
    print(f"加载了{len(measured_positions)}组位置数据")
    print(f"加载了{len(measured_quaternions)}组姿态四元数数据")
    
    # 如果有姿态四元数数据，则一并返回
    if measured_quaternions:
        return np.array(joint_angles), np.array(measured_positions), np.array(measured_quaternions)
    else:
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
    file.write(f"最大优化后误差: {np.max(optimized_errors):.6f} mm\n")  # 修复了这里的双冒号
    file.write(f"平均误差改进: {(1 - np.mean(optimized_errors) / np.mean(initial_errors)) * 100:.2f}%\n")