import os
import numpy as np

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def print_dh_params(params, prefix=""):
    """打印DH参数"""
    print(f"\n{prefix}DH参数:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={params[i*4]:.2f}, d={params[i*4+1]:.2f}, "
              f"alpha={params[i*4+2]:.2f}, a={params[i*4+3]:.2f}")

def calculate_param_changes(initial_params, final_params):
    """计算参数变化百分比"""
    param_changes = []
    for i in range(len(initial_params)):
        change = ((final_params[i] - initial_params[i]) / initial_params[i] * 100) if initial_params[i] != 0 else final_params[i]
        param_changes.append(change)
    
    print("\n参数变化百分比:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={param_changes[i*4]:.2f}%, d={param_changes[i*4+1]:.2f}%, "
              f"alpha={param_changes[i*4+2]:.2f}%, a={param_changes[i*4+3]:.2f}%")


# def setup_bounds(initial_params):
#     bounds = []
#     for param in initial_params:
#         if param == 0:
#             bounds.append((-10, 10))
#         else:
#             # 计算放宽边界时先取小值和大值
#             lower = min(param * 0.9, param * 1.1)
#             upper = max(param * 0.9, param * 1.1)
#             bounds.append((lower, upper))
#     return bounds

def setup_bounds(initial_params):
    """
    为 DH 参数设置优化边界，针对不同类型参数单独设置
    
    参数:
        initial_params: 初始 DH 参数列表
    
    返回:
        bounds: 参数边界列表
    """
    bounds = []
    
    # 遍历每个连杆的参数
    for i in range(len(initial_params) // 4):
        # 每个连杆有4个参数: theta_offset, d, alpha, a
        # 获取当前连杆的参数索引
        idx = i * 4
        
        # 1. theta_offset (角度偏移): 通常变化较小
        param = initial_params[idx]
        if param == 0:
            bounds.append((-5, 5))  # 角度偏移为0时的范围
        else:
            # 角度偏移允许变化较小，例如±5%
            lower = min(param * 0.95, param * 1.05)
            upper = max(param * 0.95, param * 1.05)
            bounds.append((lower, upper))
        
        # 2. d (连杆偏移): 可能需要更大范围
        param = initial_params[idx + 1]
        if param == 0:
            bounds.append((-15, 15))  # d为0时的范围
        else:
            # d允许变化稍大，例如±10%
            lower = min(param * 0.9, param * 1.1)
            upper = max(param * 0.9, param * 1.1)
            bounds.append((lower, upper))
        
        # 3. alpha (扭转角): 变化应该很小
        param = initial_params[idx + 2]
        if param == 0:
            bounds.append((-3, 3))  # alpha为0时的范围
        elif abs(abs(param) - 90) < 1e-6:  # 接近90度或-90度
            # 90度的扭转角通常很精确，变化应该很小
            lower = min(param * 0.98, param * 1.02)
            upper = max(param * 0.98, param * 1.02)
            bounds.append((lower, upper))
        else:
            # 其他扭转角
            lower = min(param * 0.95, param * 1.05)
            upper = max(param * 0.95, param * 1.05)
            bounds.append((lower, upper))
        
        # 4. a (连杆长度): 变化范围适中
        param = initial_params[idx + 3]
        if param == 0:
            bounds.append((-8, 8))  # a为0时的范围
        else:
            # 连杆长度允许适当变化
            lower = min(param * 0.93, param * 1.07)
            upper = max(param * 0.93, param * 1.07)
            bounds.append((lower, upper))
    
    return bounds

def rmse(errors):
    """计算均方根误差"""
    return np.sqrt(np.mean(np.square(errors)))