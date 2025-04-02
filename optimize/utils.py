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


def setup_bounds(initial_params):
    """为 DH 参数设置优化边界，限制角度变化不超过1度，长度变化不超过1mm"""
    bounds = []
    
    # 遍历每个连杆的参数
    for i in range(len(initial_params) // 4):
        # 每个连杆有4个参数: theta_offset, d, alpha, a
        idx = i * 4
        
        # 1. theta_offset (角度偏移) - 限制变化±1度
        param = initial_params[idx]
        if abs(param) < 1e-6:  # 对接近0的值特殊处理
            bounds.append((-1.0, 1.0))
        else:
            bounds.append((param - 1.0, param + 1.0))
        
        # 2. d (连杆偏移) - 限制变化±1mm，对0值特殊处理
        param = initial_params[idx + 1]
        if abs(param) < 1e-6:  # 如果d值接近0，严格限制其变化范围
            bounds.append((-1.0, 1.0))  # 绝对变化不超过1mm
        else:
            bounds.append((param - 1.0, param + 1.0))
        
        # 3. alpha (扭转角) - 限制变化±1度
        param = initial_params[idx + 2]
        if abs(param) < 1e-6:  # 对接近0的值特殊处理
            bounds.append((-1.0, 1.0))
        else:
            bounds.append((param - 1.0, param + 1.0))
        
        # 4. a (连杆长度) - 限制变化±1mm
        param = initial_params[idx + 3]
        if abs(param) < 1e-6:  # 对接近0的值特殊处理
            bounds.append((-1.0, 1.0))
        else:
            bounds.append((param - 1.0, param + 1.0))
    
    # 打印边界设置以便调试
    print("\n参数优化边界:")
    for i in range(6):
        print(f"Link {i+1}: theta_offset={bounds[i*4]}, d={bounds[i*4+1]}, "
              f"alpha={bounds[i*4+2]}, a={bounds[i*4+3]}")
    
    return bounds


def rmse(errors):
    """计算均方根误差"""
    return np.sqrt(np.mean(np.square(errors)))