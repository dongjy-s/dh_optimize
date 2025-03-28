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

def setup_bounds(initial_dh_params):
    """设置参数搜索范围"""
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
    return bounds

def rmse(errors):
    """计算均方根误差"""
    return np.sqrt(np.mean(np.square(errors)))