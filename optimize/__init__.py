"""
机器人DH参数优化包

这个包包含了用于优化机器人DH参数的各种工具和算法。
"""

# 可以在这里导出常用的函数，使其可以直接从包中导入
from .kinematics import error_function, forward_kinematics_with_params
from .data_utils import load_data, save_formatted_dh_params
from .optimization import differential_evolution, optimize_with_lm
from .validation import validate_optimization
from .tool_transform import create_tool_transform
from .boundaries import setup_adaptive_bounds, adjust_bounds_dynamically, DEFAULT_PARAM_RANGES, DE_CONFIG, INITIAL_SCALE, MIN_SCALE, ADJUSTMENT_RATE, BOUNDARY_ADJUSTMENT_INTERVALS

# 包的版本号
__version__ = '0.1.0'