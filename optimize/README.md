# 机器人运动学参数优化模块

本模块包含了用于工业机器人DH参数优化的核心算法，支持多种优化策略和参数配置。

## 模块结构

- `forward_kinematics.py`: 机器人正向运动学计算
- `kinematics.py`: 误差函数和运动学辅助功能
- `optimization.py`: 优化算法实现 (DE和LM)
- `data_utils.py`: 数据加载和处理功能
- `utils.py`: 通用工具函数
- `validation.py`: 优化结果验证工具
- `tool_transform.py`: 工具坐标变换处理

## 关键函数

### 差分进化算法 (DE)

```python
differential_evolution(func, bounds, *args, popsize=15, maxiter=100, F=0.5, CR=0.7, 
                       seed=None, callback=None, measured_quaternions=None, 
                       position_weight=1.0, quaternion_weight=0.0, history=None)
```

支持动态边界调整的差分进化算法，通过回调函数在优化过程中调整参数边界。

### Levenberg-Marquardt优化

```python
optimize_with_lm(initial_params, joint_angles, measured_positions, error_func, bounds=None,
                measured_quaternions=None, position_weight=1.0, quaternion_weight=0.5)
```

使用LM算法对预优化的参数进行精细优化，支持参数约束。

## 误差计算

误差函数支持仅位置误差或同时包含位置和姿态误差的优化：

```python
error_function(params, joint_angles, measured_positions, 
               measured_quaternions=None, position_weight=1.0, quaternion_weight=0.0)
```

## 参数配置

优化算法支持灵活的参数配置，可通过 bounds_config.py 文件设置：
- 参数搜索范围
- 优化算法参数
- 动态边界调整策略
