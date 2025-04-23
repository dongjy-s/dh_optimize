# 机器人参数优化工具

## 项目简介

本项目是一个用于机器人DH参数优化的Python工具包，主要功能包括：
- 基于Levenberg-Marquardt算法的参数优化
- 差分进化算法实现
- 四元数和欧拉角转换工具
- 机器人正向运动学计算

## 安装说明

1. 确保已安装Python 3.7+版本
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备输入数据：
   - 关节角度数据
   - 测量位置数据
   - (可选)测量姿态四元数数据

2. 运行优化：
```python
from optimize.optimization import optimize_with_lm

# 初始化参数
initial_params = [...]
joint_angles = [...]
measured_positions = [...]

# 运行优化
optimized_params, final_rmse = optimize_with_lm(
    initial_params,
    joint_angles,
    measured_positions
)
```

## 示例

```python
# 示例代码展示如何使用工具
from tools.get_pos import extract_quaternion_data

# 从pos.txt文件提取四元数数据
quaternions = extract_quaternion_data("data/pos.txt", "data/quaternion_data.txt")
```

## 目录结构

```
├── data/                # 数据文件
├── optimize/            # 优化算法实现
├── tools/               # 数据处理工具
├── main.py              # 主程序入口
└── requirements.txt     # 依赖包列表
```
