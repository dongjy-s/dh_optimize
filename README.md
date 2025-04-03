# 机器人DH参数优化程序

此程序用于优化工业机器人的DH参数，提高机器人的绝对精度。

## 功能特点

- 支持基于实测数据优化DH参数
- 支持差分进化算法 (DE) 和 Levenberg-Marquardt (LM) 优化算法
- 支持动态调整参数边界
- 支持位置和姿态四元数误差的联合优化

## 使用步骤

1. 准备数据文件（`data.txt` 或 `dat_local.txt`）：
   - 包含关节角度和对应的实测位置
   - 如需包含姿态信息，还需提供实测四元数

2. 配置参数边界：
   - 复制 `bounds_config_example.py` 为 `bounds_config.py`
   - 根据需要修改参数范围和优化设置

3. 运行优化程序：
   ```
   python main.py
   ```

4. 查看结果：
   - 优化后的DH参数保存在 `result/optimized_dh_params.txt`
   - 误差对比结果保存在 `result/error_comparison.txt`

## 配置说明

### 参数范围配置

在 `bounds_config.py` 中配置每个连杆的参数优化范围：

```python
PARAM_RANGES = {
    # 连杆索引: [theta_range, d_range, alpha_range, a_range]
    1: [0.0, 0.0, 0.0, 0.0],    # 0.0 表示固定参数，不参与优化
    2: [1.5, 2.0, 0.0, 2.0],    # 正数表示优化参数的搜索范围
    # ...
}
```

## 输出结果

优化完成后，程序将输出：
- 初始DH参数和优化后的DH参数
- 参数变化量和百分比
- 误差改进情况
- 位置精度的平均改进和最大改进
