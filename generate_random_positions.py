import numpy as np
import random
from forward_kinematics import RokaeRobot

def generate_random_joints():
    """
    在关节限位范围内生成随机关节角度
    
    Returns:
        list: 包含6个关节角度的列表
    """
    robot = RokaeRobot()
    random_joints = []
    
    for i in range(6):
        min_val = robot.joint_limits[i][0]
        max_val = robot.joint_limits[i][1]
        random_angle = random.uniform(min_val, max_val)
        random_joints.append(random_angle)
    
    return random_joints

def add_noise(position, noise_range=0.1):
    """
    为位置添加随机噪声
    
    Args:
        position: 原始位置坐标 [x, y, z]
        noise_range: 噪声范围，默认为0.1mm
        
    Returns:
        添加噪声后的位置坐标
    """
    pos_with_noise = []
    for coord in position:
        # 在[-noise_range, noise_range]范围内添加随机噪声
        noise = random.uniform(-noise_range, noise_range)
        pos_with_noise.append(coord + noise)
    return pos_with_noise

def main():
    robot = RokaeRobot()
    num_samples = 50
    
    # 创建存储位置的列表
    positions = []      # 存储原始FK计算的位置
    positions_real = [] # 存储添加扰动后的位置
    joint_angles = []
    
    print(f"生成{num_samples}组随机关节角度并计算末端位置...")
    
    # 生成随机样本并计算FK
    for i in range(num_samples):
        # 生成随机关节角度
        q_deg = generate_random_joints()
        
        # 计算前向运动学
        result = robot.forward_kinematics(q_deg, verbose=False)
        
        if result['valid']:
            # 获取位置
            pos = result['position']
            
            # 在原始位置基础上添加随机噪声 (±0.1mm)
            pos_real = add_noise(pos)
            
            # 保存数据
            positions.append(pos)
            positions_real.append(pos_real)
            joint_angles.append(q_deg)
            
            print(f"样本 {i+1}/{num_samples}:")
            print(f"  关节角度 = {[round(q, 2) for q in q_deg]}")
            print(f"  原始位置 = {[round(p, 4) for p in pos]}")
            print(f"  带噪声位置 = {[round(p, 4) for p in pos_real]}")
        else:
            # 如果关节角度无效，重新生成
            print(f"样本 {i+1}/{num_samples}: 无效关节角度，重新生成")
            i -= 1
    
    # 将结果保存到文件
    with open('Pos_real.txt', 'w', encoding="utf-8") as f:
        # 写入标题行
        f.write("# 关节角度(度) | 末端位置(mm)含0.1mm随机误差\n")
        f.write("# q1 q2 q3 q4 q5 q6 | x y z\n")
        
        for i in range(num_samples):
            # 格式化关节角度
            joints_str = " ".join([f"{angle:.4f}" for angle in joint_angles[i]])
            # 格式化位置 - 使用带噪声的位置
            pos_str = " ".join([f"{coord:.4f}" for coord in positions_real[i]])
            # 写入行
            f.write(f"{joints_str} | {pos_str}\n")
    
    # 可选：也保存原始FK计算的位置，用于比较
    with open('Pos_ideal.txt', 'w', encoding="utf-8") as f:
        # 写入标题行
        f.write("# 关节角度(度) | 末端位置(mm)理想值\n")
        f.write("# q1 q2 q3 q4 q5 q6 | x y z\n")
        
        for i in range(num_samples):
            # 格式化关节角度
            joints_str = " ".join([f"{angle:.4f}" for angle in joint_angles[i]])
            # 格式化位置 - 使用原始FK计算的位置
            pos_str = " ".join([f"{coord:.4f}" for coord in positions[i]])
            # 写入行
            f.write(f"{joints_str} | {pos_str}\n")
    
    print(f"已成功生成{num_samples}组数据并保存至 Pos_real.txt 文件")
    print(f"原始FK计算的位置数据保存至 Pos_ideal.txt 文件")

if __name__ == "__main__":
    main()