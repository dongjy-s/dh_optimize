import sys
import os
import random

# 将项目根目录添加到系统路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimize.forward_kinematics import RokaeRobot

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

def get_noisy_dh_params(dh_params, noise_d=1.0, noise_a=1.0, noise_alpha=0.1, noise_theta=0.1):
    """
    给 DH 参数添加噪声，每次调用 FK 时使用噪声后的参数计算
    Args:
        dh_params: 原始DH参数列表，格式为 [θ_offset, d, α, a, θ_offset, d, α, a, ...]
        noise_d: 对 d 的噪声范围（mm），默认±1mm
        noise_a: 对 a 的噪声范围（mm），默认±1mm
        noise_alpha: 对 α 的噪声范围（度），默认±0.1°
        noise_theta: 对 θ_offset 的噪声范围（度），默认±0.1°
    Returns:
        添加噪声后的 DH 参数列表
    """
    noisy_params = []
    num_links = len(dh_params) // 4
    for i in range(num_links):
        theta_offset = dh_params[i*4] + random.uniform(-noise_theta, noise_theta)
        d = dh_params[i*4+1] + random.uniform(-noise_d, noise_d)
        alpha = dh_params[i*4+2] + random.uniform(-noise_alpha, noise_alpha)
        a = dh_params[i*4+3] + random.uniform(-noise_a, noise_a)
        noisy_params.extend([theta_offset, d, alpha, a])
    return noisy_params

def main():
    robot = RokaeRobot()
    num_samples = 50

    # 保存最初的DH参数，用于后续对比（不添加噪声）
    base_dh_params = []
    for i in range(6):
        base_dh_params.extend(robot.modified_dh_params[i])

    # 创建存储位置与关节角的列表
    positions = []       # 存储原始FK计算的位置（基于原始DH参数）
    positions_real = []  # 存储每次计算时使用添加噪声后的DH参数得到的末端位置
    joint_angles = []
    
    print(f"生成 {num_samples} 组随机关节角并计算末端位置...")
    
    for sample in range(num_samples):
        # 生成随机关节角度
        q_deg = generate_random_joints()

        # 计算理想的 FK（使用原始DH参数）
        result_ideal = robot.forward_kinematics(q_deg, verbose=False)
        if not result_ideal['valid']:
            print(f"样本 {sample+1}/{num_samples}: 无效关节角度，重新生成")
            continue
        
        pos_ideal = result_ideal['position']
        positions.append(pos_ideal)
        
        # 生成添加噪声后的 DH 参数，并更新机器人的 DH 参数
        noisy_dh = get_noisy_dh_params(base_dh_params)
        for i in range(6):
            robot.modified_dh_params[i][0] = noisy_dh[i*4]       # θ_offset
            robot.modified_dh_params[i][1] = noisy_dh[i*4+1]       # d
            robot.modified_dh_params[i][2] = noisy_dh[i*4+2]       # α
            robot.modified_dh_params[i][3] = noisy_dh[i*4+3]       # a
        
        # 计算带噪声DH参数下的 FK
        result_noisy = robot.forward_kinematics(q_deg, verbose=False)
        if not result_noisy['valid']:
            # 若噪声引入导致无效，则返回一个标记大误差
            pos_noisy = [1e6, 1e6, 1e6]
        else:
            pos_noisy = result_noisy['position']
        
        # 为最终的 position 每个坐标额外添加 ±0.01 mm 的测量误差
        pos_noisy = [p + random.uniform(-0.01, 0.01) for p in pos_noisy]
            
        positions_real.append(pos_noisy)
        joint_angles.append(q_deg)
        
        print(f"样本 {sample+1}/{num_samples}:")
        print(f"  关节角度 = {[round(q, 2) for q in q_deg]}")
        print(f"  理想位置 = {[round(p, 4) for p in pos_ideal]}")
        print(f"  噪声位置 = {[round(p, 4) for p in pos_noisy]}")
    
    # 将含噪声的 FK 结果保存到文件
    with open('Pos_real.txt', 'w', encoding="utf-8") as f:
        f.write("# 关节角度(度) | 末端位置(mm)  (使用噪声DH参数计算 + 测量误差)\n")
        f.write("# q1 q2 q3 q4 q5 q6 | x y z\n")
        for i in range(len(joint_angles)):
            joints_str = " ".join([f"{angle:.4f}" for angle in joint_angles[i]])
            pos_str = " ".join([f"{coord:.4f}" for coord in positions_real[i]])
            f.write(f"{joints_str} | {pos_str}\n")
    
    # 保存理想的 FK 结果以作比较
    with open('Pos_ideal.txt', 'w', encoding="utf-8") as f:
        f.write("# 关节角度(度) | 末端位置(mm) 理想值\n")
        f.write("# q1 q2 q3 q4 q5 q6 | x y z\n")
        for i in range(len(joint_angles)):
            joints_str = " ".join([f"{angle:.4f}" for angle in joint_angles[i]])
            pos_str = " ".join([f"{coord:.4f}" for coord in positions[i]])
            f.write(f"{joints_str} | {pos_str}\n")
    
    print(f"已成功生成 {len(joint_angles)} 组数据并保存至 Pos_real.txt 文件")
    print(f"理想的 FK 结果已保存至 Pos_ideal.txt 文件")

if __name__ == "__main__":
    main()