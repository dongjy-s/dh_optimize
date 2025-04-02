import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize.forward_kinematics import RokaeRobot
import numpy as np

def format_joint_angles(angles):
    """格式化关节角度显示"""
    return ', '.join([f"{angle:.4f}" for angle in angles])

def format_position(position):
    """格式化位置显示"""
    return f"[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]"

def test_forward_kinematics():
    """测试正运动学计算"""
    print("\n============= 正运动学测试 =============")
    
    # 创建机器人对象
    robot = RokaeRobot()
    
    # 工具相对于末端法兰的位姿定义
    tool_position = [1.081, 1.1316, 97.2485]  # 单位: mm
    tool_quaternion = [0.5003, 0.5012, 0.5002, 0.4983]  # 四元数 [x, y, z, w]
    
    # 创建工具变换矩阵
    def quaternion_to_rotation_matrix(q):
        """四元数转旋转矩阵"""
        x, y, z, w = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    def rotation_matrix_to_euler_angles(R):
        """旋转矩阵转欧拉角 (ZYX顺序，单位:度)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        return np.array([z, y, x]) * 180 / np.pi  # 转换为度
    
    # 创建工具变换矩阵
    tool_rotation = quaternion_to_rotation_matrix(tool_quaternion)
    tool_transform = np.eye(4)
    tool_transform[:3, :3] = tool_rotation
    tool_transform[:3, 3] = tool_position
    
    # 测试用的关节角度列表
    test_angles = [
        [42.91441824 , -0.414388123 , 49.04196013 , -119.3252973 , 78.65535552 , -5.225972875],  # 样例1
        [26.18229564 , 47.10895029 , 20.44052241 , -143.5911443 , 87.23868486 , -6.971798826],   # 样例2
        [2.220422813 , -1.47124169 , 42.49735792 , 7.258371315 , -50.66180738 , -154.8039343], # 样例3
        [-27.15330845 , 27.1695025 , 7.237915975 , -59.0302648 , -67.29938775 , -105.5208513]   # 样例4
    ]
    
    # 计算并显示结果
    for i, angles in enumerate(test_angles):
        print(f"\n测试样例 {i+1}:")
        print(f"关节角度: {format_joint_angles(angles)}")
        
        # 计算正运动学
        result = robot.forward_kinematics(angles)
        
        if result['valid']:    
            # 计算工具位姿
            print("\n工具位姿计算:")
            
            # 获取末端法兰相对于基座的变换矩阵
            flange_transform = np.eye(4)
            flange_transform[:3, :3] = result['rotation_matrix']
            flange_transform[:3, 3] = result['position']
            
            # 计算工具相对于基座的变换矩阵
            tool_base_transform = np.dot(flange_transform, tool_transform)
            
            # 提取工具相对于基座的位置
            tool_base_position = tool_base_transform[:3, 3]
            print(f"工具位置: {format_position(tool_base_position)} mm")
            
            # 计算欧拉角
            tool_base_euler = rotation_matrix_to_euler_angles(tool_base_transform[:3, :3])
            print(f"工具姿态 (ZYX欧拉角): [{tool_base_euler[0]:.4f}, {tool_base_euler[1]:.4f}, {tool_base_euler[2]:.4f}] 度")
        else:
            print(f"计算失败: {result['error_msg']}")

def test_dh_parameters():
    """测试不同DH参数的效果"""
    print("\n============= DH参数测试 =============")
    
    # 创建机器人对象
    robot = RokaeRobot()
    
    # 显示原始DH参数
    print("原始DH参数:")
    for i, params in enumerate(robot.modified_dh_params):
        print(f"连杆 {i+1}: [theta_offset={params[0]:.2f}, d={params[1]:.2f}, alpha={params[2]:.2f}, a={params[3]:.2f}]")
    
    # 一组固定的关节角度用于比较
    test_angle = [0, 0, 0, 0, 0, 0]
    
    # 计算原始参数下的末端位置
    original_result = robot.forward_kinematics(test_angle)
    print(f"\n零位姿态下末端位置: {format_position(original_result['position'])} mm")
    
    # 添加工具位姿计算
    print("\n============= 工具位姿计算 =============")
    
    # 工具相对于末端法兰的位姿
    tool_position = [1.081, 1.1316, 97.2485]  # 单位: mm
    tool_quaternion = [0.5003, 0.5012, 0.5002, 0.4983]  # 四元数 [x, y, z, w]
    
    # 创建工具变换矩阵
    def quaternion_to_rotation_matrix(q):
        """四元数转旋转矩阵"""
        x, y, z, w = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    # 创建工具变换矩阵
    tool_rotation = quaternion_to_rotation_matrix(tool_quaternion)
    tool_transform = np.eye(4)
    tool_transform[:3, :3] = tool_rotation
    tool_transform[:3, 3] = tool_position
    
    print("工具相对于末端法兰的变换矩阵:")
    print(np.array2string(tool_transform, precision=4, suppress_small=True))
    
    # 获取末端法兰相对于基座的变换矩阵
    flange_transform = np.eye(4)
    flange_transform[:3, :3] = original_result['rotation_matrix']
    flange_transform[:3, 3] = original_result['position']
    
    print("\n末端法兰相对于基座的变换矩阵:")
    print(np.array2string(flange_transform, precision=4, suppress_small=True))
    
    # 计算工具相对于基座的变换矩阵
    tool_base_transform = np.dot(flange_transform, tool_transform)
    
    print("\n工具相对于基座的变换矩阵:")
    print(np.array2string(tool_base_transform, precision=4, suppress_small=True))
    
    # 提取工具相对于基座的位置
    tool_base_position = tool_base_transform[:3, 3]
    print(f"\n工具相对于基座的位置: {format_position(tool_base_position)} mm")
    
    # 从旋转矩阵提取欧拉角 (ZYX顺序)
    def rotation_matrix_to_euler_angles(R):
        """旋转矩阵转欧拉角 (ZYX顺序，单位:度)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        return np.array([z, y, x]) * 180 / np.pi  # 转换为度
    
    tool_base_euler = rotation_matrix_to_euler_angles(tool_base_transform[:3, :3])
    print(f"工具相对于基座的姿态 (ZYX欧拉角): [{tool_base_euler[0]:.4f}, {tool_base_euler[1]:.4f}, {tool_base_euler[2]:.4f}] 度")

if __name__ == "__main__":
    # 测试正运动学计算
    test_forward_kinematics()
    
    # 测试DH参数效果
    test_dh_parameters()
