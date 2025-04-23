import numpy as np
import math

# 处理导入路径问题：既可以作为模块导入，也可以单独运行
try:
    # 当作为包的一部分导入时使用相对导入
    from .tool_transform import create_tool_transform, rotation_matrix_to_quaternion
except ImportError:
    # 当直接运行该文件时使用绝对导入
    from tool_transform import create_tool_transform, rotation_matrix_to_quaternion

class RokaeRobot:
    """
    洛克机器人类，提供DH参数定义和运动学计算功能
    """
    def __init__(self):
        # 关节限位 [min, max] (单位:度)
        self.joint_limits = [
            [-170, 170],  # 关节1范围
            [-96, 130],   # 关节2范围
            [-195, 65],   # 关节3范围
            [-179, 170],  # 关节4范围
            [-95, 95],    # 关节5范围
            [-180, 180]   # 关节6范围
        ]
        
        # 改进DH参数: [theta_offset_i, d_i, alpha_i, a_i]
        self.modified_dh_params = [
            [0,   380,   0,   0],     # 关节 1 (i=1)
            [-90,  0,   -90,  30],    # 关节 2 (i=2)
            [0,    0,    0,   440],   # 关节 3 (i=3)
            [0,   435,  -90,  35],    # 关节 4 (i=4)
            [0,    0,    90,   0],    # 关节 5 (i=5)
            [180,  83,  -90,   0]     # 关节 6 (i=6)
        ]
        
        # 工具相对于末端法兰的位姿定义
        self.tool_position = [1.081, 1.1316, 97.2485]  # 单位: mm
        self.tool_quaternion = [0.5003, 0.5012, 0.5002, 0.4983]  # 四元数 [x, y, z, w]
        self.tool_transform = create_tool_transform(self.tool_position, self.tool_quaternion)
        
        # 设置打印精度，抑制科学计数法
        np.set_printoptions(precision=4, suppress=True)
    
    def modified_dh_matrix(self, alpha_deg, a, d, theta_deg):
        """
        根据改进DH参数计算变换矩阵
        
        参数:
            alpha_deg: alpha角度 (度)
            a: 连杆长度 (mm)
            d: 连杆偏移 (mm)
            theta_deg: theta角度 (度)
            
        返回:
            numpy.ndarray: 4x4变换矩阵
        """
        # 将角度从度转换为弧度
        alpha_rad = math.radians(alpha_deg)
        theta_rad = math.radians(theta_deg)

        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        cos_alpha = math.cos(alpha_rad)
        sin_alpha = math.sin(alpha_rad)

        # 改进DH变换矩阵公式:
        # A_i = Rot_z(theta_i) * Trans_z(d_i) * Rot_x(alpha_i) * Trans_x(a_i)
        A = np.array([
            [cos_theta, -sin_theta, 0, a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
            [0, 0, 0, 1]
        ])
        return A
    
    def check_joint_limits(self, q_deg):
        """
        检查关节角度是否在限位范围内
        
        参数:
            q_deg: 关节角度列表 (度)
            
        返回:
            tuple: (是否有效, 错误信息)
        """
        for i, q in enumerate(q_deg):
            if q < self.joint_limits[i][0] or q > self.joint_limits[i][1]:
                return False, f"关节{i+1}角度 {q}° 超出限位范围 [{self.joint_limits[i][0]}°, {self.joint_limits[i][1]}°]"
        return True, ""
    
    def forward_kinematics(self, q_deg, verbose=False, use_tool=True):
        # 检查关节角度是否在限位范围内
        is_valid, error_msg = self.check_joint_limits(q_deg)
        if not is_valid:
            if verbose:
                print(f"错误: {error_msg}")
                print("正运动学计算已中止。")
            return {
                'valid': False,
                'error_msg': error_msg,
                'transform_matrix': None,
                'position': None,
                'rotation_matrix': None
            }
        
        # 初始化总变换矩阵 T_0^6 为单位矩阵
        T_0_6 = np.identity(4)
        
        if verbose:
            print("--- 计算过程 (改进DH参数) ---")
            print(f"所有关节角度在限位范围内，继续计算...")
        
        # 循环计算每个关节的变换矩阵并累乘
        for i in range(6):
            # 从modified_dh_params列表中获取当前关节的参数
            theta_offset_i = self.modified_dh_params[i][0]
            d_i = self.modified_dh_params[i][1]
            alpha_i = self.modified_dh_params[i][2]
            a_i = self.modified_dh_params[i][3]

            # 获取当前关节的可变角度 q_i
            q_i = q_deg[i]

            # 计算实际的 theta_i = q_i + theta_offset_i
            theta_i = q_i + theta_offset_i

            # 计算当前关节的变换矩阵 A_i (T_(i-1)^i)
            A_i = self.modified_dh_matrix(alpha_i, a_i, d_i, theta_i)

            # 累积变换: T_0^i = T_0^(i-1) * A_i
            T_0_6 = np.dot(T_0_6, A_i)
        
        # 提取法兰位置和姿态
        flange_position = T_0_6[0:3, 3]
        flange_rotation = T_0_6[0:3, 0:3]
        flange_quaternion = rotation_matrix_to_quaternion(flange_rotation)
        
        # 根据use_tool标志决定是否计算工具位置
        if use_tool:
            # 计算包含工具的变换矩阵
            T_0_tool = np.dot(T_0_6, self.tool_transform)
            final_position = T_0_tool[0:3, 3]
            final_rotation = T_0_tool[0:3, 0:3]
            final_quaternion = rotation_matrix_to_quaternion(final_rotation)
            final_transform = T_0_tool
        else:
            final_position = flange_position
            final_rotation = flange_rotation
            final_quaternion = flange_quaternion
            final_transform = T_0_6
        
        if verbose:
            print(f"输入关节角度 (q1 to q6, degrees): {q_deg}")   
            print(f"末端执行器位置 (x, y, z) in mm: [{flange_position[0]:.4f}, {flange_position[1]:.4f}, {flange_position[2]:.4f}]")
            print(f"末端执行器姿态 (四元数 qx, qy, qz, qw): [{flange_quaternion[0]:.4f}, {flange_quaternion[1]:.4f}, {flange_quaternion[2]:.4f}, {flange_quaternion[3]:.4f}]")
            
            if use_tool:
                # 使用特定格式输出工具位置
                print(f"工具位置 (x, y, z) in mm: [{final_position[0]:.4f}, {final_position[1]:.4f}, {final_position[2]:.4f}]")
                print(f"工具姿态 (四元数 qx, qy, qz, qw): [{final_quaternion[0]:.4f}, {final_quaternion[1]:.4f}, {final_quaternion[2]:.4f}, {final_quaternion[3]:.4f}]")
        
        # 将位置和姿态四元数合并为一个列表
        # 修复：使用np.concatenate而不是list加法
        position_with_quaternion = np.concatenate((final_position, final_quaternion))
        
        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': final_transform,
            'position': position_with_quaternion,  # 现在包含 [x, y, z, qx, qy, qz, qw]
            'rotation_matrix': final_rotation
        }


# 示例用法
if __name__ == "__main__":
    # 创建机器人对象
    robot = RokaeRobot()
    
    # 设置关节角度
    q_deg = [0 , 0 , 0 , 0 , 0 , 0]  # [q1, q2, q3, q4, q5, q6]  
    
    # 计算正运动学并打印详细过程
    result = robot.forward_kinematics(q_deg, verbose=True, use_tool=True)

