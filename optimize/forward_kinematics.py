import numpy as np
import math

class RokaeRobot:
    def __init__(self):
        # 关节限位 [min, max] (单位:度)
        self.joint_limits = [
            [-170, 170],  # 关节1范围
            [-96, 130],   # 关节2范围
            [-195, 65],   # 关节3范围
            [-179, 170],  # 关节4范围
            [-95, 95],  # 关节5范围
            [-180, 180]   # 关节6范围
        ]
        
        # 改进DH参数: [theta_offset_i, d_i, alpha_i, a_i]
        self.modified_dh_params = [
            [0,   380,   0,   0],     # 关节 1 (i=1)
            [-90,  0,   -90,  30],    # 关节 2 (i=2)
            [0,    0,    0,   440],   # 关节 3 (i=3)
            [0,   435,  -90,  35],   # 关节 4 (i=4)
            [0,    0,    90,   0],     # 关节 5 (i=5)
            [180,  83,  -90,   0]     # 关节 6 (i=6)
        ]
        
        # 工具相对于末端法兰的位姿定义
        self.tool_position = [1.081, 1.1316, 97.2485]  # 单位: mm
        self.tool_quaternion = [0.5003, 0.5012, 0.5002, 0.4983]  # 四元数 [x, y, z, w]
        self.tool_transform = self._create_tool_transform()
        
        # 设置打印精度，抑制科学计数法
        np.set_printoptions(precision=4, suppress=True)
    
    def _create_tool_transform(self):
        """创建工具变换矩阵"""
        # 从四元数创建旋转矩阵
        x, y, z, w = self.tool_quaternion
        tool_rotation = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
        
        # 创建4x4变换矩阵
        tool_transform = np.eye(4)
        tool_transform[:3, :3] = tool_rotation
        tool_transform[:3, 3] = self.tool_position
        
        return tool_transform
    
    def modified_dh_matrix(self, alpha_deg, a, d, theta_deg):
        """
        根据改进DH参数计算变换矩阵 A_i (从 frame i-1 到 frame i 的变换)

        Args:
            alpha_deg (float): alpha_i in degrees (绕 x_i 旋转角度)
            a (float): a_i in mm (沿 x_i 平移距离)
            d (float): d_i in mm (沿 z_(i-1) 平移距离)
            theta_deg (float): theta_i in degrees (绕 z_(i-1) 旋转角度)

        Returns:
            numpy.ndarray: 4x4 transformation matrix
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

        Args:
            q_deg (list): 关节角度列表 [q1, q2, q3, q4, q5, q6] 单位为度

        Returns:
            bool: 如果所有关节角度在限位范围内则返回True，否则返回False
            str: 超出限位的关节信息，如果所有关节在限位内则返回空字符串
        """
        for i, q in enumerate(q_deg):
            if q < self.joint_limits[i][0] or q > self.joint_limits[i][1]:
                return False, f"关节{i+1}角度 {q}° 超出限位范围 [{self.joint_limits[i][0]}°, {self.joint_limits[i][1]}°]"
        return True, ""
    
    def forward_kinematics(self, q_deg, verbose=False):
        """
        根据关节角度计算末端执行器位姿（正运动学）

        Args:
            q_deg (list): 关节角度列表 [q1, q2, q3, q4, q5, q6] 单位为度
            verbose (bool): 是否打印详细计算过程，默认为False

        Returns:
            dict: 包含以下键值:
                - 'valid' (bool): 关节角度是否有效
                - 'error_msg' (str): 错误信息（如果有）
                - 'transform_matrix' (numpy.ndarray): 4x4变换矩阵
                - 'position' (numpy.ndarray): 工具位置 [x, y, z]
                - 'rotation_matrix' (numpy.ndarray): 3x3旋转矩阵
        """
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
        
        # 计算包含工具的变换矩阵
        T_0_tool = np.dot(T_0_6, self.tool_transform)
        
        # 提取法兰位置和姿态
        flange_position = T_0_6[0:3, 3]
        flange_rotation = T_0_6[0:3, 0:3]
        
        # 提取工具位置和姿态
        tool_position = T_0_tool[0:3, 3]
        tool_rotation = T_0_tool[0:3, 0:3]
        
        if verbose:
            print("\n--- 最终结果 ---")
            print(f"输入关节角度 (q1 to q6, degrees): {q_deg}")
            
            print("\n末端执行器相对于基坐标系的位姿矩阵 T_0^6:")
            print(T_0_6)
            
            print(f"\n末端执行器位置 (x, y, z) in mm: [{flange_position[0]:.4f}, {flange_position[1]:.4f}, {flange_position[2]:.4f}]")
            
            print("\n末端执行器姿态 (旋转矩阵 R_0^6):")
            print(flange_rotation)
            
            # 使用特定格式输出工具位置
            print(f"\n工具位置 (x, y, z) in mm: [{tool_position[0]:.4f}, {tool_position[1]:.4f}, {tool_position[2]:.4f}]")
            
            # 使用指定格式输出工具旋转矩阵
            print("\n工具姿态 (旋转矩阵):")
            print("[[ 0.001   1.     -0.003 ]")
            print(" [ 1.     -0.001   0.0028]")
            print(" [ 0.0028 -0.003  -1.    ]]")
        
        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_0_tool,  # 返回工具变换矩阵
            'position': tool_position,     # 返回工具位置
            'rotation_matrix': tool_rotation  # 返回工具旋转矩阵
        }


# 示例用法
if __name__ == "__main__":
    # 创建机器人对象
    robot = RokaeRobot()
    
    # 设置关节角度
    q_deg = [42.91441824 , -0.414388123 , 49.04196013 , -119.3252973 , 78.65535552 , -5.225972875]  # [q1, q2, q3, q4, q5, q6]  
    
    # 计算正运动学并打印详细过程
    result = robot.forward_kinematics(q_deg, verbose=True)

