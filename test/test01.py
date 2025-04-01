import numpy as np
import math
from scipy.spatial.transform import Rotation

class RokaeRobot:
    def __init__(self):
        # 关节限位 [min, max] (单位:度)
        self.joint_limits = [
            [-100, 100],  # 关节1范围
            [-99, 100],   # 关节2范围
            [-200, 75],   # 关节3范围
            [-100, 100],  # 关节4范围
            [-95, 95],  # 关节5范围
            [-180, 180]   # 关节6范围
        ]

        # 改进DH参数: [theta_offset_i, d_i, alpha_i, a_i]
        # 格式: [theta_offset (角度), d (毫米), alpha (角度), a (毫米)]
        self.modified_dh_params = [
            [0,   490,   0,   0],     # 关节 1 (i=1)
            [-90,  0,   -90,  85],    # 关节 2 (i=2)
            [0,    0,    0,   640],   # 关节 3 (i=3)
            [0,   720,  -90,  205],   # 关节 4 (i=4)
            [0,    0,    90,   0],     # 关节 5 (i=5)
            [180,  75,  -90,   0]     # 关节 6 (i=6)
        ]

        # 工具坐标系相对于末端坐标系(Frame 6)的位姿
        self.tool_pose = {
            'position': np.array([-0.196, -1.3694, 92.5868]),  # 单位: mm
            'quaternion': np.array([-0.495, 0.5046, 0.5051, -0.4953])  # rx, ry, rz, w (scipy convention)
        }

        # 设置打印精度，抑制科学计数法
        np.set_printoptions(precision=4, suppress=True)

    def modified_dh_matrix(self, alpha_deg, a, d, theta_deg):
        """
        根据改进DH参数计算变换矩阵 A_i (从 frame i-1 到 frame i 的变换)
        改进DH变换顺序: A_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)

        Args:
            alpha_deg (float): alpha_i in degrees (绕新 x_i 旋转角度)
            a (float): a_i in mm (沿新 x_i 平移距离)
            d (float): d_i in mm (沿旧 z_(i-1) 平移距离)
            theta_deg (float): theta_i in degrees (绕旧 z_(i-1) 旋转角度)

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

        # 改进DH变换矩阵公式
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
        根据关节角度计算末端执行器(Frame 6)相对于基坐标系(Frame 0)的位姿（正运动学）

        Args:
            q_deg (list): 关节角度列表 [q1, q2, q3, q4, q5, q6] 单位为度
            verbose (bool): 是否打印详细计算过程，默认为False

        Returns:
            dict: 包含以下键值:
                - 'valid' (bool): 关节角度是否有效
                - 'error_msg' (str): 错误信息（如果有）
                - 'transform_matrix' (numpy.ndarray): T_0_6, 4x4变换矩阵
                - 'position' (numpy.ndarray): 末端位置 [x, y, z] (mm)
                - 'rotation_matrix' (numpy.ndarray): 3x3旋转矩阵 R_0_6
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

        # 初始化总变换矩阵 T_0_6 为单位矩阵
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

            # 计算当前关节的变换矩阵 A_i (T_(i-1)_i)
            A_i = self.modified_dh_matrix(alpha_i, a_i, d_i, theta_i)

            # 累积变换: T_0_i = T_0_(i-1) * A_i
            T_0_6 = np.dot(T_0_6, A_i)

        # 提取位置和姿态
        position = T_0_6[0:3, 3]
        rotation_matrix = T_0_6[0:3, 0:3]

        if verbose:
            print("\n--- 末端法兰(Frame 6)相对于基坐标系(Frame 0)的最终结果 ---")
            print(f"输入关节角度 (q1 to q6, degrees): {q_deg}")
            print("\n末端执行器相对于基坐标系的位姿矩阵 T_0_6:")
            print(T_0_6)
            print(f"\n末端执行器位置 (x, y, z) in mm: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            print("\n末端执行器姿态 (旋转矩阵 R_0_6):")
            print(rotation_matrix)

        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_0_6,
            'position': position,
            'rotation_matrix': rotation_matrix
        }

    def get_tool_pose(self):
        """
        获取工具坐标系(Tool)相对于末端坐标系(Frame 6)的位姿 T_6_tool

        Returns:
            dict: 包含以下键值:
                - 'position' (numpy.ndarray): 工具位置 [x, y, z] (mm)
                - 'quaternion' (numpy.ndarray): 工具姿态四元数 [rx, ry, rz, w]
                - 'rotation_matrix' (numpy.ndarray): 3x3旋转矩阵 R_6_tool
                - 'transform_matrix' (numpy.ndarray): 4x4变换矩阵 T_6_tool
        """
        # 从四元数创建旋转矩阵
        quat = self.tool_pose['quaternion']
        # scipy.spatial.transform.Rotation expects [x, y, z, w]
        rotation = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        rot_matrix = rotation.as_matrix()

        # 创建工具相对于末端的变换矩阵
        T_6_tool = np.eye(4)
        T_6_tool[0:3, 0:3] = rot_matrix
        T_6_tool[0:3, 3] = self.tool_pose['position']

        return {
            'position': self.tool_pose['position'],
            'quaternion': self.tool_pose['quaternion'],
            'rotation_matrix': rot_matrix,
            'transform_matrix': T_6_tool
        }

    def get_tool_pose_in_base(self, q_deg, verbose=False):
        """
        计算工具坐标系(Tool)相对于基坐标系(Frame 0)的位姿 T_0_tool

        Args:
            q_deg (list): 关节角度列表 [q1, q2, q3, q4, q5, q6] 单位为度
            verbose (bool): 是否打印详细计算过程，默认为False

        Returns:
            dict: 包含以下键值:
                - 'valid' (bool): 关节角度是否有效
                - 'error_msg' (str): 错误信息（如果有）
                - 'transform_matrix' (numpy.ndarray): 工具相对于基坐标系的4x4变换矩阵 T_0_tool
                - 'position' (numpy.ndarray): 工具相对于基坐标系的位置 [x, y, z] (mm)
                - 'rotation_matrix' (numpy.ndarray): 工具相对于基坐标系的3x3旋转矩阵 R_0_tool
                - 'quaternion' (numpy.ndarray): 工具相对于基坐标系的姿态四元数 [rx, ry, rz, w]
        """
        # 首先计算末端相对于基坐标系的位姿 T_0_6
        flange_result = self.forward_kinematics(q_deg, verbose=False)

        if not flange_result['valid']:
            # 如果关节角无效，直接返回错误信息
            return flange_result

        # 获取工具相对于末端的位姿 T_6_tool
        tool_pose = self.get_tool_pose()

        # 计算工具相对于基坐标系的位姿: T_0_tool = T_0_6 * T_6_tool
        T_0_6 = flange_result['transform_matrix']
        T_6_tool = tool_pose['transform_matrix']
        T_0_tool = np.dot(T_0_6, T_6_tool)

        # 提取位置和旋转
        position = T_0_tool[0:3, 3]
        rotation_matrix = T_0_tool[0:3, 0:3]

        # 计算四元数
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # Returns [x, y, z, w]

        if verbose:
            print("\n--- 工具坐标系(Tool)相对于基坐标系(Frame 0)的位姿 ---")
            print("\n工具相对于末端(Frame 6)的位姿 (T_6_tool):")
            print(f"  位置 (x, y, z) in mm: {tool_pose['position']}")
            print(f"  姿态四元数 (rx, ry, rz, w): {tool_pose['quaternion']}")
            print("\n工具相对于基坐标系的位姿矩阵 T_0_tool:")
            print(T_0_tool)
            print(f"\n工具位置相对于基坐标系 (x, y, z) in mm: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            print(f"\n工具姿态相对于基坐标系 (四元数 rx, ry, rz, w): [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
            print("\n工具姿态相对于基坐标系 (旋转矩阵 R_0_tool):")
            print(rotation_matrix)

        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_0_tool,
            'position': position,
            'rotation_matrix': rotation_matrix,
            'quaternion': quaternion
        }

    def get_tool_pose_in_tracker(self, q_deg, verbose=False):
        """
        计算工具坐标系(Tool)相对于激光跟踪仪坐标系(Tracker)的位姿 T_tracker_tool
        假设已知: 机器人基坐标系(Base/0)相对于激光跟踪仪坐标系(Tracker)的位姿 T_tracker_base

        Args:
            q_deg (list): 关节角度列表 [q1, q2, q3, q4, q5, q6] 单位为度
            verbose (bool): 是否打印详细计算过程，默认为False

        Returns:
            dict: 包含以下键值:
                - 'valid' (bool): 关节角度是否有效
                - 'error_msg' (str): 错误信息（如果有）
                - 'transform_matrix' (numpy.ndarray): 工具相对于激光跟踪仪的4x4变换矩阵 T_tracker_tool
                - 'position' (numpy.ndarray): 工具相对于激光跟踪仪的位置 [x, y, z] (mm)
                - 'rotation_matrix' (numpy.ndarray): 工具相对于激光跟踪仪的3x3旋转矩阵 R_tracker_tool
                - 'quaternion' (numpy.ndarray): 工具相对于激光跟踪仪的姿态四元数 [rx, ry, rz, w]
        """
        # 首先计算工具相对于基座(Frame 0)的位姿 T_0_tool
        tool_in_base_result = self.get_tool_pose_in_base(q_deg, verbose=False)

        if not tool_in_base_result['valid']:
            # 如果关节角无效，直接返回错误信息
            return tool_in_base_result

        # --- 定义: 机器人基坐标系(Base/0)相对于激光跟踪仪(Tracker)的位姿 ---
        # 注意：这里的含义与之前版本不同，现在表示 T_tracker_base
        base_in_tracker = {
            'position': np.array([3030.3058, 2941.3799, 141.4515]),  # 基座原点在跟踪仪坐标系下的坐标 (mm)
            'quaternion': np.array([0.0011, -0.009, 0.7192, -0.6948])  # 基座姿态在跟踪仪坐标系下的四元数 (rx, ry, rz, w)
        }

        # --- 构建: 机器人基坐标系相对于激光跟踪仪的变换矩阵 T_tracker_base ---
        # 从四元数创建旋转矩阵 R_tracker_base
        base_quat = base_in_tracker['quaternion']
        base_rotation = Rotation.from_quat([base_quat[0], base_quat[1], base_quat[2], base_quat[3]])
        base_rot_matrix = base_rotation.as_matrix()

        # 创建基座相对于激光跟踪仪的变换矩阵 T_tracker_base
        T_tracker_base = np.eye(4)
        T_tracker_base[0:3, 0:3] = base_rot_matrix
        T_tracker_base[0:3, 3] = base_in_tracker['position']

        # --- 计算: 工具相对于激光跟踪仪的位姿 T_tracker_tool ---
        # T_tracker_tool = T_tracker_base * T_base_tool (其中 T_base_tool 就是 T_0_tool)
        T_0_tool = tool_in_base_result['transform_matrix']
        T_tracker_tool = np.dot(T_tracker_base, T_0_tool) # 注意：这里直接使用 T_tracker_base，不再求逆

        # --- 提取最终结果 ---
        position = T_tracker_tool[0:3, 3]
        rotation_matrix = T_tracker_tool[0:3, 0:3]

        # 计算四元数
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # Returns [x, y, z, w]

        if verbose:
            print("\n--- 工具坐标系(Tool)相对于激光跟踪仪(Tracker)的位姿 ---")
            print("\n基座(Frame 0)相对于激光跟踪仪(Tracker)的位姿 (T_tracker_base):")
            print(f"  位置 (x, y, z) in mm: {base_in_tracker['position']}")
            print(f"  姿态四元数 (rx, ry, rz, w): {base_in_tracker['quaternion']}")
            # print("\n  变换矩阵 T_tracker_base:") # 可选打印
            # print(T_tracker_base)
            print("\n工具相对于激光跟踪仪的位姿矩阵 T_tracker_tool:")
            print(T_tracker_tool)
            print(f"\n工具位置相对于激光跟踪仪 (x, y, z) in mm: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            print(f"\n工具姿态相对于激光跟踪仪 (四元数 rx, ry, rz, w): [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
            print("\n工具姿态相对于激光跟踪仪 (旋转矩阵 R_tracker_tool):")
            print(rotation_matrix)

        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_tracker_tool,
            'position': position,
            'rotation_matrix': rotation_matrix,
            'quaternion': quaternion
        }


# 示例用法
if __name__ == "__main__":
    # 创建机器人对象
    robot = RokaeRobot()

    # 设置关节角度 (示例数据)
    q_deg = [2.6361, 29.5175, 6.1004, 34.7483, -47.3762, 35.7916] # 使用稍微简化的小数位数

    print("\n=== 末端(Frame 6)位姿计算 ===")
    # 计算正运动学 (T_0_6) 并打印详细过程
    result = robot.forward_kinematics(q_deg, verbose=True)

    print("\n\n=== 工具(Tool)相对于基座(Frame 0)的位姿计算 ===")
    # 计算工具位姿 (T_0_tool) 并打印详细过程
    tool_result = robot.get_tool_pose_in_base(q_deg, verbose=True)

    print("\n\n=== 工具(Tool)相对于激光跟踪仪(Tracker)的位姿计算 ===")
    # 计算工具相对于激光跟踪仪的位姿 (T_tracker_tool) 并打印详细过程
    # 假设已知 T_tracker_base (基座在跟踪仪坐标系下的位姿)
    tracker_result = robot.get_tool_pose_in_tracker(q_deg, verbose=True)

    # 如果计算有效，可以访问结果
    if tracker_result and tracker_result['valid']:
        print("\n--- 最终提取的工具在跟踪仪坐标系下的位姿 ---")
        print(f"位置: {tracker_result['position']}")
        print(f"姿态 (四元数 xyzw): {tracker_result['quaternion']}")