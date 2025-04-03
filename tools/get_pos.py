import os
import numpy as np
import math

def euler_to_quaternion(rx, ry, rz, degrees=True):
    """
    将欧拉角(rx, ry, rz)转换为四元数(qx, qy, qz, qw)
    
    Args:
        rx, ry, rz: 欧拉角，按照ZYX顺序（依次为绕Z轴、Y轴、X轴旋转）
        degrees: 输入是否为角度（默认True）
        
    Returns:
        list: 四元数 [qx, qy, qz, qw]
    """
    # 如果输入是角度，转换为弧度
    if degrees:
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
    
    # 计算各个角的一半的正弦和余弦值
    cx = math.cos(rx / 2)
    sx = math.sin(rx / 2)
    cy = math.cos(ry / 2)
    sy = math.sin(ry / 2)
    cz = math.cos(rz / 2)
    sz = math.sin(rz / 2)
    
    # 计算四元数(ZYX顺序)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    
    # 归一化四元数
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
    
    return [qx, qy, qz, qw]

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建它"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def extract_quaternion_data(pos_file_path, output_quat_file=None):
    """
    从pos.txt文件中提取方向数据，并直接转换为四元数格式
    
    Args:
        pos_file_path: pos.txt文件的路径
        output_quat_file: 输出四元数文件的路径
        
    Returns:
        numpy.ndarray: 四元数数组
    """
    euler_data = []  # 临时存储欧拉角 [rx, ry, rz]
    
    # 读取pos.txt文件
    with open(pos_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过标题行
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 7:  # 确保行有足够的列
                try:
                    # 根据文件格式，Nominal Rx, Ry, Rz 是第5, 6, 7列
                    rx = float(parts[4])
                    ry = float(parts[5])
                    rz = float(parts[6])
                    euler_data.append([rx, ry, rz])
                except (ValueError, IndexError):
                    continue
    
    # 将欧拉角转换为四元数
    quaternion_data = []
    for rx, ry, rz in euler_data:
        quat = euler_to_quaternion(rx, ry, rz)
        quaternion_data.append(quat)
    
    quaternion_array = np.array(quaternion_data)
    
    # 格式化为逗号分隔的字符串
    quat_formatted = []
    
    for quat in quaternion_data:
        # 四元数数据(不带空格)
        quat_row = f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}"
        quat_formatted.append(quat_row)
    
    # 保存四元数数据
    if output_quat_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_quat_file)
        ensure_dir(output_dir)
        
        with open(output_quat_file, 'w', encoding='utf-8') as f:
            f.write("// filepath: " + output_quat_file + "\n")
            f.write("\n".join(quat_formatted))
        print(f"四元数数据已保存到 {output_quat_file}")
    
    return quaternion_array

if __name__ == "__main__":
    # 设置当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置项目根目录
    project_dir = os.path.dirname(script_dir)
    
    # 创建data目录
    data_dir = os.path.join(project_dir, "data")
    ensure_dir(data_dir)
    
    # 输入和输出文件路径
    pos_file = os.path.join(project_dir, "pos.txt")
    
    # 输出文件 - 放在data目录下
    quat_file = os.path.join(data_dir, "quaternion_data.txt")
    
    try:
        if os.path.exists(pos_file):
            # 提取四元数数据并保存
            quaternions = extract_quaternion_data(
                pos_file, 
                quat_file
            )
            print(f"已成功生成 {len(quaternions)} 个四元数数据")
            
            # 打印四元数样本统计信息
            if len(quaternions) > 0:
                qx_avg = np.mean(quaternions[:,0])
                qy_avg = np.mean(quaternions[:,1])
                qz_avg = np.mean(quaternions[:,2])
                qw_avg = np.mean(quaternions[:,3])
                print(f"四元数平均值: [{qx_avg:.4f}, {qy_avg:.4f}, {qz_avg:.4f}, {qw_avg:.4f}]")
        else:
            print(f"错误: 找不到文件 {pos_file}")
            # 尝试在当前工作目录中查找pos.txt
            current_dir_pos_file = "pos.txt"
            if os.path.exists(current_dir_pos_file):
                print(f"在当前目录找到pos.txt，将使用该文件")
                # 使用当前目录的文件，但输出仍放在项目的data目录下
                extract_quaternion_data(
                    current_dir_pos_file, 
                    quat_file
                )
            else:
                print("无法找到pos.txt文件。请确保文件存在于项目目录或当前工作目录中。")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()