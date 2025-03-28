import numpy as np

def format_dh_params(input_file, output_file):
    """将优化后的DH参数文件重新格式化为更可读的形式"""
    
    # 读取原始DH参数
    params = np.loadtxt(input_file)
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 优化后的DH参数\n")
        f.write("# 格式: [theta_offset, d, alpha, a] (单位: 度/mm)\n\n")
        
        # 为每个连杆写入参数
        for i in range(6):
            start_idx = i * 4
            theta_offset = params[start_idx]
            d = params[start_idx + 1]
            alpha = params[start_idx + 2]
            a = params[start_idx + 3]
            
            f.write(f"# 连杆 {i+1}\n")
            f.write(f"theta_offset = {theta_offset:.6f} 度\n")
            f.write(f"d           = {d:.6f} mm\n")
            f.write(f"alpha       = {alpha:.6f} 度\n")
            f.write(f"a           = {a:.6f} mm\n\n")
        
        # 也以原始格式保留数据，便于程序读取
        f.write("# 原始数据格式 (按行: theta_offset_1, d_1, alpha_1, a_1, ... theta_offset_6, d_6, alpha_6, a_6)\n")
        for param in params:
            f.write(f"{param:.6f}\n")

if __name__ == "__main__":
    input_file = "optimized_dh_params.txt"
    output_file = "optimized_dh_params_formatted.txt"
    format_dh_params(input_file, output_file)
    print(f"参数已格式化并保存到 {output_file}")