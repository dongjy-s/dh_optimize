import os
import sys
import traceback
from optimize.forward_kinematics import RokaeRobot
from optimize.utils import ensure_dir, print_dh_params, calculate_param_changes, setup_bounds, rmse
from optimize.data_utils import load_data, save_formatted_dh_params
from optimize.kinematics import error_function
from optimize.optimization import differential_evolution, optimize_with_lm
from optimize.validation import validate_optimization

def main():
    try:
        # 创建目录
        ensure_dir('result')
        
        # 数据文件路径，使用相对路径并检查文件是否存在
        data_file = 'data.txt'
        if not os.path.exists(data_file):
            print(f"错误: 找不到数据文件 '{data_file}'")
            print(f"当前工作目录: {os.getcwd()}")
            print("请确保从正确的目录运行程序或提供正确的文件路径")
            return
        
        # 加载测量数据
        try:
            joint_angles, measured_positions = load_data(data_file)
            print(f"加载了 {len(joint_angles)} 组样本数据")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            traceback.print_exc()
            return
        
        # 获取初始DH参数
        robot = RokaeRobot()
        initial_dh_params = []
        for i in range(6):
            initial_dh_params.extend(robot.modified_dh_params[i])
        
        print_dh_params(initial_dh_params, "初始")
        
        # 计算初始误差
        initial_errors = error_function(initial_dh_params, joint_angles, measured_positions)
        initial_rmse = rmse(initial_errors)
        print(f"初始RMSE: {initial_rmse:.6f} mm")
        
        # 设置参数边界
        bounds = setup_bounds(initial_dh_params)
        
        # 第一阶段：使用DE算法进行全局搜索
        print("\n开始DE全局优化...")
        # 减小种群大小以提高效率，但仍保持有效性
        de_optimized_params, de_fitness, de_history = differential_evolution(
            error_function, bounds, joint_angles, measured_positions, 
            popsize=30, maxiter=100, F=0.5, CR=0.9
        )
        
        print_dh_params(de_optimized_params, "DE优化")
        print(f"DE优化后RMSE: {de_fitness:.6f} mm")
        
        # 检查DE优化是否有效，避免无效结果传入LM算法
        if de_fitness >= initial_rmse:
            print("警告: DE优化没有改善结果，将使用初始参数进行LM优化")
            de_optimized_params = initial_dh_params
        
        # 第二阶段：使用LM算法进行局部优化
        try:
            final_params, final_rmse = optimize_with_lm(
                de_optimized_params, joint_angles, measured_positions, error_function, bounds
            )
        except Exception as e:
            print(f"LM优化时出错: {e}")
            traceback.print_exc()
            print("将使用DE优化结果作为最终结果")
            final_params, final_rmse = de_optimized_params, de_fitness

        print_dh_params(final_params, "最终优化")
        print(f"最终优化后RMSE: {final_rmse:.6f} mm")
        
        # 计算参数变化
        calculate_param_changes(initial_dh_params, final_params)
        
        # 验证优化效果
        validate_optimization(joint_angles, measured_positions, initial_dh_params, final_params)

        # 保存优化后的参数
        result_file = 'result/optimized_dh_params.txt'
        save_formatted_dh_params(result_file, final_params, initial_dh_params)
        print(f"优化后的参数已保存到 {result_file}")
        
        return final_params, final_rmse
    
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()