from optimize.forward_kinematics import RokaeRobot
from optimize.utils import ensure_dir, print_dh_params, calculate_param_changes, setup_bounds, rmse
from optimize.data_utils import load_data, save_formatted_dh_params
from optimize.kinematics import error_function
from optimize.optimization import differential_evolution, optimize_with_lm, optimize_with_local, optimize_with_minimize
from optimize.validation import validate_optimization

def main():
    # 创建目录
    ensure_dir('result')
    
    # 加载测量数据
    joint_angles, measured_positions = load_data('data.txt')
    print(f"加载了 {len(joint_angles)} 组样本数据")
    
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
    de_optimized_params, de_fitness, de_history = differential_evolution(
        error_function, bounds, joint_angles, measured_positions, 
        popsize=50, maxiter=100, F=0.5, CR=0.9
    )
    
    print_dh_params(de_optimized_params, "DE优化")
    print(f"DE优化后RMSE: {de_fitness:.6f} mm")
    
    # 第二阶段：使用LM算法进行局部优化
    final_params, final_rmse = optimize_with_lm(
        de_optimized_params, joint_angles, measured_positions, error_function
    )

    # # 第二阶段：使用局部优化方法（例如选择 'trf' 方法）进行局部优化
    # final_params, final_rmse = optimize_with_local(
    # de_optimized_params, joint_angles, measured_positions, error_function,
    # method='trf', ftol=1e-8, xtol=1e-8
    # )
    
#     # 使用 minimize优化 方法
#     final_params, final_rmse = optimize_with_minimize(
#     de_optimized_params, joint_angles, measured_positions, error_function, bounds, method='SLSQP'
# )
    print_dh_params(final_params, "最终优化")
    print(f"最终优化后RMSE: {final_rmse:.6f} mm")
    
    # 计算参数变化
    calculate_param_changes(initial_dh_params, final_params)
    
    
    # 保存优化后的参数
    result_file = 'result/optimized_dh_params.txt'
    save_formatted_dh_params(result_file, final_params, initial_dh_params)
    print(f"优化后的参数已保存到 {result_file}")
    
    # 验证优化效果
    validate_optimization(joint_angles, measured_positions, initial_dh_params, final_params)

if __name__ == "__main__":
    main()