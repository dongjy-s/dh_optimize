import matplotlib.pyplot as plt

def plot_convergence(de_history, initial_rmse, final_rmse, save_path=None):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(de_history)), de_history, label='DE', marker='o', markersize=3)
    plt.axhline(y=final_rmse, color='r', linestyle='--', label=f'LM Final: {final_rmse:.6f}')
    plt.axhline(y=initial_rmse, color='g', linestyle='--', label=f'Initial: {initial_rmse:.6f}')
    plt.xlabel('Generation')
    plt.ylabel('RMSE (mm)')
    plt.title('DH Parameter Optimization Convergence')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path)
        print(f"收敛曲线已保存到 {save_path}")
    
    plt.close()

def plot_error_comparison(initial_errors, optimized_errors, save_path=None):
    """绘制误差比较"""
    plt.figure(figsize=(12, 6))
    plt.plot(initial_errors, label='Initial Parameters', marker='o', markersize=3, alpha=0.7)
    plt.plot(optimized_errors, label='Optimized Parameters', marker='x', markersize=3, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Position Error (mm)')
    plt.title('Position Error Comparison')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"误差比较图已保存到 {save_path}")
    
    plt.close()