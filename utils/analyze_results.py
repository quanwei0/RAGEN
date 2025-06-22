import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义算法颜色映射（包含新的turn+token level算法的两种类型）
ALGORITHM_COLORS = {
    # 只保留需要的算法类型
    'mask_True_MTGAE_False_V1': '#D32F2F',    # 深红色 - V1版本改为False
    'mask_True_MTGAE_True_V2': '#FF8C00',     # 橙色 - 与深红色形成对比
    # turn+token level算法的两种类型
    'turn+token_level_gae': '#9932CC',        # 深紫色
    'turn+token_level_avg': '#228B22',        # 森林绿色
    # 保持兼容性
    'turn+token_level': '#9932CC'             # 深紫色 - 兼容旧版本
}

def smooth_curve(y, kernel_size=20):
    """Apply smoothing using kernel of length kernel_size with proper boundary handling"""
    if len(y) < kernel_size:
        # If data is shorter than kernel, just return original data
        return y
    
    # Use pandas rolling mean which handles boundaries better
    df = pd.Series(y)
    # Use center=True for symmetric window, min_periods=1 to avoid NaN at boundaries
    smoothed = df.rolling(window=kernel_size, center=True, min_periods=1).mean()
    return smoothed.values

def extract_algorithm_type(row):
    """Extract algorithm type from DataFrame row"""
    # 如果已经有algorithm_type列，直接使用
    if 'algorithm_type' in row and pd.notna(row['algorithm_type']):
        algo_type = row['algorithm_type']
        version = row['version']
        
        # 对于turn+token level算法，直接返回（包括gae和avg类型）
        if 'turn+token_level' in algo_type:
            return version  # version字段已经包含完整的算法类型信息
        
        # 对于其他算法，组合算法类型和版本
        return f"{algo_type}_{version}"
    
    # 兼容旧版本：从run_name和version提取
    run_name = row['run_name']
    version = row['version']
    
    if "0.9" in run_name:
        if "gae" in run_name.lower():
            return 'turn+token_level_gae'
        elif "avg" in run_name.lower():
            return 'turn+token_level_avg'
        else:
            return 'turn+token_level'
    
    base_type = None
    if 'mask_False_MTGAE_True' in run_name:
        base_type = 'mask_False_MTGAE_True'
    elif 'mask_True_MTGAE_True' in run_name:
        base_type = 'mask_True_MTGAE_True'
    elif 'mask_False_MTGAE_False' in run_name:
        base_type = 'mask_False_MTGAE_False'
    elif 'mask_True_MTGAE_False' in run_name:
        base_type = 'mask_True_MTGAE_False'
    
    if base_type:
        return f"{base_type}_{version}"
    else:
        return 'Unknown'

def get_algorithm_display_name(algo_type):
    """Get display name for algorithm type"""
    display_names = {
        # 只保留需要的算法类型显示名称
        'mask_True_MTGAE_False_V1': 'Mask=True, MTGAE=False (V1)',  # V1版本改为False
        'mask_True_MTGAE_True_V2': 'Mask=True, MTGAE=True (V2)',
        # turn+token level算法的两种类型
        'turn+token_level_gae': 'Turn+Token Level (GAE)',
        'turn+token_level_avg': 'Turn+Token Level (AVG)',
        # 保持兼容性
        'turn+token_level': 'Turn+Token Level'
    }
    return display_names.get(algo_type, algo_type)

def get_line_style(algo_type):
    """Get line style for algorithm type - 全部使用实线"""
    return '-'  # 全部使用实线

def get_line_width(algo_type):
    """Get line width for algorithm type"""
    if 'turn+token_level' in algo_type:
        return 3     # 更粗的线
    else:
        return 2.5   # 统一线宽

def plot_success_rates_subplot(df, ax, data_type, title_suffix):
    """Plot success rate curves for a specific data type (train or val-env)"""
    print(f"绘制{data_type}成功率曲线...")
    
    # Determine the success column based on data type
    if data_type == 'train':
        success_col = 'train/WebShop/success'
    else:  # val-env
        success_col = 'val-env/WebShop/success'
    
    # Filter data that has the required success column
    plot_df = df.dropna(subset=[success_col])
    
    if len(plot_df) == 0:
        print(f"警告: 没有找到{data_type}数据")
        ax.text(0.5, 0.5, f'No {data_type} data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12)
        return {}
    
    algorithm_stats = {}
    
    for algo_type in plot_df['algorithm_type'].unique():
        algo_data = plot_df[plot_df['algorithm_type'] == algo_type]
        run_ids = algo_data['run_id'].unique()
        
        print(f"\n{algo_type} ({data_type}):")
        print(f"  运行数量 (seeds): {len(run_ids)}")
        
        # For each run_id, calculate success rate evolution over steps
        all_runs_data = []
        
        for run_id in run_ids:
            run_data = algo_data[algo_data['run_id'] == run_id].copy()
            run_data = run_data.sort_values('_step')
            
            # For val-env data, we expect fewer data points (every 10 steps)
            min_data_points = 10 if data_type == 'val-env' else 100
            
            # Ensure we have enough data points
            if len(run_data) > min_data_points:
                all_runs_data.append(run_data[['_step', success_col]].values)
        
        if not all_runs_data:
            continue
            
        # Find common step range across all runs
        min_steps = max([data[:, 0].min() for data in all_runs_data])
        max_steps = min([data[:, 0].max() for data in all_runs_data])
        
        # Create common step grid
        # For val-env data, use step size of 10; for train data, use step size of 1
        step_size = 10 if data_type == 'val-env' else 1
        common_steps = np.arange(min_steps, max_steps + step_size, step_size)
        
        # Interpolate each run to common steps
        interpolated_runs = []
        for data in all_runs_data:
            steps, success_rates = data[:, 0], data[:, 1]
            # Use linear interpolation
            interpolated_success = np.interp(common_steps, steps, success_rates)
            interpolated_runs.append(interpolated_success)
        
        if not interpolated_runs:
            continue
            
        # Convert to numpy array
        interpolated_runs = np.array(interpolated_runs)
        
        # Calculate mean and standard deviation
        mean_success = np.mean(interpolated_runs, axis=0)
        std_success = np.std(interpolated_runs, axis=0)
        
        # Apply smoothing (use smaller kernel for val-env data due to fewer points)
        kernel_size = 1 if data_type == 'val-env' else 20
        smoothed_mean = smooth_curve(mean_success, kernel_size=kernel_size)
        smoothed_std = smooth_curve(std_success, kernel_size=kernel_size)
        
        # Store statistics
        algorithm_stats[algo_type] = {
            'steps': common_steps,
            'mean': smoothed_mean,
            'std': smoothed_std,
            'num_runs': len(interpolated_runs)
        }
        
        # Plot curves
        color = ALGORITHM_COLORS.get(algo_type, '#333333')
        display_name = get_algorithm_display_name(algo_type)
        linestyle = get_line_style(algo_type)
        linewidth = get_line_width(algo_type)
        
        ax.plot(common_steps, smoothed_mean, color=color, linewidth=linewidth, 
                linestyle=linestyle, label=f'{display_name} (n={len(interpolated_runs)})')
        
        # Add variance shading
        ax.fill_between(common_steps, 
                       smoothed_mean - smoothed_std, 
                       smoothed_mean + smoothed_std, 
                       color=color, alpha=0.2)
        
        print(f"  最终平均成功率: {smoothed_mean[-1]:.4f} ± {smoothed_std[-1]:.4f}")
        print(f"  最大平均成功率: {smoothed_mean.max():.4f}")
    
    # Set up subplot
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Success Rate', fontsize=10)
    ax.set_title(f'{data_type.title()} Success Rate {title_suffix}', fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.0)
    
    return algorithm_stats

def plot_success_rates(df, output_file):
    """Plot success rate curves with train and val-env subplots"""
    print("绘制成功率曲线...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot train data
    train_stats = plot_success_rates_subplot(df, ax1, 'train', 
                                            'Comparison: V1(False) vs V2(True) vs Turn+Token Level (GAE/AVG)')
    
    # Plot val-env data
    val_stats = plot_success_rates_subplot(df, ax2, 'val-env', 
                                          'Comparison: V1(False) vs V2(True) vs Turn+Token Level (GAE/AVG)')
    
    # Set overall title
    fig.suptitle('1.5B PPO Success Rate Comparison: Train vs Validation', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")
    plt.show()
    
    return train_stats, val_stats

def analyze_ppo_results(input_file, output_file):
    """Main analysis function"""
    # Read CSV file
    print(f"加载数据从: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        return
    
    df = pd.read_csv(input_file)
    
    # Filter out rows with missing success rates
    initial_rows = len(df)
    df = df.dropna(subset=['train/WebShop/success'], how='all')
    print(f"过滤后的数据行数: {len(df)} (原始: {initial_rows})")
    
    # Extract algorithm type (combining base type and version)
    df['algorithm_type'] = df.apply(extract_algorithm_type, axis=1)
    
    # Filter out unknown types
    df = df[df['algorithm_type'] != 'Unknown']
    
    # 只保留指定的算法类型：mask_True_MTGAE_False的V1版本、mask_True_MTGAE_True的V2版本和turn+token_level的两种类型
    target_algorithms = [
        'mask_True_MTGAE_False_V1',   # V1版本改为False
        'mask_True_MTGAE_True_V2', 
        'turn+token_level_gae',
        'turn+token_level_avg'
        # 'turn+token_level'  # 保持兼容性
    ]
    df = df[df['algorithm_type'].isin(target_algorithms)]
    
    print(f"数据概览 (仅保留: mask_True_MTGAE_False V1 + mask_True_MTGAE_True V2 + Turn+Token Level GAE/AVG):")
    print(f"总行数: {len(df)}")
    
    if 'version' in df.columns:
        print(f"版本分布:")
        print(df['version'].value_counts())
    
    print(f"算法类型分布:")
    print(df['algorithm_type'].value_counts())
    print(f"唯一运行ID数量: {df['run_id'].nunique()}")
    
    # 详细显示每种算法类型的分布
    print(f"\n详细算法分布:")
    for algo_type in df['algorithm_type'].unique():
        algo_data = df[df['algorithm_type'] == algo_type]
        print(f"{get_algorithm_display_name(algo_type)}:")
        print(f"  运行数量: {algo_data['run_id'].nunique()}")
        print(f"  数据点数量: {len(algo_data)}")
    
    # 检查是否有足够的数据
    if len(df) == 0:
        print(f"\n❌ 错误: 没有找到指定的算法数据!")
        print(f"   请确保CSV文件中存在相关实验数据")
        return
    
    # Plot success rates
    train_stats, val_stats = plot_success_rates(df, output_file)
    
    # Print final statistics
    print("\n=== 最终统计数据 ===")
    print("\nTrain 成功率统计:")
    for algo_type, stats in train_stats.items():
        display_name = get_algorithm_display_name(algo_type)
        print(f"\n{display_name}:")
        print(f"  运行数量: {stats['num_runs']}")
        print(f"  最终成功率: {stats['mean'][-1]:.4f} ± {stats['std'][-1]:.4f}")
        print(f"  最大成功率: {stats['mean'].max():.4f}")
        print(f"  训练步数范围: {stats['steps'][0]:.0f} - {stats['steps'][-1]:.0f}")
    
    print("\nVal-env 成功率统计:")
    for algo_type, stats in val_stats.items():
        display_name = get_algorithm_display_name(algo_type)
        print(f"\n{display_name}:")
        print(f"  运行数量: {stats['num_runs']}")
        print(f"  最终成功率: {stats['mean'][-1]:.4f} ± {stats['std'][-1]:.4f}")
        print(f"  最大成功率: {stats['mean'].max():.4f}")
        print(f"  验证步数范围: {stats['steps'][0]:.0f} - {stats['steps'][-1]:.0f}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='分析PPO实验结果并生成成功率对比图表')
    
    parser.add_argument('--input', '-i', 
                       type=str, 
                       default='filtered_1.5b_ppo_runs_with_v2_and_turn_token2.csv',
                       help='输入CSV文件路径')
    
    parser.add_argument('--output', '-o', 
                       type=str, 
                       default='ppo_success_rate_comparison.png',
                       help='输出图片文件路径 (默认: ppo_success_rate_comparison.png)')
    
    args = parser.parse_args()
    
    # Validate input file extension
    if not args.input.endswith('.csv'):
        print("警告: 输入文件不是CSV格式，请确保文件格式正确")
    
    # Validate output file extension
    valid_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
    if not any(args.output.endswith(ext) for ext in valid_extensions):
        print("警告: 输出文件扩展名不在支持的格式中 (.png, .jpg, .jpeg, .pdf, .svg)")
        print("将默认使用PNG格式")
        if '.' not in args.output:
            args.output += '.png'
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    
    # Run analysis
    analyze_ppo_results(args.input, args.output)

if __name__ == "__main__":
    main()