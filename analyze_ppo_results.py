import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义算法颜色映射（高对比度颜色用于V1 vs V2对比）
ALGORITHM_COLORS = {
    # V1版本 - 深色系
    'mask_False_MTGAE_True_V1': '#2E86AB',    # 深蓝色
    'mask_True_MTGAE_True_V1': '#D32F2F',     # 深红色 - 更高对比度
    'mask_False_MTGAE_False_V1': '#F18F01',   # 深橙色
    'mask_True_MTGAE_False_V1': '#2F7D32',    # 深绿色
    # V2版本 - 亮色系（高对比度）
    'mask_False_MTGAE_True_V2': '#87CEEB',    # 浅蓝色
    'mask_True_MTGAE_True_V2': '#FF8C00',     # 亮橙色 - 与深红色形成强对比
    'mask_False_MTGAE_False_V2': '#FFB347',   # 浅橙色
    'mask_True_MTGAE_False_V2': '#90EE90'     # 浅绿色
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

def extract_algorithm_type(run_name, version):
    """Extract algorithm type from run_name and version"""
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
        # V1版本
        'mask_False_MTGAE_True_V1': 'Mask=False, MTGAE=True (V1)',
        'mask_True_MTGAE_True_V1': 'Mask=True, MTGAE=True (V1)',
        'mask_False_MTGAE_False_V1': 'Mask=False, MTGAE=False (V1)',
        'mask_True_MTGAE_False_V1': 'Mask=True, MTGAE=False (V1)',
        # V2版本
        'mask_False_MTGAE_True_V2': 'Mask=False, MTGAE=True (V2)',
        'mask_True_MTGAE_True_V2': 'Mask=True, MTGAE=True (V2)',
        'mask_False_MTGAE_False_V2': 'Mask=False, MTGAE=False (V2)',
        'mask_True_MTGAE_False_V2': 'Mask=True, MTGAE=False (V2)'
    }
    return display_names.get(algo_type, algo_type)

def plot_success_rates(df):
    """Plot success rate curves"""
    print("绘制成功率曲线...")
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    algorithm_stats = {}
    
    for algo_type in df['algorithm_type'].unique():
        algo_data = df[df['algorithm_type'] == algo_type]
        run_ids = algo_data['run_id'].unique()
        
        print(f"\n{algo_type}:")
        print(f"  运行数量 (seeds): {len(run_ids)}")
        
        # For each run_id, calculate success rate evolution over steps
        all_runs_data = []
        
        for run_id in run_ids:
            run_data = algo_data[algo_data['run_id'] == run_id].copy()
            run_data = run_data.sort_values('_step')
            
            # Ensure we have enough data points (filter out samples with less than 100 data points)
            if len(run_data) > 100:
                all_runs_data.append(run_data[['_step', 'train/WebShop/success']].values)
        
        if not all_runs_data:
            continue
            
        # Find common step range across all runs
        min_steps = max([data[:, 0].min() for data in all_runs_data])
        max_steps = min([data[:, 0].max() for data in all_runs_data])
        
        # Create common step grid
        common_steps = np.arange(min_steps, max_steps + 1, 1)
        
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
        
        # Apply smoothing
        smoothed_mean = smooth_curve(mean_success, kernel_size=20)
        smoothed_std = smooth_curve(std_success, kernel_size=20)
        
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
        
        # 使用不同的线型区分V1和V2版本
        linestyle = '-' if 'V1' in algo_type else '--'
        linewidth = 2.5 if 'V1' in algo_type else 2
        
        plt.plot(common_steps, smoothed_mean, color=color, linewidth=linewidth, 
                linestyle=linestyle, label=f'{display_name} (n={len(interpolated_runs)})')
        
        # Add variance shading
        plt.fill_between(common_steps, 
                        smoothed_mean - smoothed_std, 
                        smoothed_mean + smoothed_std, 
                        color=color, alpha=0.2)
        
        print(f"  最终平均成功率: {smoothed_mean[-1]:.4f} ± {smoothed_std[-1]:.4f}")
        print(f"  最大平均成功率: {smoothed_mean.max():.4f}")
    
    # Set up plot
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('1.5B PPO Success Rate: Mask=True, MTGAE=True (V1 vs V2)', fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_success.png', dpi=300, bbox_inches='tight')
    plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_success.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return algorithm_stats

def plot_rewards(df):
    """Plot reward curves"""
    print("绘制奖励曲线...")
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    algorithm_stats = {}
    
    for algo_type in df['algorithm_type'].unique():
        algo_data = df[df['algorithm_type'] == algo_type]
        run_ids = algo_data['run_id'].unique()
        
        print(f"\n{algo_type}:")
        print(f"  运行数量 (seeds): {len(run_ids)}")
        
        # For each run_id, calculate reward evolution over steps
        all_runs_data = []
        
        for run_id in run_ids:
            run_data = algo_data[algo_data['run_id'] == run_id].copy()
            run_data = run_data.sort_values('_step')
            
            # Ensure we have enough data points (filter out samples with less than 100 data points)
            if len(run_data) > 100:
                all_runs_data.append(run_data[['_step', 'train/WebShop/reward']].values)
        
        if not all_runs_data:
            continue
            
        # Find common step range across all runs
        min_steps = max([data[:, 0].min() for data in all_runs_data])
        max_steps = min([data[:, 0].max() for data in all_runs_data])
        
        # Create common step grid
        common_steps = np.arange(min_steps, max_steps + 1, 1)
        
        # Interpolate each run to common steps
        interpolated_runs = []
        for data in all_runs_data:
            steps, rewards = data[:, 0], data[:, 1]
            # Use linear interpolation
            interpolated_reward = np.interp(common_steps, steps, rewards)
            interpolated_runs.append(interpolated_reward)
        
        if not interpolated_runs:
            continue
            
        # Convert to numpy array
        interpolated_runs = np.array(interpolated_runs)
        
        # Calculate mean and standard deviation
        mean_reward = np.mean(interpolated_runs, axis=0)
        std_reward = np.std(interpolated_runs, axis=0)
        
        # Apply smoothing
        smoothed_mean = smooth_curve(mean_reward, kernel_size=20)
        smoothed_std = smooth_curve(std_reward, kernel_size=20)
        
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
        
        # 使用不同的线型区分V1和V2版本
        linestyle = '-' if 'V1' in algo_type else '--'
        linewidth = 2.5 if 'V1' in algo_type else 2
        
        plt.plot(common_steps, smoothed_mean, color=color, linewidth=linewidth, 
                linestyle=linestyle, label=f'{display_name} (n={len(interpolated_runs)})')
        
        # Add variance shading
        plt.fill_between(common_steps, 
                        smoothed_mean - smoothed_std, 
                        smoothed_mean + smoothed_std, 
                        color=color, alpha=0.2)
        
        print(f"  最终平均奖励: {smoothed_mean[-1]:.4f} ± {smoothed_std[-1]:.4f}")
        print(f"  最大平均奖励: {smoothed_mean.max():.4f}")
    
    # Set up plot
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('奖励', fontsize=12)
    plt.title('1.5B PPO 奖励曲线: Mask=True, MTGAE=True (V1 vs V2)', fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_reward.png', dpi=300, bbox_inches='tight')
    plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_reward.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return algorithm_stats

def analyze_ppo_results():
    # Read CSV file
    print("加载数据...")
    df = pd.read_csv('filtered_1.5b_ppo_runs_with_v2.csv')
    
    # Filter out rows with missing success rates or rewards
    df = df.dropna(subset=['train/WebShop/success', 'train/WebShop/reward'])
    
    # Extract algorithm type (combining base type and version)
    df['algorithm_type'] = df.apply(lambda row: extract_algorithm_type(row['run_name'], row['version']), axis=1)
    
    # Filter out unknown types
    df = df[df['algorithm_type'] != 'Unknown']
    
    # 只保留 mask_True_MTGAE_True 的V1和V2版本
    target_algorithms = ['mask_True_MTGAE_True_V1', 'mask_True_MTGAE_True_V2']
    df = df[df['algorithm_type'].isin(target_algorithms)]
    
    print(f"数据概览 (仅 mask_True_MTGAE_True 的V1 vs V2):")
    print(f"总行数: {len(df)}")
    print(f"版本分布:")
    print(df['version'].value_counts())
    print(f"算法类型分布:")
    print(df['algorithm_type'].value_counts())
    print(f"唯一运行ID数量: {df['run_id'].nunique()}")
    
    # 详细显示V1和V2版本的分布
    print(f"\n详细版本分布:")
    for version in ['V1', 'V2']:
        version_data = df[df['version'] == version]
        if len(version_data) > 0:
            print(f"{version}版本:")
            print(f"  运行数量: {version_data['run_id'].nunique()}")
            print(f"  数据点数量: {len(version_data)}")
            print(f"  算法类型: {list(version_data['algorithm_type'].unique())}")
        else:
            print(f"{version}版本: 无数据")
    
    # 如果没有V2数据，给出提示
    if len(df[df['version'] == 'V2']) == 0:
        print(f"\n⚠️  注意: 当前没有 mask_True_MTGAE_True 的V2版本数据!")
        print(f"   请检查WandB中是否存在: xxx_mask_True_MTGAE_True_v2")
        print(f"   当前只绘制V1版本的曲线")
    
    # 检查是否有足够的数据
    if len(df) == 0:
        print(f"\n❌ 错误: 没有找到 mask_True_MTGAE_True 的任何数据!")
        print(f"   请确保WandB中存在相关实验")
        return
    
    # Plot success rates
    success_stats = plot_success_rates(df)
    
    # Plot rewards
    # reward_stats = plot_rewards(df)
    
    # Print final statistics
    print("\n=== 最终统计数据 ===")
    print("\n成功率统计:")
    for algo_type, stats in success_stats.items():
        display_name = get_algorithm_display_name(algo_type)
        print(f"\n{display_name}:")
        print(f"  运行数量: {stats['num_runs']}")
        print(f"  最终成功率: {stats['mean'][-1]:.4f} ± {stats['std'][-1]:.4f}")
        print(f"  最大成功率: {stats['mean'].max():.4f}")
        print(f"  训练步数范围: {stats['steps'][0]:.0f} - {stats['steps'][-1]:.0f}")
        
    # print("\n奖励统计:")
    # for algo_type, stats in reward_stats.items():
    #     display_name = get_algorithm_display_name(algo_type)
    #     print(f"\n{display_name}:")
    #     print(f"  运行数量: {stats['num_runs']}")
    #     print(f"  最终奖励: {stats['mean'][-1]:.4f} ± {stats['std'][-1]:.4f}")
    #     print(f"  最大奖励: {stats['mean'].max():.4f}")
    #     print(f"  训练步数范围: {stats['steps'][0]:.0f} - {stats['steps'][-1]:.0f}")

if __name__ == "__main__":
    analyze_ppo_results() 