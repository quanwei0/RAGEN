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

# 定义算法颜色映射（包含所有mask配置的V1和V2版本）
ALGORITHM_COLORS = {
    # V1版本的各种配置
    'mask_True_MTGAE_True_V1': '#D32F2F',     # 深红色
    'mask_True_MTGAE_False_V1': '#B71C1C',    # 更深的红色
    'mask_False_MTGAE_True_V1': '#E57373',    # 浅红色
    'mask_False_MTGAE_False_V1': '#FFCDD2',   # 很浅的红色
    # V2版本的各种配置
    'mask_True_MTGAE_True_V2': '#FF8C00',     # 橙色
    'mask_True_MTGAE_False_V2': '#FF6F00',    # 深橙色
    'mask_False_MTGAE_True_V2': '#FFB74D',    # 浅橙色
    'mask_False_MTGAE_False_V2': '#FFCC80',   # 很浅的橙色
    # turn+token level算法的两种类型
    'turn+token_level_gae': '#9932CC',        # 深紫色
    'turn+token_level_avg': '#228B22',        # 森林绿色
    # bilevel算法
    'bilevel': '#1E90FF',                     # 道奇蓝
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
        
        # 对于bilevel算法，直接返回
        if algo_type == 'bilevel':
            return 'bilevel'
        
        # 对于turn+token level算法，直接返回（包括gae和avg类型）
        if 'turn+token_level' in algo_type:
            return version  # version字段已经包含完整的算法类型信息
        
        # 对于其他算法，组合算法类型和版本
        return f"{algo_type}_{version}"
    
    # 兼容旧版本：从run_name和version提取
    run_name = row['run_name']
    version = row['version']
    
    if "bilevel" in run_name.lower():
        return 'bilevel'
    elif "0.9" in run_name:
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
        # V1和V2版本的各种mask配置
        'mask_True_MTGAE_True_V1': 'Mask=True, MTGAE=True (V1)',
        'mask_True_MTGAE_True_V2': 'Mask=True, MTGAE=True (V2)',
        'mask_True_MTGAE_False_V1': 'Mask=True, MTGAE=False (V1)',
        'mask_True_MTGAE_False_V2': 'Mask=True, MTGAE=False (V2)',
        'mask_False_MTGAE_True_V1': 'Mask=False, MTGAE=True (V1)',
        'mask_False_MTGAE_True_V2': 'Mask=False, MTGAE=True (V2)',
        'mask_False_MTGAE_False_V1': 'Mask=False, MTGAE=False (V1)',
        'mask_False_MTGAE_False_V2': 'Mask=False, MTGAE=False (V2)',
        # turn+token level算法的两种类型
        'turn+token_level_gae': 'Turn+Token Level (GAE)',
        'turn+token_level_avg': 'Turn+Token Level (AVG)',
        # bilevel算法
        'bilevel': 'Bilevel',
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
        used_runs_info = []  # 记录实际使用的运行信息
        
        for run_id in run_ids:
            run_data = algo_data[algo_data['run_id'] == run_id].copy()
            run_data = run_data.sort_values('_step')
            
            # Ensure we have enough data points (filter out samples with less than 100 data points)
            if len(run_data) > 250:
                all_runs_data.append(run_data[['_step', 'train/WebShop/success']].values)
                # 记录使用的运行信息
                run_name = run_data['run_name'].iloc[0] if 'run_name' in run_data.columns else f"run_id_{run_id}"
                used_runs_info.append({
                    'run_id': run_id,
                    'run_name': run_name,
                    'data_points': len(run_data)
                })
        
        # Add additional bilevel data if this is bilevel algorithm
        if algo_type == 'bilevel':
            # Additional bilevel success rate data
            additional_bilevel_data = [0.0, 0.031, 0.125, 0.031, 0.0, 0.0, 0.281, 0.094, 0.062, 0.0, 0.0, 0.031, 0.0, 0.0, 0.0, 0.031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031, 0.031, 0.0, 0.0, 0.031, 0.031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.156, 0.0, 0.0, 0.0, 0.0, 0.031, 0.031, 0.031, 0.031, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.062, 0.0, 0.094, 0.0, 0.0, 0.281, 0.0, 0.0, 0.0, 0.0, 0.062, 0.031, 0.0, 0.0, 0.0, 0.0, 0.031, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031, 0.0, 0.0, 0.062, 0.125, 0.062, 0.0, 0.062, 0.188, 0.156, 0.031, 0.0, 0.031, 0.188, 0.031, 0.188, 0.219, 0.0, 0.094, 0.0, 0.219, 0.312, 0.0, 0.0, 0.0, 0.188, 0.062, 0.188, 0.531, 0.125, 0.25, 0.75, 0.312, 0.281, 0.0, 0.031, 0.031, 0.219, 0.188, 0.656, 0.406, 0.0, 0.344, 0.438, 0.0, 0.469, 0.469, 0.062, 0.0, 0.375, 0.812, 0.375, 0.625, 0.312, 0.469, 0.656, 0.031, 0.0, 0.531, 0.062, 0.344, 0.0, 0.969, 0.406, 0.031, 0.5, 0.375, 0.656, 0.406, 0.0, 0.0, 0.062, 0.188, 0.0, 0.812, 0.219, 0.781, 0.031, 0.0, 0.406, 0.875, 0.625, 0.5, 0.094, 0.5, 0.438, 0.469, 0.531, 0.0, 0.531, 0.562, 0.5, 0.375, 0.031, 0.406, 0.594, 0.469, 0.406, 0.0, 0.625, 0.406, 0.0, 0.0, 0.406, 0.125, 0.188, 0.469, 0.938, 0.875, 0.5, 0.5, 0.469, 0.469, 0.312, 0.5, 0.0, 0.5, 0.406, 0.125, 0.531, 0.906, 0.906, 0.188, 0.406, 0.0, 0.031, 0.5, 0.312, 0.562, 0.969, 0.969, 0.5, 0.281, 0.0, 0.656, 0.0, 0.0, 0.438, 0.0, 0.625, 0.844, 0.406, 0.438, 0.062, 0.438, 0.969, 0.625, 0.5, 0.125, 0.625, 0.781, 0.938, 0.875, 0.406, 0.438, 0.406, 0.0, 0.031, 0.656, 0.469, 0.594, 0.5, 0.438, 0.375, 0.531, 0.094, 0.031, 0.312, 0.531, 0.719, 0.5, 0.438, 0.875, 0.438, 0.5, 0.0, 0.469, 0.781, 0.156, 0.906, 0.0, 0.5, 0.531, 0.906, 0.938, 0.75, 0.719, 0.125, 0.969, 0.594, 0.0, 0.344, 0.656, 1.0, 0.0, 0.375, 0.5, 0.531, 0.0, 0.0, 0.719, 0.0, 0.5, 0.438, 0.375, 0.0, 0.469, 0.062]
            
            # Create step array for additional data (assuming it starts from step 0)
            additional_steps = np.arange(len(additional_bilevel_data))
            additional_data_array = np.column_stack([additional_steps, additional_bilevel_data])
            
            # Add to all_runs_data
            all_runs_data.append(additional_data_array)
            used_runs_info.append({
                'run_id': 'additional_bilevel_data',
                'run_name': 'Manual_Bilevel_Success_Rate_Data',
                'data_points': len(additional_bilevel_data)
            })
            print(f"  添加了额外的bilevel数据: {len(additional_bilevel_data)} 个数据点")
        
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
        linestyle = get_line_style(algo_type)
        linewidth = get_line_width(algo_type)
        
        plt.plot(common_steps, smoothed_mean, color=color, linewidth=linewidth, 
                linestyle=linestyle, label=f'{display_name} (n={len(interpolated_runs)})')
        
        # Add variance shading
        plt.fill_between(common_steps, 
                        smoothed_mean - smoothed_std, 
                        smoothed_mean + smoothed_std, 
                        color=color, alpha=0.2)
        
        print(f"  最终平均成功率: {smoothed_mean[-1]:.4f} ± {smoothed_std[-1]:.4f}")
        print(f"  最大平均成功率: {smoothed_mean.max():.4f}")
        
        # Print detailed run information for bilevel algorithm
        if algo_type == 'bilevel':
            print(f"\n=== BILEVEL算法使用的运行详情 ===")
            print(f"总共使用了 {len(used_runs_info)} 条运行数据:")
            for i, run_info in enumerate(used_runs_info, 1):
                print(f"  {i}. Run ID: {run_info['run_id']}")
                print(f"     Run Name: {run_info['run_name']}")
                print(f"     Data Points: {run_info['data_points']}")
            print(f"=== BILEVEL详情结束 ===\n")
    
    # Set up plot
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('1.5B PPO Success Rate Comparison: Mask=True&MTGAE=True (V1 vs V2) vs Turn+Token Level vs Bilevel', fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_comparison.png', dpi=300, bbox_inches='tight')
    # plt.savefig('1.5b_ppo_mask_true_mtgae_true_v1_v2_comparison.pdf', dpi=300, bbox_inches='tight')
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
        linestyle = get_line_style(algo_type)
        linewidth = get_line_width(algo_type)
        
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
    plt.title('1.5B PPO 奖励曲线比较: V1(True True) vs V2(True False) vs Turn+Token Level (GAE/AVG)', fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('1.5b_ppo_algorithm_comparison_gae_avg_reward.png', dpi=300, bbox_inches='tight')
    plt.savefig('1.5b_ppo_algorithm_comparison_gae_avg_reward.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return algorithm_stats

def analyze_ppo_results():
    # Read CSV file
    print("加载数据...")
    df = pd.read_csv('filtered_1.5b_ppo_runs_with_v2_and_turn_token2.csv')
    
    # Filter out rows with missing success rates or rewards
    df = df.dropna(subset=['train/WebShop/success', 'train/WebShop/reward'])
    # df = df.dropna(subset=['val-env/WebShop/success', 'val-env/WebShop/reward'])
    
    # Extract algorithm type (combining base type and version)
    df['algorithm_type'] = df.apply(extract_algorithm_type, axis=1)
    
    # Filter out unknown types
    df = df[df['algorithm_type'] != 'Unknown']
    
    # 只保留指定的算法类型：mask_True_MTGAE_True的V1和V2版本、turn+token_level的两种类型和bilevel算法
    target_algorithms = [
        'mask_True_MTGAE_True_V1',   # V1版本：mask=True, MTGAE=True
        'mask_True_MTGAE_True_V2',   # V2版本：mask=True, MTGAE=True (带_v2后缀的实验)
        'turn+token_level_gae',
        'turn+token_level_avg',
        'bilevel'                    # bilevel算法
        # 'turn+token_level'  # 保持兼容性
    ]
    df = df[df['algorithm_type'].isin(target_algorithms)]
    
    print(f"数据概览 (仅保留: Mask=True&MTGAE=True的V1和V2版本 + Turn+Token Level GAE/AVG + Bilevel):")
    print(f"总行数: {len(df)}")
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
        print(f"   请确保WandB中存在相关实验 (Mask=True&MTGAE=True的V1和V2版本, Turn+Token Level GAE/AVG, Bilevel)")
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