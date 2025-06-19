import pandas as pd 
import wandb
import re
api = wandb.Api()

def check_run_name_pattern(run_name):
    """
    检查运行名称是否符合要求：
    1. 包含1.5b和ppo
    2. 末尾有mask_{False/True}_MTGAR_{True/False}模式
    """
    # 检查是否包含1.5b和ppo
    if "3b" not in run_name or "ppo" not in run_name:
        return False
    
    # 检查末尾是否有mask_{False/True}_MTGAE_{True/False}模式
    pattern = r'mask_(False|True)_MTGAE_(True|False)$'
    return bool(re.search(pattern, run_name))

# 获取项目运行
runs = api.runs("rl_agent/ragen_latest")

# 存储符合条件的运行数据
filtered_runs_data = []
filtered_run_names = []

print("开始筛选运行...")
print("筛选条件：")
print("1. 运行名称包含 '1.5b' 和 'ppo'")
print("2. 运行名称末尾符合 'mask_{False/True}_MTGAE_{True/False}' 模式")
print("-" * 50)

for run in runs:
    if check_run_name_pattern(run.name):
        print(f"✓ 符合条件: {run.name}")
        filtered_run_names.append(run.name)
        
        try:
            # 获取历史数据
            history = run.history()
            
            # 提取感兴趣的指标
            metrics_of_interest = [
                # 'train/WebShop/success_find',
                'train/WebShop/success', 
                'train/WebShop/reward'
            ]
            
            # 创建包含运行信息的DataFrame
            available_metrics = [col for col in metrics_of_interest if col in history.columns]
            if available_metrics:
                run_data = history[['_step'] + available_metrics].copy()
                
                # 添加运行标识信息
                run_data['run_name'] = run.name
                run_data['run_id'] = run.id
                
                # 添加配置信息
                config = {k: v for k, v in run.config.items() if not k.startswith('_')}
                for key, value in config.items():
                    run_data[f'config_{key}'] = value
                
                filtered_runs_data.append(run_data)
                print(f"  └─ 数据点数量: {len(run_data)}")
                print(f"  └─ 可用指标: {available_metrics}")
            else:
                print(f"  └─ 警告: 没有找到预期的指标数据")
                
        except Exception as e:
            print(f"  └─ 错误: 无法获取数据 - {e}")
    else:
        print(f"✗ 不符合条件: {run.name}")

print("-" * 50)
print(f"筛选完成！共找到 {len(filtered_run_names)} 个符合条件的运行")

if filtered_runs_data:
    # 合并所有符合条件的运行数据
    combined_df = pd.concat(filtered_runs_data, ignore_index=True)
    
    # 保存到CSV文件
    output_file = "filtered_1.5b_ppo_runs.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"✓ 数据已保存到: {output_file}")
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"- 总数据点: {len(combined_df)}")
    print(f"- 包含的运行: {combined_df['run_name'].nunique()}")
    print(f"- 可用指标: {[col for col in combined_df.columns if col.startswith('train/')]}")
    
    # 显示每个运行的数据概览
    print(f"\n各运行数据点分布:")
    run_counts = combined_df.groupby('run_name').size()
    for run_name, count in run_counts.items():
        print(f"- {run_name}: {count} 个数据点")
    
    # 显示前几行数据预览
    print(f"\n数据预览:")
    print(combined_df.head())
    
else:
    print("⚠️  没有找到符合条件的运行数据")

print(f"\n符合条件的运行名称列表:")
for name in filtered_run_names:
    print(f"- {name}") 