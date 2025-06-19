import pandas as pd 
import wandb
import re
api = wandb.Api()

def check_run_name_pattern(run_name):
    """
    检查运行名称是否符合要求：
    1. 包含1.5b和ppo
    2. 末尾有mask_{False/True}_MTGAE_{True/False}模式 (可选V2后缀)
    """
    # 检查是否包含1.5b和ppo
    if "1.5b" not in run_name or "ppo" not in run_name:
        return False
    
    # 检查末尾是否有mask_{False/True}_MTGAE_{True/False}模式，可选v2后缀
    pattern = r'mask_(False|True)_MTGAE_(True|False)(_v2)?$'
    return bool(re.search(pattern, run_name))

def extract_version_info(run_name):
    """
    提取版本信息：判断是否为V2版本
    """
    return "V2" if run_name.endswith("v2") else "V1"

# 获取项目运行
runs = api.runs("rl_agent/ragen_latest")

# 存储符合条件的运行数据
filtered_runs_data = []
filtered_run_names = []

print("开始筛选运行...")
print("筛选条件：")
print("1. 运行名称包含 '1.5b' 和 'ppo'")
print("2. 运行名称末尾符合 'mask_{False/True}_MTGAE_{True/False}' 模式 (可选_v2后缀)")
print("\n正在扫描WandB运行...")
print("-" * 50)

# 先统计所有运行
all_runs_count = 0
v2_runs_found = []

for run in runs:
    all_runs_count += 1
    
    # 检查是否包含v2
    if 'v2' in run.name:
        v2_runs_found.append(run.name)
    
    if check_run_name_pattern(run.name):
        version = extract_version_info(run.name)
        print(f"✓ 符合条件: {run.name} (版本: {version})")
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
                run_data['version'] = extract_version_info(run.name)
                
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
print(f"扫描完成！")
print(f"- 总运行数: {all_runs_count}")
print(f"- 包含v2的运行数: {len(v2_runs_found)}")
print(f"- 符合筛选条件的运行数: {len(filtered_run_names)}")

if v2_runs_found:
    print(f"\n找到的v2运行:")
    for name in v2_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(v2_runs_found) > 10:
        print(f"  ... 还有 {len(v2_runs_found)-10} 个v2运行")
else:
    print(f"\n❌ 没有找到包含v2的运行!")
    print(f"   可能的原因:")
    print(f"   1. WandB中确实没有v2版本的实验")
    print(f"   2. v2实验的命名格式不同")

if filtered_runs_data:
    # 合并所有符合条件的运行数据
    combined_df = pd.concat(filtered_runs_data, ignore_index=True)
    
    # 保存到CSV文件
    output_file = "filtered_1.5b_ppo_runs_with_v2.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"✓ 数据已保存到: {output_file}")
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"- 总数据点: {len(combined_df)}")
    print(f"- 包含的运行: {combined_df['run_name'].nunique()}")
    print(f"- V1版本运行: {len(combined_df[combined_df['version'] == 'V1']['run_name'].unique())}")
    print(f"- V2版本运行: {len(combined_df[combined_df['version'] == 'V2']['run_name'].unique())}")
    print(f"- 可用指标: {[col for col in combined_df.columns if col.startswith('train/')]}")
    
    # 显示每个版本的运行数据概览
    print(f"\n各运行数据点分布:")
    for version in ['V1', 'V2']:
        version_data = combined_df[combined_df['version'] == version]
        if len(version_data) > 0:
            print(f"\n{version}版本:")
            run_counts = version_data.groupby('run_name').size()
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