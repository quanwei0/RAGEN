import pandas as pd 
import wandb
import re
api = wandb.Api()

def check_run_name_pattern(run_name):
    """
    检查运行名称是否符合要求：
    1. 包含1.5b和ppo
    2. 末尾有mask_{False/True}_MTGAE_{True/False}模式 (可选V2后缀) 或者包含0.9 或者包含bilevel
    """
    # 检查是否包含1.5b和ppo
    if "1.5b" not in run_name or "ppo" not in run_name:
        return False
    
    # 检查末尾是否有mask_{False/True}_MTGAE_{True/False}模式，可选v2后缀
    mask_pattern = r'mask_(False|True)_MTGAE_(True|False)(_v2)?$'
    
    # 或者检查是否包含0.9 (turn+token level算法)
    has_0_9 = "0.9" in run_name
    
    # 或者检查是否包含bilevel (bilevel算法)
    has_bilevel = "bilevel" in run_name.lower()
    
    return bool(re.search(mask_pattern, run_name)) or has_0_9 or has_bilevel

def extract_version_info(run_name):
    """
    提取版本信息：判断是否为V1、V2版本、turn+token level算法或bilevel算法
    - 带_v2后缀 -> V2版本
    - 不带_v2后缀的mask实验 -> V1版本
    - 包含0.9 -> turn+token level算法
    - 包含bilevel -> bilevel算法
    """
    if "bilevel" in run_name.lower():
        return "bilevel"
    elif "0.9" in run_name:
        if "gae" in run_name.lower():
            return "turn+token_level_gae"
        elif "avg" in run_name.lower():
            return "turn+token_level_avg"
        else:
            return "turn+token_level"  # 保持兼容性
    elif '_v2' in run_name:
        return "V2"  # 带_v2后缀的是V2版本
    elif 'mask_' in run_name and 'MTGAE_' in run_name:
        return "V1"  # 不带_v2后缀的mask实验是V1版本
    else:
        return "V1"  # 默认为V1

def extract_algorithm_type_from_name(run_name):
    """
    从运行名称中提取算法类型
    """
    if "bilevel" in run_name.lower():
        return "bilevel"
    elif "0.9" in run_name:
        if "gae" in run_name.lower():
            return "turn+token_level_gae"
        elif "avg" in run_name.lower():
            return "turn+token_level_avg"
        else:
            return "turn+token_level"  # 保持兼容性
    elif 'mask_False_MTGAE_True' in run_name:
        return 'mask_False_MTGAE_True'
    elif 'mask_True_MTGAE_True' in run_name:
        return 'mask_True_MTGAE_True'
    elif 'mask_False_MTGAE_False' in run_name:
        return 'mask_False_MTGAE_False'
    elif 'mask_True_MTGAE_False' in run_name:
        return 'mask_True_MTGAE_False'
    else:
        return 'Unknown'

# 获取项目运行
runs = api.runs("rl_agent/ragen_latest")

# 存储符合条件的运行数据
filtered_runs_data = []
filtered_run_names = []

print("开始筛选运行...")
print("筛选条件：")
print("1. 运行名称包含 '1.5b' 和 'ppo'")
print("2. 运行名称末尾符合 'mask_{False/True}_MTGAE_{True/False}' 模式 (可选_v2后缀)")
print("3. 或者运行名称包含 '0.9' (turn+token level算法)")
print("\n重点抓取的算法类型：")
print("- V1版本: mask_True_MTGAE_True")
print("- V2版本: mask_True_MTGAE_False")
print("- Turn+Token Level: 包含0.9+gae 或 包含0.9+avg")
print("- Bilevel: 包含bilevel")
print("\n正在扫描WandB运行...")
print("-" * 50)

# 先统计所有运行
all_runs_count = 0
v1_true_true_runs_found = []  # V1版本 mask_True_MTGAE_True
v2_runs_found = []
turn_token_gae_runs_found = []
turn_token_avg_runs_found = []
turn_token_other_runs_found = []
bilevel_runs_found = []  # bilevel算法

for run in runs:
    all_runs_count += 1
    
    # 检查是否是V1版本的mask_True_MTGAE_True
    if 'mask_True_MTGAE_True' in run.name:
        v1_true_true_runs_found.append(run.name)
    
    # 检查是否是V2版本的mask_True_MTGAE_False
    if 'mask_True_MTGAE_False' in run.name:
        v2_runs_found.append(run.name)
    
    # 检查是否包含0.9并细分类型
    if '0.9' in run.name:
        if 'gae' in run.name.lower():
            turn_token_gae_runs_found.append(run.name)
        elif 'avg' in run.name.lower():
            turn_token_avg_runs_found.append(run.name)
        else:
            turn_token_other_runs_found.append(run.name)
    
    # 检查是否是bilevel算法
    if 'bilevel' in run.name.lower():
        bilevel_runs_found.append(run.name)
    
    if check_run_name_pattern(run.name):
        version = extract_version_info(run.name)
        algo_type = extract_algorithm_type_from_name(run.name)
        print(f"✓ 符合条件: {run.name} (版本: {version}, 算法类型: {algo_type})")
        filtered_run_names.append(run.name)
        
        try:
            # 获取历史数据
            history = run.history()
            
            # 提取感兴趣的指标
            metrics_of_interest = [
                # 'train/WebShop/success_find',
                'train/WebShop/success', 
                'train/WebShop/reward',
                'val-env/WebShop/success',
                'val-env/WebShop/reward',
            ]
            
            # 创建包含运行信息的DataFrame
            available_metrics = [col for col in metrics_of_interest if col in history.columns]
            if available_metrics:
                run_data = history[['_step'] + available_metrics].copy()
                
                # 添加运行标识信息
                run_data['run_name'] = run.name
                run_data['run_id'] = run.id
                run_data['version'] = extract_version_info(run.name)
                run_data['algorithm_type'] = extract_algorithm_type_from_name(run.name)
                
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
print(f"- V1版本mask_True_MTGAE_True运行数: {len(v1_true_true_runs_found)}")  
print(f"- V2版本mask_True_MTGAE_False运行数: {len(v2_runs_found)}")  
print(f"- 包含0.9+gae的运行数: {len(turn_token_gae_runs_found)}")
print(f"- 包含0.9+avg的运行数: {len(turn_token_avg_runs_found)}")
print(f"- 包含0.9+其他的运行数: {len(turn_token_other_runs_found)}")
print(f"- 包含bilevel的运行数: {len(bilevel_runs_found)}")
print(f"- 符合筛选条件的运行数: {len(filtered_run_names)}")

if v1_true_true_runs_found:
    print(f"\n找到的V1版本mask_True_MTGAE_True运行:")
    for name in v1_true_true_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(v1_true_true_runs_found) > 10:
        print(f"  ... 还有 {len(v1_true_true_runs_found)-10} 个V1 mask_True_MTGAE_True运行")
else:
    print(f"\n❌ 没有找到V1版本mask_True_MTGAE_True的运行!")
    print(f"   请检查WandB中是否存在: xxx_mask_True_MTGAE_True")

if v2_runs_found:
    print(f"\n找到的V2版本mask_True_MTGAE_False运行:")
    for name in v2_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(v2_runs_found) > 10:
        print(f"  ... 还有 {len(v2_runs_found)-10} 个V2运行")
else:
    print(f"\n❌ 没有找到V2版本mask_True_MTGAE_False的运行!")

if turn_token_gae_runs_found:
    print(f"\n找到的turn+token level (GAE)运行:")
    for name in turn_token_gae_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(turn_token_gae_runs_found) > 10:
        print(f"  ... 还有 {len(turn_token_gae_runs_found)-10} 个GAE运行")

if turn_token_avg_runs_found:
    print(f"\n找到的turn+token level (AVG)运行:")
    for name in turn_token_avg_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(turn_token_avg_runs_found) > 10:
        print(f"  ... 还有 {len(turn_token_avg_runs_found)-10} 个AVG运行")

if turn_token_other_runs_found:
    print(f"\n找到的turn+token level (其他)运行:")
    for name in turn_token_other_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(turn_token_other_runs_found) > 10:
        print(f"  ... 还有 {len(turn_token_other_runs_found)-10} 个其他运行")

if bilevel_runs_found:
    print(f"\n找到的bilevel算法运行:")
    for name in bilevel_runs_found[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(bilevel_runs_found) > 10:
        print(f"  ... 还有 {len(bilevel_runs_found)-10} 个bilevel运行")
else:
    print(f"\n❌ 没有找到bilevel算法的运行!")
    print(f"   请检查WandB中是否存在包含'bilevel'的实验")

if not turn_token_gae_runs_found and not turn_token_avg_runs_found and not turn_token_other_runs_found:
    print(f"\n❌ 没有找到包含0.9的运行!")
    print(f"   可能的原因:")
    print(f"   1. WandB中确实没有包含0.9的实验")
    print(f"   2. 包含0.9的实验命名格式不同")

if filtered_runs_data:
    # 合并所有符合条件的运行数据
    combined_df = pd.concat(filtered_runs_data, ignore_index=True)
    
    # 保存到CSV文件
    output_file = "filtered_1.5b_ppo_runs_with_v2_and_turn_token2.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"✓ 数据已保存到: {output_file}")
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"- 总数据点: {len(combined_df)}")
    print(f"- 包含的运行: {combined_df['run_name'].nunique()}")
    print(f"- V1版本运行: {len(combined_df[combined_df['version'] == 'V1']['run_name'].unique())}")
    print(f"  └─ 其中mask_True_MTGAE_True: {len(combined_df[(combined_df['version'] == 'V1') & (combined_df['algorithm_type'] == 'mask_True_MTGAE_True')]['run_name'].unique())}")
    print(f"- V2版本运行: {len(combined_df[combined_df['version'] == 'V2']['run_name'].unique())}")
    print(f"  └─ 其中mask_True_MTGAE_False: {len(combined_df[(combined_df['version'] == 'V2') & (combined_df['algorithm_type'] == 'mask_True_MTGAE_False')]['run_name'].unique())}")
    print(f"- turn+token level (GAE)运行: {len(combined_df[combined_df['version'] == 'turn+token_level_gae']['run_name'].unique())}")
    print(f"- turn+token level (AVG)运行: {len(combined_df[combined_df['version'] == 'turn+token_level_avg']['run_name'].unique())}")
    print(f"- turn+token level (其他)运行: {len(combined_df[combined_df['version'] == 'turn+token_level']['run_name'].unique())}")
    print(f"- bilevel算法运行: {len(combined_df[combined_df['version'] == 'bilevel']['run_name'].unique())}")
    print(f"- 可用指标: {[col for col in combined_df.columns if col.startswith('train/')]}")
    
    # 显示每个版本的运行数据概览
    print(f"\n各运行数据点分布:")
    for version in ['V1', 'V2', 'turn+token_level_gae', 'turn+token_level_avg', 'turn+token_level', 'bilevel']:
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