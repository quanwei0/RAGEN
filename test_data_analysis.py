import pandas as pd

# 读取数据
df = pd.read_csv('filtered_1.5b_ppo_runs_with_v2.csv')

print("=== 数据基本信息 ===")
print(f"总行数: {len(df)}")
print(f"总运行数: {df['run_id'].nunique()}")

print("\n=== 版本分布 ===")
print(df['version'].value_counts())

print("\n=== 运行名称样本 ===")
unique_runs = df['run_name'].unique()
print(f"唯一运行名称数量: {len(unique_runs)}")
print("前10个运行名称:")
for i, name in enumerate(unique_runs[:10]):
    print(f"{i+1}. {name}")

print("\n=== 检查是否有V2结尾的运行名称 ===")
v2_runs = [name for name in unique_runs if name.endswith('V2')]
print(f"以V2结尾的运行数量: {len(v2_runs)}")
if v2_runs:
    print("V2运行名称:")
    for name in v2_runs:
        print(f"- {name}")
else:
    print("没有找到以V2结尾的运行") 