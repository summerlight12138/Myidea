import os
import glob
import pandas as pd
import sys

def find_accuracy_csv(run_dir):
    """在 outputs 目录下寻找 accuracy csv 文件"""
    pattern = os.path.join(run_dir, "outputs", "*accuracy*.csv")
    files = glob.glob(pattern)
    # 优先找名字里不带 full_report 的，或者直接找第一个
    for f in files:
        if "full_accuracy_report" not in f:
            return f
    return files[0] if files else None

def parse_run(run_dir):
    """解析一个 run 目录，返回字典数据"""
    run_name = os.path.basename(run_dir)
    csv_path = find_accuracy_csv(run_dir)
    
    if not csv_path:
        print(f"Skipping {run_name}: No accuracy CSV found.")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # 假设 csv 的结构是第一行表头，最后一行是 Average
        # 或者我们根据第一列来找
        
        # 我们需要提取几类数据：
        # 1. Overall Average (通常在 'Average' 列的 'Average' 行，或者类似的)
        # 让我们先打印一下列名看看结构，通常是:
        # Index, Anomaly Detection, ..., Average
        
        # 提取 Average 行
        avg_row = df[df.iloc[:, 0] == 'Average']
        if avg_row.empty:
            print(f"Skipping {run_name}: No 'Average' row found in CSV.")
            return None
            
        data = {'Run Name': run_name}
        
        # 把这一行的所有数值都拿出来
        for col in df.columns[1:]: # 跳过第一列索引
            val = avg_row.iloc[0][col]
            data[col] = val
            
        return data
    except Exception as e:
        print(f"Error parsing {run_name}: {e}")
        return None

def main():
    runs_root = "/data/XL/MMAD_Base/MMAD-main/runs"
    all_data = []
    
    print(f"Scanning runs in {runs_root}...")
    
    # 遍历所有子目录
    for d in sorted(os.listdir(runs_root)):
        full_path = os.path.join(runs_root, d)
        if os.path.isdir(full_path):
            run_data = parse_run(full_path)
            if run_data:
                all_data.append(run_data)
    
    if not all_data:
        print("No valid run data found.")
        return

    # 创建 DataFrame
    df_matrix = pd.DataFrame(all_data)
    
    # 调整列顺序，把 Run Name 放第一，Average 放第二
    cols = ['Run Name', 'Average'] + [c for c in df_matrix.columns if c not in ['Run Name', 'Average']]
    # 过滤掉不存在的列
    cols = [c for c in cols if c in df_matrix.columns]
    
    df_matrix = df_matrix[cols]
    
    output_path = os.path.join(runs_root, "baseline_matrix.csv")
    df_matrix.to_csv(output_path, index=False)
    print(f"Matrix generated: {output_path}")
    
    # 简单的打印预览
    print("-" * 80)
    print(df_matrix.to_string(index=False))
    print("-" * 80)

if __name__ == "__main__":
    main()
