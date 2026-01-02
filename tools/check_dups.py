import json
import collections
import sys

def check_duplicates(json_path):
    print(f"Checking {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    total_records = len(data)
    print(f"Total records: {total_records}")

    # 使用 (image_path, question_id) 作为唯一键来检查重复
    # 如果没有 question_id，就用 image_path + question_content
    unique_keys = set()
    duplicates = []

    for item in data:
        # 构造唯一标识
        # 这里假设 'id' 是唯一的，如果没有 id，可以用 image + question
        key = item.get('id')
        if not key:
            key = f"{item.get('image')}_{item.get('question')}"
        
        if key in unique_keys:
            duplicates.append(key)
        else:
            unique_keys.add(key)

    num_duplicates = len(duplicates)
    unique_count = len(unique_keys)

    print(f"Unique records: {unique_count}")
    print(f"Duplicate records: {num_duplicates}")
    
    if num_duplicates > 0:
        print(f"Duplicate rate: {num_duplicates / total_records:.2%}")
        # 看看重复的是哪些 dataset
        dup_datasets = collections.Counter()
        for item in data:
            key = item.get('id') or f"{item.get('image')}_{item.get('question')}"
            if key in duplicates: # 注意：这里逻辑稍微有点粗糙，因为 set 里存的是第一次出现的，这里会统计所有出现的
                pass 
        
        print("Duplicates confirmed. This confirms the user's suspicion.")
    else:
        print("No duplicates found. The data is clean.")

if __name__ == "__main__":
    path = "/data/XL/MMAD_Base/MMAD-main/runs/2025-12-31_Qwen2.5-VL-7B_1shot_similar_nodk/outputs/answers_1_shot_Qwen2.5-VL-7B-Instruct_Similar_template.json"
    check_duplicates(path)
