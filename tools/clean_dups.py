import json
import os
import shutil

def clean_json(json_path):
    print(f"Cleaning {json_path}...")
    
    # 备份原文件
    backup_path = json_path + ".bak"
    shutil.copy(json_path, backup_path)
    print(f"Backup created at {backup_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    unique_data = []
    seen_ids = set()
    
    # 保留最后一次出现的记录（假设最后一次是最新的）
    # 或者第一次也行，既然是 reproduce，内容应该稳定
    # 我们倒序遍历，保留最后出现的，然后反转回来
    for item in reversed(data):
        # 构造唯一键
        key = item.get('id')
        if not key:
            key = f"{item.get('image')}_{item.get('question')}"
            
        if key not in seen_ids:
            seen_ids.add(key)
            unique_data.append(item)
            
    # 反转回来，保持原有顺序
    unique_data.reverse()
    
    print(f"Original: {len(data)}, Cleaned: {len(unique_data)}, Removed: {len(data) - len(unique_data)}")
    
    # 写回文件
    with open(json_path, 'w') as f:
        json.dump(unique_data, f, indent=4)
    print("Cleaned file saved.")

if __name__ == "__main__":
    # 清洗 1-shot similar
    path1 = "/data/XL/MMAD_Base/MMAD-main/runs/2025-12-31_Qwen2.5-VL-7B_1shot_similar_nodk/outputs/answers_1_shot_Qwen2.5-VL-7B-Instruct_Similar_template.json"
    clean_json(path1)
    
    # 清洗 2-shot similar (以防万一)
    path2 = "/data/XL/MMAD_Base/MMAD-main/runs/2025-12-31_Qwen2.5-VL-7B_2shot_similar_nodk/outputs/answers_2_shot_Qwen2.5-VL-7B-Instruct_Similar_template.json"
    if os.path.exists(path2):
        clean_json(path2)
