import json
import os
import shutil
import sys
from typing import Dict, List

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tools.patchcore_migrate import PatchCoreExpert
from tools.patchcore_migrate.patchcore_expert import pickle


def main():
    """小白版说明：用 DS-MVTec 的 good 划分结果，在同一域内为每一类建立 PatchCore 记忆库并记录阈值信息。"""
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    runs_dir = os.path.join(root_dir, "runs")
    run_name = "patchcore_bank_ds_mvtec_bottle_cable"
    run_dir = os.path.join(runs_dir, run_name)
    save_dir = os.path.join(run_dir, "outputs")
    os.makedirs(save_dir, exist_ok=True)
    classes = ["bottle", "cable"]
    split_json = os.path.join(dataset_dir, "stage3_good_split_bottle_cable.json")
    with open(split_json, "r") as f:
        split = json.load(f)
    config = {
        "backbone_name": "wideresnet50",
        "layers_to_extract_from": ["layer2", "layer3"],
        "resize": 256,
        "imagesize": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "pretrain_embed_dim": 1024,
        "target_embed_dim": 1024,
        "patchsize": 3,
        "patchstride": 1,
        "nn_k": 1,
        "coreset": {
            "type": "approx_greedy",
            "percentage": 0.1,
            "number_of_starting_points": 10,
            "dimension_to_project_features_to": 128,
        },
        "batch_size": 16,
        "vis_smoothing_sigma": 4.0,
    }
    expert = PatchCoreExpert(config)
    log: Dict[str, Dict] = {}
    log["config"] = config
    log["split_source"] = os.path.relpath(split_json, start=run_dir)
    for cls in classes:
        cls_split = split.get(cls)
        if not cls_split:
            continue
        bank_keys: List[str] = cls_split.get("bank_good", [])
        thr_keys: List[str] = cls_split.get("thr_good", [])
        eval_keys: List[str] = cls_split.get("eval_good", [])
        bank_paths: List[str] = [os.path.join(dataset_dir, key) for key in bank_keys]
        thr_paths: List[str] = [os.path.join(dataset_dir, key) for key in thr_keys]
        if not bank_paths or not thr_paths:
            continue
        bank_path = expert.build_bank(bank_paths, cls, save_dir, thr_image_paths=thr_paths)
        with open(bank_path, "rb") as f:
            bank_obj = pickle.load(f)
        thresholds = bank_obj.get("thresholds", {})
        train_stats = bank_obj.get("train_stats", {})
        log[cls] = {
            "num_bank_good": len(bank_paths),
            "num_thr_good": len(thr_paths),
            "num_eval_good": len(eval_keys),
            "bank_path": os.path.relpath(bank_path, start=run_dir),
            "thresholds": thresholds,
            "train_stats": train_stats,
        }
    os.makedirs(run_dir, exist_ok=True)
    split_copy_dst = os.path.join(run_dir, "stage3_good_split_bottle_cable.json")
    try:
        shutil.copy(split_json, split_copy_dst)
    except OSError:
        pass
    log_path = os.path.join(run_dir, "run_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
