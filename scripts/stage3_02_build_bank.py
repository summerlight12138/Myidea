import json
import os
import sys
from typing import Dict, List

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tools.patchcore_migrate import PatchCoreExpert


def collect_mvtec_ad_train_good(mmad_root: str, classes: List[str]) -> Dict[str, List[str]]:
    base = os.path.join(mmad_root, "MVTec-AD")
    result: Dict[str, List[str]] = {}
    for cls in classes:
        train_good_dir = os.path.join(base, cls, "train", "good")
        if not os.path.isdir(train_good_dir):
            continue
        paths = []
        for fname in sorted(os.listdir(train_good_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(train_good_dir, fname))
        result[cls] = paths
    return result


def main():
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    runs_dir = os.path.join(root_dir, "runs")
    run_name = "patchcore_bank_ds_mvtec_bottle_cable"
    save_dir = os.path.join(runs_dir, run_name, "outputs")
    os.makedirs(save_dir, exist_ok=True)
    classes = ["bottle", "cable"]
    train_paths = collect_mvtec_ad_train_good(dataset_dir, classes)
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
    }
    expert = PatchCoreExpert(config)
    log = {}
    for cls, paths in train_paths.items():
        if not paths:
            continue
        bank_path = expert.build_bank(paths, cls, save_dir)
        log[cls] = {
            "num_train_good": len(paths),
            "bank_path": os.path.relpath(bank_path, start=os.path.join(runs_dir, run_name)),
        }
    log_path = os.path.join(runs_dir, run_name, "run_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
