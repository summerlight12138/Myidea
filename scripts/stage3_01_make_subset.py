import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tools.patchcore_migrate.dataset_layout import (
    list_normal_images,
    list_anomaly_images,
    make_image_key,
)


def stable_bucket(key: str, seed: str) -> float:
    """小白版说明：把字符串稳定地映射到 0~1 之间的小数，用来做可复现切分。"""
    h = hashlib.md5((seed + "::" + key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_good_paths(
    good_paths: List[str],
    seed: str,
    min_thr: int = 30,
    min_eval: int = 30,
) -> Tuple[List[str], List[str], List[str]]:
    """小白版说明：对一个类的所有 good 图，稳定地分成 bank/thr/eval 三份，并尽量保证 thr/eval 数量不太少。"""
    if not good_paths:
        return [], [], []
    paths_sorted = sorted(good_paths)
    bank: List[str] = []
    thr: List[str] = []
    eval_: List[str] = []
    for p in paths_sorted:
        r = stable_bucket(p, seed)
        if r < 0.7:
            bank.append(p)
        elif r < 0.85:
            thr.append(p)
        else:
            eval_.append(p)
    total = len(paths_sorted)
    target_thr = min(min_thr, total // 3) if total >= 3 else len(thr)
    target_eval = min(min_eval, total // 3) if total >= 3 else len(eval_)
    if len(thr) < target_thr and len(bank) > 0:
        need = target_thr - len(thr)
        move = min(need, len(bank))
        thr.extend(bank[-move:])
        bank = bank[:-move]
    if len(eval_) < target_eval and len(bank) > 0:
        need = target_eval - len(eval_)
        move = min(need, len(bank))
        eval_.extend(bank[-move:])
        bank = bank[:-move]
    return bank, thr, eval_


def main():
    """小白版说明：直接从 DS-MVTec 文件夹扫 bottle/cable 的 good/defect 图，做三份切分并生成 eval 子集 json。"""
    root = root_dir
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    subset_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable.json")
    split_json = os.path.join(dataset_dir, "stage3_good_split_bottle_cable.json")
    eval_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable_eval.json")
    dataset_root = dataset_dir
    dataset_name = "DS-MVTec"
    classes = ["bottle", "cable"]
    all_subset: Dict[str, Dict] = {}
    good_split: Dict[str, Dict[str, List[str]]] = {}
    for cls in classes:
        good_paths = list_normal_images(dataset_root, dataset_name, cls)
        anom_paths = list_anomaly_images(dataset_root, dataset_name, cls)
        bank_paths, thr_paths, eval_good_paths = split_good_paths(
            good_paths, seed=f"stage3_split_{cls}"
        )
        bank_keys = [make_image_key(dataset_root, p) for p in bank_paths]
        thr_keys = [make_image_key(dataset_root, p) for p in thr_paths]
        eval_good_keys = [make_image_key(dataset_root, p) for p in eval_good_paths]
        good_split[cls] = {
            "bank_good": bank_keys,
            "thr_good": thr_keys,
            "eval_good": eval_good_keys,
        }
        for p in good_paths:
            key = make_image_key(dataset_root, p)
            all_subset[key] = {
                "dataset_name": dataset_name,
                "class_name": cls,
                "label": 0,
            }
        for p in anom_paths:
            key = make_image_key(dataset_root, p)
            all_subset[key] = {
                "dataset_name": dataset_name,
                "class_name": cls,
                "label": 1,
            }
    with open(subset_json, "w") as f:
        json.dump(all_subset, f, indent=2, ensure_ascii=False)
    with open(split_json, "w") as f:
        json.dump(good_split, f, indent=2, ensure_ascii=False)
    eval_subset: Dict[str, Dict] = {}
    for cls in classes:
        split_entry = good_split.get(cls, {})
        eval_good_keys = set(split_entry.get("eval_good", []))
        for key, value in all_subset.items():
            if value.get("class_name") != cls:
                continue
            is_good = value.get("label", 0) == 0
            if is_good:
                if key not in eval_good_keys:
                    continue
            eval_subset[key] = value
    with open(eval_json, "w") as f:
        json.dump(eval_subset, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
