import json
import os
import sys
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tools.patchcore_migrate.metrics_patchcore import (
    compute_image_metrics,
    compute_pixel_metrics,
    compute_overkill_from_evidence,
)
from tools.patchcore_migrate.dataset_layout import resolve_mask_path


def load_evidences(path: str) -> List[Dict]:
    """小白版说明：把 evidence.jsonl 一行一行读进来，变成 Python 列表。"""
    evidences: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evidences.append(json.loads(line))
    return evidences


def main():
    """小白版说明：只在 eval 集上，按类别统计 PatchCore 的图像级和像素级指标，并写到一个 csv 里。"""
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    runs_dir = os.path.join(root_dir, "runs")
    evidence_run = "patchcore_evidence_ds_mvtec_bottle_cable"
    output_root = os.path.join(runs_dir, evidence_run)
    outputs_dir = os.path.join(output_root, "outputs")
    subset_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable_eval.json")
    evidence_path = os.path.join(outputs_dir, "evidence.jsonl")
    evidences = load_evidences(evidence_path)
    with open(subset_json, "r") as f:
        subset = json.load(f)
    classes = ["bottle", "cable"]
    per_class_scores: Dict[str, List[float]] = {c: [] for c in classes}
    per_class_labels: Dict[str, List[int]] = {c: [] for c in classes}
    per_class_maps: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    per_class_masks: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    per_class_image_labels: Dict[str, Dict[str, int]] = {c: {} for c in classes}
    resize_val = None
    crop_size = None
    for ev in evidences:
        image_key = ev["image_key"]
        _ = subset.get(image_key)
        parts = image_key.split("/")
        if len(parts) < 3:
            continue
        dataset_name = parts[0]
        class_name = parts[1]
        if dataset_name != "DS-MVTec" or class_name not in classes:
            continue
        score = float(ev["anomaly_score"])
        raw_map = np.load(ev["map_path_raw"]).astype(np.float32)
        if resize_val is None:
            resize_val = ev["resize"]
        if crop_size is None:
            crop_size = ev["crop"]
        is_good = "/good/" in image_key
        label = 0 if is_good else 1
        per_class_scores[class_name].append(score)
        per_class_labels[class_name].append(label)
        per_class_maps[class_name].append(raw_map)
        per_class_image_labels[class_name][image_key] = label
        if is_good:
            h, w = raw_map.shape
            zero_mask = np.zeros((h, w), dtype=np.float32)
            per_class_masks[class_name].append(zero_mask)
            continue
        image_path = os.path.join(dataset_dir, image_key)
        mask_path = resolve_mask_path(dataset_dir, dataset_name, image_path)
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            transform_mask = transforms.Compose(
                [
                    transforms.Resize(resize_val),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                ]
            )
            mask_t = transform_mask(mask_img)
            mask_np = mask_t.squeeze(0).numpy().astype(np.float32)
            per_class_masks[class_name].append(mask_np)
    rows: List[List[str]] = []
    header = [
        "class_name",
        "num_eval_total",
        "num_good",
        "num_anom",
        "AUROC_image",
        "AUPR_image",
        "AUROC_pixel",
        "AUPR_pixel",
        "overkill_rate_score",
        "overkill_rate_area",
        "image_reason",
        "pixel_reason",
    ]
    macro_auroc_image: List[float] = []
    macro_aupr_image: List[float] = []
    macro_auroc_pixel: List[float] = []
    macro_aupr_pixel: List[float] = []
    macro_overkill_score: List[float] = []
    macro_overkill_area: List[float] = []
    for cls in classes:
        scores = per_class_scores[cls]
        labels = per_class_labels[cls]
        maps = per_class_maps[cls]
        masks = per_class_masks[cls]
        image_labels = per_class_image_labels[cls]
        num_total = len(labels)
        num_good = int(sum(1 for v in labels if v == 0))
        num_anom = int(sum(1 for v in labels if v == 1))
        img_metrics = compute_image_metrics(scores, labels)
        px_metrics = compute_pixel_metrics(maps, masks)
        overkill_metrics = compute_overkill_from_evidence(
            [ev for ev in evidences if per_class_image_labels[cls].get(ev["image_key"], -1) == 0 or per_class_image_labels[cls].get(ev["image_key"], -1) == 1],
            image_labels,
        )
        auroc_img = img_metrics["auroc_image"]
        aupr_img = img_metrics["aupr_image"]
        auroc_px = px_metrics["auroc_pixel"]
        aupr_px = px_metrics["aupr_pixel"]
        overkill_score = overkill_metrics["overkill_rate_score"]
        overkill_area = overkill_metrics["overkill_rate_area"]
        if auroc_img is not None:
            macro_auroc_image.append(float(auroc_img))
        if aupr_img is not None:
            macro_aupr_image.append(float(aupr_img))
        if auroc_px is not None:
            macro_auroc_pixel.append(float(auroc_px))
        if aupr_px is not None:
            macro_aupr_pixel.append(float(aupr_px))
        macro_overkill_score.append(float(overkill_score))
        macro_overkill_area.append(float(overkill_area))
        row = [
            cls,
            str(num_total),
            str(num_good),
            str(num_anom),
            "NA" if auroc_img is None else str(auroc_img),
            "NA" if aupr_img is None else str(aupr_img),
            "NA" if auroc_px is None else str(auroc_px),
            "NA" if aupr_px is None else str(aupr_px),
            str(overkill_score),
            str(overkill_area),
            img_metrics.get("reason_image", ""),
            px_metrics.get("reason_pixel", ""),
        ]
        rows.append(row)
    if macro_auroc_image or macro_aupr_image or macro_auroc_pixel or macro_aupr_pixel:
        def safe_mean(values: List[float]) -> str:
            return "NA" if not values else str(float(np.mean(values)))
        macro_row = [
            "macro_avg",
            "",
            "",
            "",
            safe_mean(macro_auroc_image),
            safe_mean(macro_aupr_image),
            safe_mean(macro_auroc_pixel),
            safe_mean(macro_aupr_pixel),
            safe_mean(macro_overkill_score),
            safe_mean(macro_overkill_area),
            "",
            "",
        ]
        rows.append(macro_row)
    metrics_csv = os.path.join(outputs_dir, "metrics_patchcore.csv")
    with open(metrics_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
