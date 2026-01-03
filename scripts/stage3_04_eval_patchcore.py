import json
import os
import sys

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


def load_evidences(path):
    evidences = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evidences.append(json.loads(line))
    return evidences


def main():
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    runs_dir = os.path.join(root_dir, "runs")
    evidence_run = "patchcore_evidence_ds_mvtec_bottle_cable"
    output_root = os.path.join(runs_dir, evidence_run)
    outputs_dir = os.path.join(output_root, "outputs")
    subset_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable.json")
    evidence_path = os.path.join(outputs_dir, "evidence.jsonl")
    evidences = load_evidences(evidence_path)
    with open(subset_json, "r") as f:
        subset = json.load(f)
    scores = []
    labels = []
    maps = []
    masks = []
    image_labels = {}
    resize_val = None
    crop_size = None
    for ev in evidences:
        image_key = ev["image_key"]
        entry = subset.get(image_key)
        if entry is None:
            continue
        image_scores = ev["anomaly_score"]
        scores.append(float(image_scores))
        raw_map = np.load(ev["map_path_raw"])
        maps.append(raw_map.astype(np.float32))
        if resize_val is None:
            resize_val = ev["resize"]
        if crop_size is None:
            crop_size = ev["crop"]
        mask_rel = entry.get("mask_path", "")
        if mask_rel:
            parts = image_key.split("/")
            dataset_name = parts[0]
            class_name = parts[1]
            mask_path = os.path.join(dataset_dir, dataset_name, class_name, mask_rel)
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
                transform_mask = transforms.Compose(
                    [
                        transforms.Resize(resize_val),
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(),
                    ]
                )
                mask_t = transform_mask(mask_img)
                mask_np = mask_t.squeeze(0).numpy()
                has_anomaly = float(mask_np.sum()) > 0.0
                labels.append(1 if has_anomaly else 0)
                masks.append(mask_np.astype(np.float32))
                image_labels[image_key] = 1 if has_anomaly else 0
            else:
                h, w = raw_map.shape
                mask_np = np.zeros((h, w), dtype=np.float32)
                labels.append(0)
                masks.append(mask_np)
                image_labels[image_key] = 0
        else:
            h, w = raw_map.shape
            mask_np = np.zeros((h, w), dtype=np.float32)
            labels.append(0)
            masks.append(mask_np)
            image_labels[image_key] = 0
    image_metrics = compute_image_metrics(scores, labels)
    pixel_metrics = compute_pixel_metrics(maps, masks)
    overkill_metrics = compute_overkill_from_evidence(evidences, image_labels)
    metrics_csv = os.path.join(outputs_dir, "metrics_patchcore.csv")
    header = [
        "AUROC_image",
        "AUPR_image",
        "AUROC_pixel",
        "AUPR_pixel",
        "overkill_rate_score",
        "overkill_rate_area",
    ]
    values = [
        image_metrics["auroc_image"],
        image_metrics["aupr_image"],
        pixel_metrics["auroc_pixel"],
        pixel_metrics["aupr_pixel"],
        overkill_metrics["overkill_rate_score"],
        overkill_metrics["overkill_rate_area"],
    ]
    with open(metrics_csv, "w") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(str(v) for v in values) + "\n")


if __name__ == "__main__":
    main()
