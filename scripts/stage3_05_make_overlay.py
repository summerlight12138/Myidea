import json
import os

import cv2
import numpy as np


def load_evidences(path):
    evidences = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evidences.append(json.loads(line))
    return evidences


def apply_colormap_overlay(image, mask, alpha=0.5):
    image_f = image.astype(np.float32)
    norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_u8 = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = alpha * image_f + (1.0 - alpha) * heatmap.astype(np.float32)
    return overlay.astype(np.uint8)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(root, "dataset", "MMAD")
    runs_dir = os.path.join(root, "runs")
    evidence_run = "patchcore_evidence_ds_mvtec_bottle_cable"
    output_root = os.path.join(runs_dir, evidence_run)
    outputs_dir = os.path.join(output_root, "outputs")
    evidence_path = os.path.join(outputs_dir, "evidence.jsonl")
    viz_dir = os.path.join(outputs_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    evidences = load_evidences(evidence_path)
    for ev in evidences:
        image_path = ev["image_path"]
        vis_map = np.load(ev["map_path_vis"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_resized = cv2.resize(vis_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = apply_colormap_overlay(image, vis_resized)
        for bbox in ev["bboxes"]:
            x_min, y_min, x_max, y_max = bbox["bbox_orig"]
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        out_name = f"{ev['key_id']}.png"
        out_path = os.path.join(viz_dir, out_name)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, overlay_bgr)


if __name__ == "__main__":
    main()
