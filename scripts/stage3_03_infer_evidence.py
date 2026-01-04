import json
import os
import sys

import numpy as np

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tools.patchcore_migrate import PatchCoreExpert
from tools.patchcore_migrate.evidence_postprocess import build_evidence


def main():
    dataset_dir = os.path.join(root_dir, "dataset", "MMAD")
    runs_dir = os.path.join(root_dir, "runs")
    subset_json = os.path.join(dataset_dir, "mmad_ds_mvtec_bottle_cable_eval.json")
    evidence_run = "patchcore_evidence_ds_mvtec_bottle_cable"
    bank_run = "patchcore_bank_ds_mvtec_bottle_cable"
    output_root = os.path.join(runs_dir, evidence_run)
    outputs_dir = os.path.join(output_root, "outputs")
    maps_raw_dir = os.path.join(outputs_dir, "maps_raw")
    maps_vis_dir = os.path.join(outputs_dir, "maps_vis")
    viz_dir = os.path.join(outputs_dir, "viz")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(maps_raw_dir, exist_ok=True)
    os.makedirs(maps_vis_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    with open(subset_json, "r") as f:
        subset = json.load(f)
    bank_dir = os.path.join(runs_dir, bank_run, "outputs")
    bank_paths = {
        "bottle": os.path.join(bank_dir, "bank_bottle.pkl"),
        "cable": os.path.join(bank_dir, "bank_cable.pkl"),
    }
    expert = PatchCoreExpert({})
    evidence_path = os.path.join(outputs_dir, "evidence.jsonl")
    map_stats_rows = []
    with open(evidence_path, "w") as fout:
        for image_key, entry in subset.items():
            parts = image_key.split("/")
            if len(parts) < 2:
                continue
            dataset_name = parts[0]
            class_name = parts[1]
            if dataset_name != "DS-MVTec":
                continue
            bank_path = bank_paths.get(class_name)
            if bank_path is None or not os.path.exists(bank_path):
                continue
            image_path = os.path.join(dataset_dir, image_key)
            patchcore_output = expert.infer(image_path, bank_path)
            key_id = PatchCoreExpert.make_key_id(image_key)
            raw_map = patchcore_output["raw_map"]
            vis_map = patchcore_output["vis_map"]
            raw_path = PatchCoreExpert.save_map(raw_map, maps_raw_dir, key_id, "")
            vis_path = PatchCoreExpert.save_map(vis_map, maps_vis_dir, key_id, "")
            evidence = build_evidence(
                image_key=image_key,
                image_path=image_path,
                class_name=class_name,
                patchcore_output=patchcore_output,
                key_id=key_id,
                map_path_raw=raw_path,
                map_path_vis=vis_path,
            )
            evidence["split"] = "eval"
            fout.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            map_stats_rows.append(
                {
                    "image_key": image_key,
                    "anomaly_score": evidence["anomaly_score"],
                    "area_ratio": evidence["area_ratio"],
                    "map_max": evidence["map_stats"]["max"],
                    "map_mean": evidence["map_stats"]["mean"],
                    "map_std": evidence["map_stats"]["std"],
                    "map_entropy": evidence["map_stats"]["entropy"],
                }
            )
    map_stats_csv = os.path.join(outputs_dir, "map_stats.csv")
    if map_stats_rows:
        keys = [
            "image_key",
            "anomaly_score",
            "area_ratio",
            "map_max",
            "map_mean",
            "map_std",
            "map_entropy",
        ]
        lines = []
        lines.append(",".join(keys))
        for row in map_stats_rows:
            values = [str(row[k]) for k in keys]
            lines.append(",".join(values))
        with open(map_stats_csv, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
