import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .patchcore_expert import PatchCoreExpert


GRID_NAMES = [
    ["左上", "上中", "右上"],
    ["左中", "中心", "右中"],
    ["左下", "下中", "右下"],
]


def compute_area_ratio(raw_map: np.ndarray, map_bin_thr: float) -> Tuple[float, np.ndarray]:
    bin_map = (raw_map >= map_bin_thr).astype(np.uint8)
    area_ratio = float(bin_map.mean())
    return area_ratio, bin_map


def _bbox_from_slice(slc: Tuple[slice, slice]) -> Tuple[int, int, int, int]:
    y_slice, x_slice = slc
    y_min = y_slice.start
    y_max = y_slice.stop - 1
    x_min = x_slice.start
    x_max = x_slice.stop - 1
    return x_min, y_min, x_max, y_max


def _grid_position_from_center(cx: float, cy: float, orig_h: int, orig_w: int) -> str:
    col = int(math.floor(cx / (orig_w / 3.0)))
    row = int(math.floor(cy / (orig_h / 3.0)))
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    return GRID_NAMES[row][col]


def extract_bboxes_and_grid(
    raw_map: np.ndarray,
    bin_map: np.ndarray,
    transform_meta: Dict,
    topk: int = 3,
    min_area_ratio: float = 0.001,
) -> Tuple[List[Dict], List[Dict]]:
    """小白版说明：在二值图里找出多个连通的小块，为每个小块算一个框和分数，再挑出分数最高的前几块。"""
    H, W = raw_map.shape
    if bin_map.sum() <= 0:
        return [], []
    labeled, num = ndimage.label(bin_map)
    if num == 0:
        return [], []
    objects = ndimage.find_objects(labeled)
    orig_h, orig_w = transform_meta["orig_size"]
    regions: List[Dict] = []
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        x_min_224, y_min_224, x_max_224, y_max_224 = _bbox_from_slice(slc)
        mask_region = labeled[slc] == idx
        area = int(mask_region.sum())
        if area <= 0:
            continue
        area_ratio = float(area / float(H * W))
        if area_ratio < min_area_ratio:
            continue
        scores_region = raw_map[slc][mask_region]
        mean_score = float(scores_region.mean())
        bbox_224 = (float(x_min_224), float(y_min_224), float(x_max_224), float(y_max_224))
        x_min_orig, y_min_orig, x_max_orig, y_max_orig = PatchCoreExpert.invert_bbox(bbox_224, transform_meta)
        cx_orig = 0.5 * (x_min_orig + x_max_orig)
        cy_orig = 0.5 * (y_min_orig + y_max_orig)
        grid_pos = _grid_position_from_center(cx_orig, cy_orig, orig_h, orig_w)
        regions.append(
            {
                "bbox_orig": [x_min_orig, y_min_orig, x_max_orig, y_max_orig],
                "bbox_224": [x_min_224, y_min_224, x_max_224, y_max_224],
                "area_ratio": area_ratio,
                "mean_score": mean_score,
                "grid_pos": grid_pos,
            }
        )
    regions.sort(key=lambda r: r["mean_score"], reverse=True)
    regions = regions[:topk]
    grid_cells: List[Dict] = []
    for row in range(3):
        for col in range(3):
            name = GRID_NAMES[row][col]
            max_score = 0.0
            cell_area_ratio = 0.0
            for r in regions:
                if r["grid_pos"] == name:
                    if r["mean_score"] > max_score:
                        max_score = r["mean_score"]
                    cell_area_ratio += r["area_ratio"]
            grid_cells.append(
                {
                    "cell": name,
                    "max_score": max_score,
                    "area_ratio": cell_area_ratio,
                }
            )
    return regions, grid_cells


def build_evidence(
    image_key: str,
    image_path: str,
    class_name: str,
    patchcore_output: Dict,
    key_id: str,
    map_path_raw: str,
    map_path_vis: str,
) -> Dict:
    raw_map = patchcore_output["raw_map"]
    thresholds = patchcore_output["thresholds"]
    transform_meta = patchcore_output["transform_meta"]
    score = float(patchcore_output["anomaly_score"])
    map_bin_thr = float(thresholds["map_bin_thr_p99_train_good"])
    score_thr = float(thresholds["score_thr_p95_train_good"])
    area_thr = float(thresholds["area_thr_p95_train_good"])
    area_ratio, bin_map = compute_area_ratio(raw_map, map_bin_thr)
    suspect = (score > score_thr) or (area_ratio > area_thr)
    if suspect:
        bboxes, grid_cells = extract_bboxes_and_grid(
            raw_map,
            bin_map,
            transform_meta,
            topk=3,
            min_area_ratio=0.001,
        )
    else:
        bboxes = []
        grid_cells = []
    evidence = {
        "image_key": image_key,
        "key_id": key_id,
        "image_path": image_path,
        "class_name": class_name,
        "orig_size": transform_meta["orig_size"],
        "model_input_size": [raw_map.shape[0], raw_map.shape[1]],
        "resize": transform_meta["resize_shorter"],
        "crop": transform_meta["crop_size"],
        "resize_mode": "resize_shorter_then_centercrop",
        "anomaly_score": score,
        "score_thr": score_thr,
        "area_ratio": area_ratio,
        "area_thr": area_thr,
        "map_bin_thr": map_bin_thr,
        "map_path_raw": map_path_raw,
        "map_path_vis": map_path_vis,
        "map_stats": {
            "max": patchcore_output["map_stats"]["max"],
            "mean": patchcore_output["map_stats"]["mean"],
            "std": patchcore_output["map_stats"]["std"],
            "entropy": patchcore_output["map_stats"]["entropy"],
            "area_ratio": area_ratio,
        },
        "suspect": suspect,
        "bboxes": bboxes,
        "grid_3x3": grid_cells,
    }
    return evidence
