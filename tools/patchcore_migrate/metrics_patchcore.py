from typing import Dict, List, Tuple

import numpy as np
from sklearn import metrics


def compute_image_metrics(scores: List[float], labels: List[int]) -> Dict:
    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    fpr, tpr, _ = metrics.roc_curve(labels_arr, scores_arr)
    auroc = metrics.roc_auc_score(labels_arr, scores_arr)
    precision, recall, _ = metrics.precision_recall_curve(labels_arr, scores_arr)
    aupr = metrics.auc(recall, precision)
    return {
        "auroc_image": float(auroc),
        "aupr_image": float(aupr),
        "fpr_image": fpr,
        "tpr_image": tpr,
    }


def compute_pixel_metrics(anomaly_maps: List[np.ndarray], masks: List[np.ndarray]) -> Dict:
    if isinstance(anomaly_maps, list):
        anomaly_maps = np.stack(anomaly_maps)
    if isinstance(masks, list):
        masks = np.stack(masks)
    flat_scores = anomaly_maps.reshape(-1).astype(np.float32)
    flat_labels = masks.reshape(-1).astype(np.int32)
    fpr, tpr, _ = metrics.roc_curve(flat_labels, flat_scores)
    auroc = metrics.roc_auc_score(flat_labels, flat_scores)
    precision, recall, _ = metrics.precision_recall_curve(flat_labels, flat_scores)
    aupr = metrics.auc(recall, precision)
    return {
        "auroc_pixel": float(auroc),
        "aupr_pixel": float(aupr),
        "fpr_pixel": fpr,
        "tpr_pixel": tpr,
    }


def compute_overkill_from_evidence(
    evidences: List[Dict],
    image_labels: Dict[str, int],
) -> Dict:
    score_flags = []
    area_flags = []
    scores = []
    areas = []
    for ev in evidences:
        key = ev["image_key"]
        lbl = image_labels.get(key, 0)
        if lbl != 0:
            continue
        thr_score = ev["score_thr"]
        thr_area = ev["area_thr"]
        score = ev["anomaly_score"]
        area_ratio = ev["area_ratio"]
        scores.append(score)
        areas.append(area_ratio)
        score_flags.append(float(score > thr_score))
        area_flags.append(float(area_ratio > thr_area))
    if len(score_flags) == 0:
        return {
            "overkill_rate_score": 0.0,
            "overkill_rate_area": 0.0,
        }
    return {
        "overkill_rate_score": float(np.mean(score_flags)),
        "overkill_rate_area": float(np.mean(area_flags)),
    }

