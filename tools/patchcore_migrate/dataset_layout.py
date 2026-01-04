import glob
import os
from typing import Dict, List, Optional


def _to_posix(path: str) -> str:
    """小白版说明：把路径里的反斜杠都换成斜杠，保证 key 形式统一。"""
    return path.replace("\\", "/")


def list_normal_images(dataset_root: str, dataset_name: str, class_name: str) -> List[str]:
    """小白版说明：给定数据集名字和类别，列出“正常训练图”的绝对路径列表。"""
    if dataset_name == "DS-MVTec":
        pattern = os.path.join(dataset_root, "DS-MVTec", class_name, "image", "good", "*.png")
        return sorted(glob.glob(pattern))
    if dataset_name == "MVTec-AD":
        pattern = os.path.join(dataset_root, "MVTec-AD", class_name, "train", "good", "*.png")
        return sorted(glob.glob(pattern))
    if dataset_name == "VisA":
        pattern = os.path.join(dataset_root, "VisA", class_name, "train", "good", "*.png")
        return sorted(glob.glob(pattern))
    if dataset_name == "GoodsAD":
        pattern = os.path.join(dataset_root, "GoodsAD", class_name, "train", "good", "*.png")
        return sorted(glob.glob(pattern))
    if dataset_name == "MVTec-LOCO":
        pattern = os.path.join(dataset_root, "MVTec-LOCO", class_name, "train", "good", "*.png")
        return sorted(glob.glob(pattern))
    return []


def list_anomaly_images(dataset_root: str, dataset_name: str, class_name: str) -> List[str]:
    """小白版说明：给定数据集和类别，列出所有异常图的绝对路径列表。"""
    if dataset_name == "DS-MVTec":
        base = os.path.join(dataset_root, "DS-MVTec", class_name, "image")
        if not os.path.isdir(base):
            return []
        out: List[str] = []
        for d in sorted(os.listdir(base)):
            defect_dir = os.path.join(base, d)
            if not os.path.isdir(defect_dir):
                continue
            if d == "good":
                continue
            out.extend(sorted(glob.glob(os.path.join(defect_dir, "*.png"))))
        return out
    if dataset_name == "MVTec-AD":
        base = os.path.join(dataset_root, "MVTec-AD", class_name, "test")
        if not os.path.isdir(base):
            return []
        out: List[str] = []
        for d in sorted(os.listdir(base)):
            defect_dir = os.path.join(base, d)
            if not os.path.isdir(defect_dir):
                continue
            if d == "good":
                continue
            out.extend(sorted(glob.glob(os.path.join(defect_dir, "*.png"))))
        return out
    if dataset_name in ("VisA", "GoodsAD"):
        base = os.path.join(dataset_root, dataset_name, class_name, "test")
        if not os.path.isdir(base):
            return []
        out: List[str] = []
        for d in sorted(os.listdir(base)):
            defect_dir = os.path.join(base, d)
            if not os.path.isdir(defect_dir):
                continue
            if d == "good":
                continue
            out.extend(sorted(glob.glob(os.path.join(defect_dir, "*.png"))))
        return out
    if dataset_name == "MVTec-LOCO":
        base = os.path.join(dataset_root, "MVTec-LOCO", class_name, "test")
        if not os.path.isdir(base):
            return []
        out: List[str] = []
        for d in ("logical_anomalies", "structural_anomalies"):
            defect_dir = os.path.join(base, d)
            if not os.path.isdir(defect_dir):
                continue
            out.extend(sorted(glob.glob(os.path.join(defect_dir, "*.png"))))
        return out
    return []


def resolve_mask_path(dataset_root: str, dataset_name: str, image_path: str) -> Optional[str]:
    """小白版说明：根据图像所在数据集，尽量推断出对应的 mask 路径。找不到就返回 None。"""
    image_path = os.path.abspath(image_path)
    if dataset_name == "DS-MVTec":
        norm = _to_posix(image_path)
        if "/image/good/" in norm:
            return None
        cand = norm.replace("/image/", "/mask/")
        if os.path.exists(cand):
            return cand
        return None
    if dataset_name == "MVTec-AD":
        norm = _to_posix(image_path)
        if "/test/good/" in norm or "/train/good/" in norm:
            return None
        parts = norm.split("/")
        if "test" in parts:
            idx = parts.index("test")
            cls = parts[idx - 1]
            defect = parts[idx + 1]
            fname = parts[idx + 2]
            cand = os.path.join(dataset_root, "MVTec-AD", cls, "ground_truth", defect, fname)
            if os.path.exists(cand):
                return cand
        return None
    if dataset_name in ("VisA", "GoodsAD"):
        norm = _to_posix(image_path)
        if "/test/good/" in norm or "/train/good/" in norm:
            return None
        parts = norm.split("/")
        try:
            dataset_idx = parts.index(dataset_name)
        except ValueError:
            return None
        cls = parts[dataset_idx + 1]
        split_name = parts[dataset_idx + 2]
        defect = parts[dataset_idx + 3]
        fname = parts[dataset_idx + 4]
        if split_name != "test":
            return None
        cand = os.path.join(dataset_root, dataset_name, cls, "ground_truth", defect, fname)
        if os.path.exists(cand):
            return cand
        return None
    if dataset_name == "MVTec-LOCO":
        norm = _to_posix(image_path)
        if "/test/good/" in norm or "/train/good/" in norm:
            return None
        parts = norm.split("/")
        try:
            idx_dataset = parts.index("MVTec-LOCO")
        except ValueError:
            return None
        cls = parts[idx_dataset + 1]
        split_name = parts[idx_dataset + 2]
        defect_type = parts[idx_dataset + 3]
        fname = parts[idx_dataset + 4]
        if split_name != "test":
            return None
        cand = os.path.join(
            dataset_root,
            "MVTec-LOCO",
            cls,
            "ground_truth",
            defect_type,
            os.path.splitext(fname)[0],
        )
        if os.path.isdir(cand):
            inner = sorted(glob.glob(os.path.join(cand, "*.png")))
            if inner:
                return inner[0]
        return None
    return None


def make_image_key(dataset_root: str, abs_path: str) -> str:
    """小白版说明：把绝对路径转成相对 dataset_root 的 key，统一用斜杠分隔。"""
    rel = os.path.relpath(abs_path, dataset_root)
    return _to_posix(rel)

