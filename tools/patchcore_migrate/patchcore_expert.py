import hashlib
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .patchcore_ref import PatchCore, FaissNN, ApproximateGreedyCoresetSampler, IdentitySampler, load_backbone


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return {"image": tensor, "path": path}


class PatchCoreExpert:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        resize = config.get("resize", 256)
        imagesize = config.get("imagesize", 224)
        mean = config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = config.get("normalize_std", [0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.imagesize = imagesize
        self.resize_shorter = resize

    def _compute_transform_meta(self, pil_image: Image.Image) -> Dict:
        orig_w, orig_h = pil_image.size
        shorter = min(orig_w, orig_h)
        scale = self.resize_shorter / float(shorter)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        crop_size = self.imagesize
        y0 = max((resized_h - crop_size) // 2, 0)
        x0 = max((resized_w - crop_size) // 2, 0)
        meta = {
            "orig_size": [orig_h, orig_w],
            "resize_shorter": self.resize_shorter,
            "resized_size": [resized_h, resized_w],
            "crop_size": crop_size,
            "crop_top_left": [y0, x0],
            "scale": scale,
        }
        return meta

    def forward_transform(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        pil_image = Image.open(image_path).convert("RGB")
        meta = self._compute_transform_meta(pil_image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0), meta

    @staticmethod
    def invert_bbox(bbox_224: Tuple[float, float, float, float], meta: Dict) -> Tuple[int, int, int, int]:
        y0, x0 = meta["crop_top_left"]
        scale = meta["scale"]
        orig_h, orig_w = meta["orig_size"]
        x_min_224, y_min_224, x_max_224, y_max_224 = bbox_224
        x_min_resized = x_min_224 + x0
        y_min_resized = y_min_224 + y0
        x_max_resized = x_max_224 + x0
        y_max_resized = y_max_224 + y0
        x_min_orig = x_min_resized / scale
        y_min_orig = y_min_resized / scale
        x_max_orig = x_max_resized / scale
        y_max_orig = y_max_resized / scale
        x_min_clamped = int(max(0, min(orig_w - 1, x_min_orig)))
        y_min_clamped = int(max(0, min(orig_h - 1, y_min_orig)))
        x_max_clamped = int(max(0, min(orig_w - 1, x_max_orig)))
        y_max_clamped = int(max(0, min(orig_h - 1, y_max_orig)))
        return x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped

    @staticmethod
    def make_key_id(image_key: str) -> str:
        h = hashlib.sha1(image_key.encode("utf-8")).hexdigest()
        return h[:16]

    @staticmethod
    def save_map(array: np.ndarray, save_dir: str, key_id: str, suffix: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{key_id}{suffix}.npy"
        path = os.path.join(save_dir, filename)
        np.save(path, array)
        return path

    def _init_model(self) -> PatchCore:
        backbone_name = self.config.get("backbone_name", "wideresnet50")
        layers = self.config.get("layers_to_extract_from", ["layer2", "layer3"])
        pretrain_dim = self.config.get("pretrain_embed_dim", 1024)
        target_dim = self.config.get("target_embed_dim", 1024)
        patchsize = self.config.get("patchsize", 3)
        patchstride = self.config.get("patchstride", 1)
        nn_k = self.config.get("nn_k", 1)
        coreset_conf = self.config.get("coreset", {"type": "approx_greedy", "percentage": 0.1})
        backbone = load_backbone(backbone_name)
        backbone.name = backbone_name
        input_shape = (3, self.imagesize, self.imagesize)
        if coreset_conf.get("type", "approx_greedy") == "approx_greedy":
            sampler = ApproximateGreedyCoresetSampler(
                percentage=coreset_conf.get("percentage", 0.1),
                device=self.device,
                number_of_starting_points=coreset_conf.get("number_of_starting_points", 10),
                dimension_to_project_features_to=coreset_conf.get("dimension_to_project_features_to", 128),
            )
        else:
            sampler = IdentitySampler()
        nn_method = FaissNN(use_gpu=torch.cuda.is_available(), num_workers=4)
        model = PatchCore(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=layers,
            device=self.device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_dim,
            target_embed_dimension=target_dim,
            patchsize=patchsize,
            patchstride=patchstride,
            anomaly_score_num_nn=nn_k,
            featuresampler=sampler,
            nn_method=nn_method,
        )
        return model

    def build_bank(self, train_image_paths: List[str], class_name: str, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        if len(train_image_paths) == 0:
            raise ValueError("No training images provided for bank construction.")
        split_idx = int(len(train_image_paths) * 0.8)
        train_paths_bank = train_image_paths[:split_idx]
        train_paths_thr = train_image_paths[split_idx:]
        dataset_bank = ImageDataset(train_paths_bank, self.transform)
        loader_bank = DataLoader(dataset_bank, batch_size=self.config.get("batch_size", 16), shuffle=False)
        model = self._init_model()
        model.fit(loader_bank)
        dataset_thr = ImageDataset(train_paths_thr, self.transform)
        loader_thr = DataLoader(dataset_thr, batch_size=self.config.get("batch_size", 16), shuffle=False)
        thr_scores: List[float] = []
        thr_raw_maps: List[np.ndarray] = []
        scores, raw_maps, _ = model.predict_with_raw(loader_thr)
        for s, r in zip(scores, raw_maps):
            thr_scores.append(float(s))
            thr_raw_maps.append(np.asarray(r, dtype=np.float32))
        if len(thr_raw_maps) == 0:
            raise ValueError("Threshold split is empty; cannot compute thresholds.")
        all_values = np.concatenate([m.reshape(-1) for m in thr_raw_maps], axis=0)
        map_bin_thr = float(np.percentile(all_values, 99.0))
        area_ratios = []
        for m in thr_raw_maps:
            bin_map = (m >= map_bin_thr).astype(np.float32)
            area_ratios.append(float(bin_map.mean()))
        score_thr = float(np.percentile(np.array(thr_scores), 95.0))
        area_thr = float(np.percentile(np.array(area_ratios), 95.0))
        thresholds = {
            "score_thr_p95_train_good": score_thr,
            "area_thr_p95_train_good": area_thr,
            "map_bin_thr_p99_train_good": map_bin_thr,
        }
        train_stats = {
            "num_train_good_total": len(train_image_paths),
            "num_train_for_bank": len(train_paths_bank),
            "num_train_for_thr": len(train_paths_thr),
        }
        index_filename = f"nn_index_{class_name}.faiss"
        index_path = os.path.join(save_dir, index_filename)
        model.anomaly_scorer.nn_method.save(index_path)
        bank = {
            "version": 1,
            "class_name": class_name,
            "config": self.config,
            "thresholds": thresholds,
            "train_stats": train_stats,
            "index_path": index_filename,
        }
        bank_path = os.path.join(save_dir, f"bank_{class_name}.pkl")
        with open(bank_path, "wb") as f:
            pickle.dump(bank, f)
        return bank_path

    def infer(self, image_path: str, bank_path: str) -> Dict:
        with open(bank_path, "rb") as f:
            bank = pickle.load(f)
        config = bank["config"]
        thresholds = bank["thresholds"]
        index_path = os.path.join(os.path.dirname(bank_path), bank["index_path"])
        expert = PatchCoreExpert(config)
        model = expert._init_model()
        model.anomaly_scorer.nn_method.load(index_path)
        tensor, meta = expert.forward_transform(image_path)
        dataset = [{"image": tensor[0]}]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores, raw_maps, vis_maps = model.predict_with_raw(loader)
        score = float(scores[0])
        raw_map = raw_maps[0].astype(np.float32)
        vis_map = vis_maps[0].astype(np.float32)
        max_val = float(raw_map.max())
        mean_val = float(raw_map.mean())
        std_val = float(raw_map.std())
        flat = raw_map.reshape(-1)
        hist, bin_edges = np.histogram(flat, bins=64, range=(float(flat.min()), float(flat.max()) + 1e-8), density=True)
        hist = hist + 1e-12
        entropy = float(-(hist * np.log(hist)).sum())
        result = {
            "image_path": image_path,
            "anomaly_score": score,
            "raw_map": raw_map,
            "vis_map": vis_map,
            "transform_meta": meta,
            "thresholds": thresholds,
            "map_stats": {
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "entropy": entropy,
            },
        }
        return result
