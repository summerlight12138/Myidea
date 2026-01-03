import copy
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ForwardHook:
    def __init__(self, hook_dict: Dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, inputs, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NetworkFeatureAggregator(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, layers_to_extract_from: List[str], device: torch.device):
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs: Dict[str, torch.Tensor] = {}
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape: Tuple[int, int, int]) -> List[int]:
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class Preprocessing(torch.nn.Module):
    def __init__(self, feature_dimensions: List[int], output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.feature_dimensions = feature_dimensions
        self.mappers = torch.nn.ModuleList()
        for dim in feature_dimensions:
            mapper = torch.nn.Linear(dim, output_dim, bias=False)
            self.mappers.append(mapper)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        mapped = []
        for feature, mapper in zip(features, self.mappers):
            b, c, h, w = feature.shape
            feature = feature.reshape(b, c, h * w).permute(0, 2, 1).reshape(-1, c)
            feature = mapper(feature)
            feature = feature.reshape(b, h * w, self.output_dim)
            mapped.append(feature)
        return torch.cat(mapped, dim=-1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device: torch.device, target_size: Union[int, Tuple[int, int]]):
        self.device = device
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            scores = patch_scores.to(self.device)
            scores = scores.unsqueeze(1)
            scores = F.interpolate(scores, size=self.target_size, mode="bilinear", align_corners=False)
            scores = scores.squeeze(1)
            patch_scores = scores.cpu().numpy()
        return [patch_score for patch_score in patch_scores]


class FaissNN:
    def __init__(self, use_gpu: bool = False, num_workers: int = 4):
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        self.features_t: Union[torch.Tensor, None] = None

    def fit(self, features: np.ndarray) -> None:
        feats = torch.from_numpy(features.astype(np.float32))
        self.features_t = feats.to(self.device)

    def run(self, k: int, query: np.ndarray, index=None) -> Tuple[np.ndarray, np.ndarray]:
        if self.features_t is None:
            raise RuntimeError("Index not fitted.")
        feats = self.features_t
        q = torch.from_numpy(query.astype(np.float32)).to(self.device)
        num_query = q.shape[0]
        batch_size = 1024
        all_dists = []
        all_indices = []
        feats_norm = (feats ** 2).sum(dim=1)
        for start in range(0, num_query, batch_size):
            end = min(num_query, start + batch_size)
            q_batch = q[start:end]
            q_norm = (q_batch ** 2).sum(dim=1, keepdim=True)
            dist_sq = q_norm + feats_norm.unsqueeze(0) - 2.0 * q_batch @ feats.t()
            dist_sq = torch.clamp(dist_sq, min=0.0)
            dist = torch.sqrt(dist_sq)
            if k >= dist.shape[1]:
                d_k, idx_k = torch.sort(dist, dim=1)
            else:
                d_k, idx_k = torch.topk(dist, k, dim=1, largest=False)
            all_dists.append(d_k.cpu().numpy())
            all_indices.append(idx_k.cpu().numpy())
        dists = np.concatenate(all_dists, axis=0)
        indices = np.concatenate(all_indices, axis=0)
        return dists, indices

    def save(self, path: str) -> None:
        if self.features_t is None:
            return
        feats = self.features_t.detach().cpu().numpy()
        np.save(path, feats)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            alt = path + ".npy"
            if not os.path.exists(alt):
                raise FileNotFoundError(path)
            path = alt
        feats = np.load(path)
        self.features_t = torch.from_numpy(feats.astype(np.float32)).to(self.device)

    def reset_index(self) -> None:
        self.features_t = None


class ConcatMerger:
    def merge(self, features: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(features, axis=1)


class NearestNeighbourScorer:
    def __init__(self, n_nearest_neighbours: int, nn_method: FaissNN) -> None:
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method
        self.imagelevel_nn = lambda query: self.nn_method.run(self.n_nearest_neighbours, query)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

    def predict(self, query_features: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _index_file(folder: str, prepend: str = "") -> str:
        return os.path.join(folder, prepend + "nn_index.faiss")

    def save(self, save_folder: str, prepend: str = "") -> None:
        index_path = self._index_file(save_folder, prepend)
        self.nn_method.save(index_path)

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        index_path = self._index_file(load_folder, prepend)
        self.nn_method.load(index_path)
