import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from .common_ref import (
    Aggregator,
    FaissNN,
    NetworkFeatureAggregator,
    Preprocessing,
    RescaleSegmentor,
    NearestNeighbourScorer,
)
from .sampler_ref import IdentitySampler


LOGGER = logging.getLogger(__name__)


class PatchMaker:
    def __init__(self, patchsize: int, stride: int):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded, number_of_total_patches
        return unfolded

    def unpatch_scores(self, x: np.ndarray, batchsize: int) -> np.ndarray:
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x: np.ndarray) -> np.ndarray:
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


class PatchCore(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def load(
        self,
        backbone: torch.nn.Module,
        layers_to_extract_from: List[str],
        device: torch.device,
        input_shape: Tuple[int, int, int],
        pretrain_embed_dimension: int,
        target_embed_dimension: int,
        patchsize: int = 3,
        patchstride: int = 1,
        anomaly_score_num_nn: int = 1,
        featuresampler: torch.nn.Module = IdentitySampler(),
        nn_method: FaissNN = FaissNN(False, 4),
        smoothing_sigma: float = 4.0,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
        _ = preprocessing.to(self.device)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device,
            target_size=input_shape[-2:],
            smoothing=smoothing_sigma,
        )
        self.featuresampler = featuresampler

    def _embed(self, images: torch.Tensor, detach: bool = True, provide_patch_shapes: bool = False):
        def _detach(features: List[torch.Tensor]):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features)):
            feat = features[i]
            patch_dims = patch_shapes[i]
            feat = feat.reshape(
                feat.shape[0],
                patch_dims[0],
                patch_dims[1],
                *feat.shape[2:],
            )
            feat = feat.permute(0, -3, -2, -1, 1, 2)
            base_shape = feat.shape
            feat = feat.reshape(-1, *feat.shape[-2:])
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            feat = feat.squeeze(1)
            feat = feat.reshape(*base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            feat = feat.permute(0, -2, -1, 1, 2, 3)
            feat = feat.reshape(len(feat), -1, *feat.shape[-3:])
            features[i] = feat
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image: torch.Tensor):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as iterator:
            for batch in iterator:
                images = batch["image"]
                features.append(_image_to_features(images))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict_with_raw(self, dataloader):
        _ = self.forward_modules.eval()
        scores = []
        raw_maps = []
        vis_maps = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as iterator:
            for batch in iterator:
                images = batch["image"]
                images = images.to(torch.float).to(self.device)
                batchsize = images.shape[0]
                with torch.no_grad():
                    features, patch_shapes = self._embed(images, provide_patch_shapes=True)
                    features = np.asarray(features)
                    patch_scores = self.anomaly_scorer.predict([features])[0]
                    image_scores = self.patch_maker.unpatch_scores(
                        patch_scores, batchsize=batchsize
                    )
                    image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                    image_scores = self.patch_maker.score(image_scores)
                    patch_scores_unpatched = self.patch_maker.unpatch_scores(
                        patch_scores, batchsize=batchsize
                    )
                    scales = patch_shapes[0]
                    patch_scores_unpatched = patch_scores_unpatched.reshape(
                        batchsize, scales[0], scales[1]
                    )
                    patch_scores_tensor = torch.from_numpy(patch_scores_unpatched).to(self.device)
                    raw = F.interpolate(
                        patch_scores_tensor.unsqueeze(1),
                        size=self.input_shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    raw_np = raw.detach().cpu().numpy()
                    masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores_unpatched)
                for s, r, m in zip(image_scores, raw_np, masks):
                    scores.append(s)
                    raw_maps.append(r)
                    vis_maps.append(m)
        return scores, raw_maps, vis_maps
