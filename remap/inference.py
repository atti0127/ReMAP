"""Final single-pass ReMAP anomaly-map computation."""

import numpy as np
import torch
import torch.nn.functional as F

import AnomalyCLIP_lib
from rate import adapt_rate, make_pyramid_auxiliary_features
from rate.core import _percentile_ranks, histogram_transport
from rate_fovea import extract_moment_fovea
from rate_fovea.evaluate import (
    dense_histogram_transport,
    inverse_crop_grid,
    positive_feature_support,
)
from rate_prompt.cache_text_bank import _prototype, bank_states
from remap.runtime import gaussian_filter2d


def fixed_prompt_pair(model, class_name, bank, device):
    """Encode fixed normal and anomalous descriptions for one class."""

    readable = class_name.replace("_", " ")
    states = bank_states(bank, class_name)
    normal = _prototype(
        model, (prompt.format(readable) for prompt in states["normal"]), device
    )
    anomaly = _prototype(
        model, (prompt.format(readable) for prompt in states["anomaly"]), device
    )
    return torch.stack((normal, anomaly))


def _anomaly_scores(features, prototypes):
    similarity, _ = AnomalyCLIP_lib.compute_similarity(features, prototypes)
    return similarity[..., 1]


@torch.no_grad()
def remap_anomaly_map(
    model,
    learned_text,
    route_pair,
    crop_pair,
    image,
    image_size=518,
    auxiliary_size=224,
    crop_size=252,
):
    """Return the final dense ReMAP anomaly map for a batch of images."""

    _, identity_layers = model.encode_image(image, [24], DPAM_layer=20)
    identity = F.normalize(identity_layers[0][:, 1:, :].float(), dim=-1)
    official = _anomaly_scores(identity, learned_text[0])

    auxiliary_features = make_pyramid_auxiliary_features(
        identity,
        auxiliary_side=auxiliary_size // 14,
        interpolation="bicubic",
    )
    rate = adapt_rate(
        learned_text[0],
        identity,
        auxiliary_features,
        official,
        candidate_evaluation="batched",
        inference_only=True,
    ).transported_patch_scores

    route_score = ((identity @ route_pair.t()) / 0.07).softmax(-1)[..., 1]
    geometry_key = torch.sqrt(
        _percentile_ranks(rate) * _percentile_ranks(route_score)
    )
    crop_image, geometry = extract_moment_fovea(
        image, geometry_key, output_size=crop_size, sigma_span=2.0
    )
    _, crop_layers = model.encode_image(crop_image, [24], DPAM_layer=20)
    crop_features = F.normalize(crop_layers[-1][:, 1:, :].float(), dim=-1)
    crop_native = _anomaly_scores(crop_features, learned_text[0])
    crop_semantic = ((crop_features @ crop_pair.t()) / 0.07).softmax(-1)[..., 1]
    crop_ordering = (
        _percentile_ranks(crop_native) + _percentile_ranks(crop_semantic)
    )
    navigated_crop = histogram_transport(crop_native, crop_ordering)
    supported_crop = positive_feature_support(
        navigated_crop, crop_features, steps=4
    )

    token_side = int(official.shape[1] ** 0.5)
    crop_side = int(navigated_crop.shape[1] ** 0.5)
    base_dense = F.interpolate(
        official.reshape(-1, 1, token_side, token_side),
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    rate_dense = F.interpolate(
        rate.reshape(-1, 1, token_side, token_side),
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    dense_grid, dense_valid = inverse_crop_grid(
        geometry, image_size, image.device
    )
    supported_dense = F.grid_sample(
        supported_crop.reshape(-1, 1, crop_side, crop_side),
        dense_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    base_logit = torch.logit(base_dense.clamp(1e-4, 1.0 - 1e-4))
    supported_logit = torch.logit(supported_dense.clamp(1e-4, 1.0 - 1e-4))
    crop_footprint = geometry[:, 2] * image_size / crop_side
    crop_precision = crop_footprint.square().reciprocal()
    base_precision = (image_size / token_side) ** -2
    crop_weight = crop_precision / (crop_precision + base_precision)
    supported_delta = (
        dense_valid
        * crop_weight[:, None, None, None]
        * F.relu(supported_logit - base_logit)
    )
    crop_enhanced = torch.sigmoid(
        torch.logit(rate_dense.clamp(1e-4, 1.0 - 1e-4)) + supported_delta
    )

    seed = gaussian_filter2d(crop_enhanced.float(), sigma=4.0)
    token_seed = F.interpolate(
        seed, (token_side, token_side), mode="bilinear", align_corners=False
    ).flatten(1)
    closed = positive_feature_support(
        token_seed, identity, steps=1
    ).reshape(-1, 1, token_side, token_side)
    token_grid, token_valid = inverse_crop_grid(
        geometry, token_side, image.device
    )
    bounded = token_seed.reshape_as(closed) + token_valid * F.relu(
        closed - token_seed.reshape_as(closed)
    )
    semantic_token = F.grid_sample(
        crop_semantic.reshape(-1, 1, crop_side, crop_side),
        token_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).flatten(1)
    base_rank = _percentile_ranks(token_seed)
    persistent_rank = (
        base_rank
        * _percentile_ranks(bounded.flatten(1))
        * _percentile_ranks(semantic_token)
    ).clamp_min(0.0).pow(1.0 / 3.0)
    bounded_rank = base_rank + token_valid.flatten(1) * F.relu(
        persistent_rank - base_rank
    )
    ordering = F.interpolate(
        bounded_rank.reshape(-1, 1, token_side, token_side),
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return dense_histogram_transport(base_dense, ordering)[:, 0].float().cpu().numpy()

