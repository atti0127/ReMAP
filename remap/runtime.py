"""Shared frozen-model, filtering, and cache-alignment utilities for ReMAP."""

import torch
import torch.nn.functional as F

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner


_GAUSSIAN_KERNELS = {}
_REFLECT_INDICES = {}


def _gaussian_kernel1d(sigma, truncate, device, dtype):
    radius = int(truncate * sigma + 0.5)
    key = (float(sigma), float(truncate), device.type, device.index, dtype)
    kernel = _GAUSSIAN_KERNELS.get(key)
    if kernel is None:
        coordinates = torch.arange(
            -radius, radius + 1, device=device, dtype=torch.float64
        )
        kernel = torch.exp(-0.5 * coordinates.square() / (sigma * sigma))
        kernel = (kernel / kernel.sum()).to(dtype=dtype)
        _GAUSSIAN_KERNELS[key] = kernel
    return kernel


def _scipy_reflect_indices(length, radius, device):
    """Indices for SciPy ndimage's half-sample symmetric ``reflect`` mode."""

    key = (int(length), int(radius), device.type, device.index)
    indices = _REFLECT_INDICES.get(key)
    if indices is None:
        positions = torch.arange(-radius, length + radius, device=device)
        folded = positions.remainder(2 * length)
        indices = torch.where(folded < length, folded, 2 * length - 1 - folded)
        _REFLECT_INDICES[key] = indices
    return indices


def gaussian_filter2d(scores, sigma=4.0, truncate=4.0):
    """GPU/CPU tensor equivalent of SciPy's default 2-D Gaussian filter.

    The kernel radius and half-sample symmetric boundary follow
    ``scipy.ndimage.gaussian_filter``. Inputs and outputs have shape BCHW and
    stay on the input device, avoiding ReMAP's GPU-to-CPU-to-GPU round trip.
    """

    if scores.ndim != 4:
        raise ValueError("Gaussian input must have shape [B,C,H,W]")
    if not scores.is_floating_point():
        raise ValueError("Gaussian input must be floating point")
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel1d(
        sigma, truncate, scores.device, scores.dtype
    )
    channels = scores.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x_indices = _scipy_reflect_indices(scores.shape[-1], radius, scores.device)
    y_indices = _scipy_reflect_indices(scores.shape[-2], radius, scores.device)
    filtered = F.conv2d(
        scores.index_select(-1, x_indices), horizontal, groups=channels
    )
    return F.conv2d(
        filtered.index_select(-2, y_indices), vertical, groups=channels
    )


def load_frozen_model(args, device):
    """Load frozen AnomalyCLIP and its learned normal/anomaly prompts."""

    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=device, design_details=design
    )
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device).eval()
    model.to(device).eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    with torch.no_grad():
        prompts, tokens, compound = prompt_learner(cls_id=None)
        text = model.encode_text_learn(prompts, tokens, compound).float()
        text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
        text = F.normalize(text, dim=-1)
    return model, text


def canonical_path(path, class_name):
    """Make cached class-folder paths independent of their dataset root."""

    value = str(path).replace("\\", "/")
    marker = f"/{class_name}/"
    if marker in value:
        return class_name + "/" + value.split(marker, 1)[1]
    if value.startswith(class_name + "/"):
        return value
    raise ValueError(f"cannot canonicalize {path!r} for {class_name!r}")
