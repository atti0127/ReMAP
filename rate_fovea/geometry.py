"""Parameter-free spatial-moment fovea extraction and alignment."""

import math

import torch
import torch.nn.functional as F


def moment_geometry(patch_scores: torch.Tensor, sigma_span: float = 2.0):
    """Return center and weighted-standard-deviation extent in [-1, 1]."""

    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [B,N]")
    side = int(patch_scores.shape[1] ** 0.5)
    if side * side != patch_scores.shape[1]:
        raise ValueError("patch_scores must form a square grid")
    axis = torch.linspace(-1.0, 1.0, side, device=patch_scores.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = xx.flatten(); y = yy.flatten()
    standardized = (
        patch_scores.float() - patch_scores.float().mean(dim=1, keepdim=True)
    ) / patch_scores.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    # Extreme-value normalization makes the spatial measure focus on the
    # statistically exceptional tail without a chosen percentile/threshold.
    concentration = math.sqrt(2.0 * math.log(patch_scores.shape[1]))
    weights = (concentration * standardized).softmax(dim=1)
    center_x = (weights * x).sum(dim=1)
    center_y = (weights * y).sum(dim=1)
    variance_x = (weights * (x - center_x[:, None]).square()).sum(dim=1)
    variance_y = (weights * (y - center_y[:, None]).square()).sum(dim=1)
    # Two weighted standard deviations cover the evidence support. Uniform or
    # diffuse evidence naturally falls back to the full image.
    half_extent = sigma_span * torch.maximum(variance_x, variance_y).sqrt()
    half_extent = half_extent.clamp(min=2.0 / side, max=1.0)
    center_x = center_x.clamp(-1.0 + half_extent, 1.0 - half_extent)
    center_y = center_y.clamp(-1.0 + half_extent, 1.0 - half_extent)
    return torch.stack((center_x, center_y, half_extent), dim=1)


def anisotropic_moment_geometry(
    patch_scores: torch.Tensor, sigma_span: float = 2.0
):
    """Return center and independent x/y evidence extents in [-1, 1]."""

    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [B,N]")
    side = int(patch_scores.shape[1] ** 0.5)
    if side * side != patch_scores.shape[1]:
        raise ValueError("patch_scores must form a square grid")
    axis = torch.linspace(-1.0, 1.0, side, device=patch_scores.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = xx.flatten(); y = yy.flatten()
    standardized = (
        patch_scores.float() - patch_scores.float().mean(dim=1, keepdim=True)
    ) / patch_scores.float().std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    concentration = math.sqrt(2.0 * math.log(patch_scores.shape[1]))
    weights = (concentration * standardized).softmax(dim=1)
    center_x = (weights * x).sum(dim=1)
    center_y = (weights * y).sum(dim=1)
    variance_x = (weights * (x - center_x[:, None]).square()).sum(dim=1)
    variance_y = (weights * (y - center_y[:, None]).square()).sum(dim=1)
    extent_x = (sigma_span * variance_x.sqrt()).clamp(2.0 / side, 1.0)
    extent_y = (sigma_span * variance_y.sqrt()).clamp(2.0 / side, 1.0)
    center_x = center_x.clamp(-1.0 + extent_x, 1.0 - extent_x)
    center_y = center_y.clamp(-1.0 + extent_y, 1.0 - extent_y)
    return torch.stack((center_x, center_y, extent_x, extent_y), dim=1)


def extraction_grid(geometry: torch.Tensor, output_size: int):
    axis = torch.linspace(-1.0, 1.0, output_size, device=geometry.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if geometry.shape[1] == 3:
        center_x, center_y, extent_x = geometry.unbind(dim=1)
        extent_y = extent_x
    elif geometry.shape[1] == 4:
        center_x, center_y, extent_x, extent_y = geometry.unbind(dim=1)
    else:
        raise ValueError("geometry must contain center x/y and one or two extents")
    grid_x = center_x[:, None, None] + extent_x[:, None, None] * xx
    grid_y = center_y[:, None, None] + extent_y[:, None, None] * yy
    return torch.stack((grid_x, grid_y), dim=-1)


def extract_center_fovea(image: torch.Tensor, output_size: int):
    """Extract the exact centered square used by the compute-matched control."""

    if image.ndim != 4 or image.shape[-2] != image.shape[-1]:
        raise ValueError("center control requires square images with shape [B,C,H,H]")
    image_size = image.shape[-1]
    if not 1 < output_size <= image_size:
        raise ValueError("output_size must be in [2, image_size]")
    # With align_corners=True, this extent samples exactly ``output_size``
    # contiguous source pixels while retaining the same output tensor shape
    # and encoder cost as Affinity-Propagated Re-observation.
    extent = (output_size - 1) / (image_size - 1)
    geometry = torch.zeros(
        (image.shape[0], 3), device=image.device, dtype=image.dtype
    )
    geometry[:, 2] = extent
    crop = F.grid_sample(
        image, extraction_grid(geometry, output_size), mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return crop, geometry


def extract_moment_fovea(image: torch.Tensor, patch_scores: torch.Tensor,
                         output_size: int = 336, sigma_span: float = 2.0,
                         anisotropic: bool = False):
    geometry_function = (
        anisotropic_moment_geometry if anisotropic else moment_geometry
    )
    geometry = geometry_function(patch_scores, sigma_span=sigma_span)
    crop = F.grid_sample(
        image, extraction_grid(geometry, output_size), mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return crop, geometry
