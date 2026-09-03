"""Core rank aggregation and exact evidence transport for RATE."""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RATETransportResult:
    adapted_text_features: torch.Tensor
    pixel_gate: float
    image_gate: float
    diagnostics: Dict[str, float]
    pixel_token_gates: Optional[torch.Tensor] = None
    transported_patch_scores: Optional[torch.Tensor] = None


def _square_side(token_count: int, name: str) -> int:
    side = int(token_count ** 0.5)
    if side * side != token_count:
        raise ValueError("{} token count {} is not a square grid".format(name, token_count))
    return side


def make_auxiliary_view(image: torch.Tensor, size: int) -> torch.Tensor:
    """Create RATE's sole auxiliary view from a normalized image tensor."""

    if image.ndim != 4:
        raise ValueError("image must have shape [B, C, H, W]")
    flipped = torch.flip(image, dims=(-1,))
    try:
        return F.interpolate(
            flipped,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    except TypeError:  # PyTorch versions predating antialias support.
        return F.interpolate(flipped, size=(size, size), mode="bicubic", align_corners=False)


def align_auxiliary_patch_features(
    auxiliary_features: torch.Tensor,
    target_token_count: int,
) -> torch.Tensor:
    """Undo the horizontal flip and lift auxiliary features to the target grid.

    ``auxiliary_features`` must not contain a CLS token.
    """

    if auxiliary_features.ndim != 3:
        raise ValueError("auxiliary_features must have shape [B, N, D]")
    low_side = _square_side(auxiliary_features.shape[1], "auxiliary")
    high_side = _square_side(target_token_count, "target")
    batch, _, dim = auxiliary_features.shape
    grid = auxiliary_features.reshape(batch, low_side, low_side, dim)
    grid = torch.flip(grid, dims=(2,))
    grid = grid.permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(high_side, high_side), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(batch, target_token_count, dim)
    return F.normalize(grid, dim=-1)


def make_pyramid_auxiliary_features(
    identity_features: torch.Tensor,
    auxiliary_side: int = 16,
    interpolation: str = "area",
) -> torch.Tensor:
    """Construct RATE's flipped low-resolution view in feature space.

    This mirrors ``make_auxiliary_view`` without another encoder call: flip
    the normalized identity grid and pool it to the auxiliary token lattice.
    RATE's existing alignment subsequently undoes the flip.
    """

    if identity_features.ndim != 3:
        raise ValueError("identity_features must have shape [B,N,D]")
    identity_side = _square_side(identity_features.shape[1], "identity")
    if auxiliary_side < 2 or auxiliary_side > identity_side:
        raise ValueError("auxiliary_side must be in [2, identity_side]")
    if interpolation not in ("area", "bilinear", "bicubic"):
        raise ValueError("interpolation must be area, bilinear, or bicubic")
    batch, _, dim = identity_features.shape
    grid = F.normalize(identity_features.detach().float(), dim=-1).reshape(
        batch, identity_side, identity_side, dim
    )
    grid = torch.flip(grid, dims=(2,)).permute(0, 3, 1, 2)
    interpolate_args = {
        "size": (auxiliary_side, auxiliary_side), "mode": interpolation
    }
    if interpolation in ("bilinear", "bicubic"):
        interpolate_args.update(align_corners=False, antialias=True)
    pooled = F.interpolate(grid, **interpolate_args)
    return F.normalize(
        pooled.permute(0, 2, 3, 1).reshape(
            batch, auxiliary_side * auxiliary_side, dim
        ),
        dim=-1,
    )


def _prototype_pair(
    base_text_features: torch.Tensor,
    residual: torch.Tensor,
    basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Restrict the update to the tangent complement of the source prompt pair.
    if basis is None:
        basis = torch.linalg.qr(
            base_text_features.transpose(0, 1), mode="reduced"
        ).Q
    tangent = residual - basis @ (basis.transpose(0, 1) @ residual)
    normal = F.normalize(base_text_features[0] - tangent, dim=0)
    anomaly = F.normalize(base_text_features[1] + tangent, dim=0)
    return torch.stack((normal, anomaly), dim=0)


def _binary_logits(features: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    return (features @ prototypes[1] - features @ prototypes[0]) / temperature


def _relative_improvement(before: torch.Tensor, after: torch.Tensor, eps: float) -> float:
    value = ((before - after) / before.clamp_min(eps)).detach().item()
    return max(0.0, float(value))


def _percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ranks in [0, 1] for each row."""

    if values.ndim != 2:
        raise ValueError("rank values must have shape [B, N]")
    order = torch.argsort(values, dim=-1, stable=True)
    positions = torch.arange(
        values.shape[1], device=values.device, dtype=values.dtype
    ) / max(1, values.shape[1] - 1)
    ranks = torch.empty_like(values)
    ranks.scatter_(1, order, positions.unsqueeze(0).expand_as(values))
    return ranks


def _standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=-1, keepdim=True)).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return centered / scale


def _rank_transport_terms(
    residual: torch.Tensor,
    base_text: torch.Tensor,
    high_patch: torch.Tensor,
    auxiliary_teacher_margin: torch.Tensor,
    base_basis: Optional[torch.Tensor] = None,
    standardized_auxiliary: Optional[torch.Tensor] = None,
    auxiliary_ranks: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    prototypes = _prototype_pair(base_text, residual, basis=base_basis)
    # Temperature, shifts, and positive scaling do not affect ranks. Raw
    # binary margins make the differentiable surrogate scale-free.
    high_margin = _binary_logits(high_patch, prototypes, 1.0)
    if standardized_auxiliary is None:
        standardized_auxiliary = _standardize(auxiliary_teacher_margin)
    surrogate = (
        _standardize(high_margin) - standardized_auxiliary
    ).square().mean()
    high_ranks = _percentile_ranks(high_margin)
    if auxiliary_ranks is None:
        auxiliary_ranks = _percentile_ranks(auxiliary_teacher_margin)
    rank_disagreement = (high_ranks - auxiliary_ranks).square().mean()
    return {
        "surrogate": surrogate,
        "rank_disagreement": rank_disagreement,
        "high_margin": high_margin,
        "high_ranks": high_ranks,
        "auxiliary_ranks": auxiliary_ranks,
        "prototypes": prototypes,
    }


def histogram_transport(
    frozen_scores: torch.Tensor,
    target_order_values: torch.Tensor,
) -> torch.Tensor:
    """Assign the original frozen score values using a target spatial order."""

    if frozen_scores.shape != target_order_values.shape or frozen_scores.ndim != 2:
        raise ValueError("transport inputs must have matching shape [B, N]")
    target_order = torch.argsort(target_order_values, dim=-1, stable=True)
    sorted_frozen = torch.sort(frozen_scores, dim=-1, stable=True).values
    transported = torch.empty_like(frozen_scores)
    transported.scatter_(1, target_order, sorted_frozen)
    return transported


def _cross_scale_rank_transport(
    base_text_features: torch.Tensor,
    high_patch_features: torch.Tensor,
    auxiliary_patch_features: torch.Tensor,
    frozen_patch_scores: Optional[torch.Tensor] = None,
    candidate_evaluation: str = "batched",
    inference_only: bool = False,
) -> RATETransportResult:
    """Hyperparameter-free, histogram-preserving patch-rank transport.

    Candidate generation uses a differentiable standardized-margin surrogate.
    Acceptance uses exact percentile ranks: cross-scale rank disagreement must
    strictly improve, prototype semantics may not swap, and the identity-view
    rank displacement may not exceed the rank uncertainty already observed
    between views. Accepted scores are a permutation of the frozen official
    scores, preventing probability-mass collapse by construction.
    """

    if base_text_features.ndim != 2 or base_text_features.shape[0] != 2:
        raise ValueError("base_text_features must have shape [2, D]")
    if high_patch_features.ndim != 3 or auxiliary_patch_features.ndim != 3:
        raise ValueError("patch features must have shape [B, N, D]")
    if high_patch_features.shape[0] != 1:
        raise ValueError("RATE cross-scale transport currently requires batch size 1")
    if candidate_evaluation not in ("sequential", "batched"):
        raise ValueError("candidate_evaluation must be sequential or batched")

    eps = torch.finfo(torch.float32).eps
    base_text = F.normalize(base_text_features.detach().float(), dim=-1)
    high_patch = F.normalize(high_patch_features.detach().float(), dim=-1)
    auxiliary_patch = F.normalize(auxiliary_patch_features.detach().float(), dim=-1)
    aligned_auxiliary = align_auxiliary_patch_features(auxiliary_patch, high_patch.shape[1])

    residual = torch.zeros(
        base_text.shape[-1], device=base_text.device, dtype=base_text.dtype, requires_grad=True
    )
    auxiliary_teacher_margin = _binary_logits(aligned_auxiliary, base_text, 1.0).detach()
    base_basis = torch.linalg.qr(
        base_text.transpose(0, 1), mode="reduced"
    ).Q
    standardized_auxiliary = _standardize(auxiliary_teacher_margin)
    auxiliary_ranks = _percentile_ranks(auxiliary_teacher_margin)
    with torch.enable_grad():
        initial = _rank_transport_terms(
            residual, base_text, high_patch, auxiliary_teacher_margin,
            base_basis=base_basis,
            standardized_auxiliary=standardized_auxiliary,
            auxiliary_ranks=auxiliary_ranks,
        )
        gradient = torch.autograd.grad(initial["surrogate"], residual)[0]

    initial_rank_disagreement = initial["rank_disagreement"].detach()
    rank_uncertainty = torch.sqrt(initial_rank_disagreement)
    best = {key: value.detach() for key, value in initial.items()}
    best_residual = torch.zeros_like(residual.detach())
    best_rank_shift = torch.zeros((), device=base_text.device, dtype=base_text.dtype)
    best_rank_objective = initial_rank_disagreement
    accepted = False

    gradient_norm = gradient.norm()
    initial_scalars = torch.stack((
        gradient_norm, initial_rank_disagreement
    )).detach().cpu().tolist()
    rank_uncertainty_value = math.sqrt(initial_scalars[1])
    best_rank_objective_value = initial_scalars[1]
    if initial_scalars[0] > eps and initial_scalars[1] > 0:
        direction = -gradient.detach() / gradient_norm
        maximum_step = 0.5 * (base_text[1] - base_text[0]).norm()
        with torch.no_grad():
            step_exponents = torch.arange(
                20, device=base_text.device, dtype=base_text.dtype
            )
            step_sizes = maximum_step / torch.pow(2.0, step_exponents)
            if candidate_evaluation == "sequential":
                for step_size in step_sizes:
                    candidate_residual = direction * step_size
                    candidate = _rank_transport_terms(
                        candidate_residual,
                        base_text,
                        high_patch,
                        auxiliary_teacher_margin,
                        base_basis=base_basis,
                        standardized_auxiliary=standardized_auxiliary,
                        auxiliary_ranks=auxiliary_ranks,
                    )
                    candidate_shift = torch.sqrt(
                        (
                            candidate["high_ranks"] - initial["high_ranks"]
                        ).square().mean()
                    )
                    candidate_objective = (
                        candidate["rank_disagreement"]
                        + candidate_shift.square()
                    )
                    normal_margin = (
                        candidate["prototypes"][0] * base_text[0]
                    ).sum() - (
                        candidate["prototypes"][0] * base_text[1]
                    ).sum()
                    anomaly_margin = (
                        candidate["prototypes"][1] * base_text[1]
                    ).sum() - (
                        candidate["prototypes"][1] * base_text[0]
                    ).sum()
                    row = torch.stack((
                        candidate["rank_disagreement"],
                        candidate_shift,
                        normal_margin,
                        anomaly_margin,
                        candidate_objective,
                    )).cpu().tolist()
                    if (
                        row[0] < initial_scalars[1]
                        and row[1] > 0
                        and row[1] <= rank_uncertainty_value + eps
                        and row[2] >= 0
                        and row[3] >= 0
                        and row[4] < best_rank_objective_value
                    ):
                        best_rank_objective = candidate_objective
                        best_rank_objective_value = row[4]
                        best_residual = candidate_residual
                        best_rank_shift = candidate_shift
                        accepted = True
                        best = {
                            key: value.detach() for key, value in candidate.items()
                        }
            else:
                candidate_residuals = direction[None] * step_sizes[:, None]
                tangent = candidate_residuals - (
                    candidate_residuals @ base_basis
                ) @ base_basis.transpose(0, 1)
                candidate_normal = F.normalize(base_text[0][None] - tangent, dim=-1)
                candidate_anomaly = F.normalize(base_text[1][None] + tangent, dim=-1)
                candidate_prototypes = torch.stack(
                    (candidate_normal, candidate_anomaly), dim=1
                )
                candidate_margins = torch.einsum(
                    "nd,kd->kn", high_patch[0], candidate_anomaly - candidate_normal
                )
                candidate_surrogates = (
                    _standardize(candidate_margins) - standardized_auxiliary
                ).square().mean(dim=1)
                candidate_ranks = _percentile_ranks(candidate_margins)
                candidate_disagreements = (
                    candidate_ranks - auxiliary_ranks
                ).square().mean(dim=1)
                candidate_shifts = torch.sqrt(
                    (candidate_ranks - initial["high_ranks"]).square().mean(dim=1)
                )
                # Equal-weight squared distances produce the intrinsic Fréchet
                # compromise. Evaluate the same analytic candidate set and
                # decision rules in one batch instead of 20 sequential launches.
                candidate_objectives = (
                    candidate_disagreements + candidate_shifts.square()
                )
                candidate_scalar_rows = torch.stack((
                    candidate_disagreements,
                    candidate_shifts,
                    (candidate_normal * base_text[0]).sum(dim=1)
                    - (candidate_normal * base_text[1]).sum(dim=1),
                    (candidate_anomaly * base_text[1]).sum(dim=1)
                    - (candidate_anomaly * base_text[0]).sum(dim=1),
                    candidate_objectives,
                ), dim=1).cpu().tolist()
                feasible_indices = [
                    index for index, row in enumerate(candidate_scalar_rows)
                    if (
                        row[0] < initial_scalars[1]
                        and row[1] > 0
                        and row[1] <= rank_uncertainty_value + eps
                        and row[2] >= 0
                        and row[3] >= 0
                    )
                ]
                if feasible_indices:
                    best_index = min(
                        feasible_indices,
                        key=lambda index: candidate_scalar_rows[index][4],
                    )
                    if candidate_scalar_rows[best_index][4] < best_rank_objective_value:
                        best_rank_objective = candidate_objectives[best_index]
                        best_rank_objective_value = candidate_scalar_rows[best_index][4]
                        best_residual = candidate_residuals[best_index]
                        best_rank_shift = candidate_shifts[best_index]
                        accepted = True
                        best = {
                            "surrogate": candidate_surrogates[best_index],
                            "rank_disagreement": candidate_disagreements[best_index],
                            "high_margin": candidate_margins[best_index:best_index + 1],
                            "high_ranks": candidate_ranks[best_index:best_index + 1],
                            "auxiliary_ranks": auxiliary_ranks,
                            "prototypes": candidate_prototypes[best_index],
                        }

    if frozen_patch_scores is None:
        frozen_scores = ((high_patch @ base_text.t()) / 0.07).softmax(dim=-1)[..., 1]
    else:
        if frozen_patch_scores.shape != high_patch.shape[:2]:
            raise ValueError("frozen_patch_scores must have shape [B, N]")
        frozen_scores = frozen_patch_scores.detach().float()
    transported_scores = None
    if accepted:
        transported_scores = histogram_transport(frozen_scores, best["high_margin"])
    if inference_only:
        return RATETransportResult(
            best["prototypes"] if accepted else base_text,
            0.0,
            0.0,
            {},
            pixel_token_gates=None,
            transported_patch_scores=transported_scores,
        )

    transport_fraction = 0.0
    histogram_max_error = 0.0
    if transported_scores is not None:
        histogram_max_error = float(
            (
                torch.sort(transported_scores, dim=-1).values
                - torch.sort(frozen_scores, dim=-1).values
            ).abs().max().item()
        )
        changed_scores = transported_scores != frozen_scores
        transport_fraction = float(changed_scores.float().mean().item())
        if transport_fraction == 0.0:
            accepted = False
            transported_scores = None

    rank_improvement = _relative_improvement(
        initial_rank_disagreement, best["rank_disagreement"], eps
    )
    total_improvement = _relative_improvement(
        initial_rank_disagreement, best_rank_objective, eps
    )
    feature_cosine = (high_patch * aligned_auxiliary).sum(dim=-1)
    correspondence = ((feature_cosine + 1.0) * 0.5).clamp(0.0, 1.0)
    diagnostics = {
        "accepted": float(accepted),
        "pixel_gate": transport_fraction,
        "image_gate": 0.0,
        "initial_loss": float(initial_rank_disagreement.item()),
        "adapted_loss": float(best_rank_objective.item()),
        "total_relative_improvement": total_improvement,
        "pixel_relative_improvement": rank_improvement,
        "image_relative_improvement": 0.0,
        "distribution_drift": 0.0,
        "stable_probability_drift": float(best_rank_shift.item()),
        "residual_norm": float(best_residual.norm().item()),
        "mean_correspondence": float(correspondence.mean().item()),
        "image_concordant": 0.0,
        "rank_uncertainty": float(rank_uncertainty.item()),
        "rank_shift": float(best_rank_shift.item()),
        "adapted_rank_disagreement": float(best["rank_disagreement"].item()),
        "transport_fraction": transport_fraction,
        "histogram_max_error": histogram_max_error,
        "candidate_evaluation_batched": float(
            candidate_evaluation == "batched"
        ),
    }
    adapted_text = best["prototypes"] if accepted else base_text
    return RATETransportResult(
        adapted_text,
        transport_fraction,
        0.0,
        diagnostics,
        pixel_token_gates=None,
        transported_patch_scores=transported_scores,
    )


def _nonlocal_rarity_ranks(
    patch_features: torch.Tensor,
    neighbor_count: int = 5,
    exclusion_radius: int = 2,
) -> torch.Tensor:
    """Estimate visual rarity from nonlocal feature repetition."""

    if patch_features.ndim != 3:
        raise ValueError("patch_features must have shape [B, N, D]")
    side = _square_side(patch_features.shape[1], "patch")
    normalized = F.normalize(patch_features.detach().float(), dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(side, device=similarity.device),
            torch.arange(side, device=similarity.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    local = (
        (coordinates[:, None] - coordinates[None, :]).abs().amax(dim=-1)
        <= exclusion_radius
    )
    similarity = similarity.masked_fill(local.unsqueeze(0), -torch.inf)
    repetition = similarity.topk(neighbor_count, dim=-1).values.mean(dim=-1)
    return 1.0 - _percentile_ranks(repetition)


def local_affinity_weights(
    patch_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Return frozen 3x3 row-stochastic local feature affinities."""

    side = _square_side(patch_features.shape[1], "patch")
    normalized = F.normalize(patch_features.detach().float(), dim=-1)
    feature_grid = normalized.transpose(1, 2).reshape(
        normalized.shape[0], normalized.shape[2], side, side
    )
    padded_features = F.pad(feature_grid, (1, 1, 1, 1), mode="replicate")
    feature_neighbors = F.unfold(padded_features, kernel_size=3).reshape(
        normalized.shape[0], normalized.shape[2], 9, -1
    )
    center = normalized.transpose(1, 2).unsqueeze(2)
    cosine = (feature_neighbors * center).sum(dim=1)
    return (cosine / temperature).softmax(dim=1)


def apply_local_affinity_weights(
    scores: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Propagate patch scores through cached 3x3 feature affinities."""

    side = _square_side(scores.shape[1], "score")
    score_grid = scores.reshape(scores.shape[0], 1, side, side)
    padded_scores = F.pad(score_grid, (1, 1, 1, 1), mode="replicate")
    score_neighbors = F.unfold(padded_scores, kernel_size=3).reshape(
        scores.shape[0], 9, -1
    )
    if weights.shape != score_neighbors.shape:
        raise ValueError("weights must have shape [B, 9, N]")
    return (weights * score_neighbors).sum(dim=1)


def _local_affinity_step(
    scores: torch.Tensor,
    patch_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    weights = local_affinity_weights(patch_features, temperature)
    return apply_local_affinity_weights(scores, weights)


def adapt_rank_aggregation_transport(
    base_text_features: torch.Tensor,
    high_patch_features: torch.Tensor,
    auxiliary_patch_features: torch.Tensor,
    frozen_patch_scores: torch.Tensor,
    voters: Tuple[str, ...] = (
        "cross_scale_affinity_rank",
        "rarity_corrected_rank",
        "extrapolated_fusion_rank",
    ),
    candidate_evaluation: str = "batched",
    inference_only: bool = False,
) -> RATETransportResult:
    """Aggregate RATE's three ranks and move the frozen score values.

    Cross-scale Affinity Rank propagates cross-scale evidence through local
    feature affinities, Rarity-Corrected Rank discounts repeated nonlocal
    patterns, and Extrapolated Fusion Rank combines the cross-scale and rarity
    orderings. Equal rank aggregation adds no learned weights, while histogram
    transport preserves the original per-image scores exactly. ``voters`` is
    exposed only for the leave-one-rank-out ablation.
    """

    valid_voters = {
        "cross_scale_affinity_rank",
        "rarity_corrected_rank",
        "extrapolated_fusion_rank",
    }
    voters = tuple(voters)
    if not voters or len(set(voters)) != len(voters):
        raise ValueError("RATE voters must be non-empty and unique")
    unknown_voters = set(voters) - valid_voters
    if unknown_voters:
        raise ValueError(f"unknown RATE voters: {sorted(unknown_voters)}")
    if frozen_patch_scores.shape != high_patch_features.shape[:2]:
        raise ValueError("frozen_patch_scores must have shape [B, N]")
    base = frozen_patch_scores.detach().float()
    cross_scale = _cross_scale_rank_transport(
        base_text_features,
        high_patch_features,
        auxiliary_patch_features,
        frozen_patch_scores=base,
        candidate_evaluation=candidate_evaluation,
        inference_only=inference_only,
    )
    v3 = (
        cross_scale.transported_patch_scores
        if cross_scale.transported_patch_scores is not None else base
    )
    base_rank = _percentile_ranks(base)
    v3_rank = _percentile_ranks(v3)
    rarity_rank = _nonlocal_rarity_ranks(high_patch_features)
    weights = local_affinity_weights(high_patch_features)

    cross_scale_affinity = v3
    for _ in range(4):
        propagated = apply_local_affinity_weights(cross_scale_affinity, weights)
        cross_scale_affinity = histogram_transport(base, propagated)

    conservative_key = v3_rank + v3_rank * (1.0 - v3_rank) * (rarity_rank - 0.5)
    rarity_corrected = histogram_transport(base, conservative_key)
    rarity_corrected = histogram_transport(
        base, apply_local_affinity_weights(rarity_corrected, weights)
    )

    extrapolated_v3 = base_rank + 2.0 * (v3_rank - base_rank)
    extrapolated_fusion = histogram_transport(
        base, (base_rank + extrapolated_v3 + rarity_rank) / 3.0
    )
    for _ in range(4):
        propagated = apply_local_affinity_weights(extrapolated_fusion, weights)
        extrapolated_fusion = histogram_transport(base, propagated)

    candidates = {
        "cross_scale_affinity_rank": cross_scale_affinity,
        "rarity_corrected_rank": rarity_corrected,
        "extrapolated_fusion_rank": extrapolated_fusion,
    }
    aggregation_key = sum(
        (_percentile_ranks(candidates[name]) for name in voters),
        torch.zeros_like(base_rank),
    )
    transported = histogram_transport(base, aggregation_key)
    if inference_only:
        return RATETransportResult(
            F.normalize(base_text_features.detach().float(), dim=-1),
            0.0,
            0.0,
            {},
            transported_patch_scores=transported,
        )
    changed = (transported != base).float().mean()
    histogram_error = (
        torch.sort(transported, dim=-1).values
        - torch.sort(base, dim=-1).values
    ).abs().max()
    rank_shift = torch.sqrt(
        (_percentile_ranks(transported) - base_rank).square().mean()
    )
    diagnostics = dict(cross_scale.diagnostics)
    diagnostics.update({
        "accepted": 1.0,
        "pixel_gate": float(changed.item()),
        "image_gate": 0.0,
        "rank_shift": float(rank_shift.item()),
        "transport_fraction": float(changed.item()),
        "histogram_max_error": float(histogram_error.item()),
        "mean_rarity_rank": float(rarity_rank.mean().item()),
        "aggregation_branches": float(len(voters)),
        "voter_cross_scale_affinity_rank": float("cross_scale_affinity_rank" in voters),
        "voter_rarity_corrected_rank": float("rarity_corrected_rank" in voters),
        "voter_extrapolated_fusion_rank": float("extrapolated_fusion_rank" in voters),
    })
    return RATETransportResult(
        F.normalize(base_text_features.detach().float(), dim=-1),
        float(changed.item()),
        0.0,
        diagnostics,
        transported_patch_scores=transported,
    )


def patch_score_map(patch_scores: torch.Tensor, image_size: int) -> torch.Tensor:
    """Lift scalar patch anomaly scores to the output resolution."""

    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [B, N]")
    side = _square_side(patch_scores.shape[1], "patch score")
    score_map = patch_scores.reshape(patch_scores.shape[0], 1, side, side)
    score_map = F.interpolate(
        score_map, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return score_map[:, 0]
