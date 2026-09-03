"""RATE: Rank Aggregation and Transport of Evidence.

This package exposes the global rank-transport component used by ReMAP.
"""

from .core import (
    RATETransportResult,
    adapt_rank_aggregation_transport,
    align_auxiliary_patch_features,
    histogram_transport,
    make_auxiliary_view,
    make_pyramid_auxiliary_features,
    patch_score_map,
)


def adapt_rate(
    base_text_features,
    high_patch_features,
    auxiliary_patch_features,
    frozen_patch_scores,
    voters=(
        "cross_scale_affinity_rank",
        "rarity_corrected_rank",
        "extrapolated_fusion_rank",
    ),
    candidate_evaluation="batched",
    inference_only=False,
):
    """Apply RATE's cross-scale rank aggregation and evidence transport."""

    return adapt_rank_aggregation_transport(
        base_text_features,
        high_patch_features,
        auxiliary_patch_features,
        frozen_patch_scores,
        voters=voters,
        candidate_evaluation=candidate_evaluation,
        inference_only=inference_only,
    )


__all__ = [
    "RATETransportResult",
    "adapt_rate",
    "align_auxiliary_patch_features",
    "histogram_transport",
    "make_auxiliary_view",
    "make_pyramid_auxiliary_features",
    "patch_score_map",
]
