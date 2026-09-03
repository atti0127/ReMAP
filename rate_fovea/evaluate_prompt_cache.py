"""Evaluate RATE and APR from aligned ReMAP caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rate.core import _percentile_ranks, histogram_transport
from rate_fovea.evaluate import (
    dense_histogram_transport, metrics, positive_feature_support,
    inverse_crop_grid, reconstruct_maps,
)


def select_full_seed_maps(semantic_maps, source):
    """Select the sole score seed for the bounded full-image affinity graph."""

    if source == "reobservation":
        selected = semantic_maps[6]
    elif source == "rate":
        selected = semantic_maps[3]
    else:
        raise ValueError(f"unknown full seed source: {source}")
    if selected is None:
        raise ValueError(f"{source} full-graph seed maps are unavailable")
    return selected


def main(args):
    cache = Path(args.cache)
    fovea = Path(args.fovea_cache)
    with (cache / "metadata.json").open() as handle:
        metadata = json.load(handle)
    with (fovea / "metadata.json").open() as handle:
        fovea_metadata = json.load(handle)
    if metadata["count"] != fovea_metadata["count"]:
        raise ValueError("RATE/fovea count mismatch")

    official = np.load(cache / "official_scores.npy", mmap_mode="r")
    branches = official[:, None, :]
    rate = np.load(
        args.rate_scores_file if args.rate_scores_file else cache / "rate_scores.npy",
        mmap_mode="r",
    )
    if rate.shape != official.shape:
        raise ValueError("RATE/official score shape mismatch")
    masks = np.load(cache / "masks.npy", mmap_mode="r")
    if args.crop_layers:
        crop_branches = np.load(
            fovea / "crop_branch_scores.npy", mmap_mode="r"
        )
        layer_names = fovea_metadata.get(
            "crop_branch_scores", ["layer_6", "layer_12", "layer_18", "layer_24"]
        )
        indices = [layer_names.index(f"layer_{layer}") for layer in args.crop_layers]
        crop = np.asarray(crop_branches[:, indices], dtype=np.float32).mean(axis=1)
    else:
        crop = np.load(fovea / "crop_scores.npy", mmap_mode="r")
    crop_features = np.load(fovea / "crop_features.npy", mmap_mode="r")
    geometry = np.load(fovea / "geometry.npy", mmap_mode="r")
    semantic_files = (
        args.crop_semantic_files
        if args.crop_semantic_files else [args.crop_semantic_file]
    )
    semantic_paths = [fovea / name for name in semantic_files]
    has_semantic = all(path.exists() for path in semantic_paths)
    variants = {}
    histogram_error = 0.0
    if not (args.final_only and has_semantic) and not args.paper_ablation:
        maps = reconstruct_maps(
            metadata, fovea_metadata, branches, crop, crop_features, geometry,
            rate, args.batch_size, args.workers, args.feature_steps,
            work_dir=args.work_dir, work_prefix="native",
            gaussian_backend=args.gaussian_backend,
        )
        (baseline, _, _, rate_maps, joint, _, feature_joint, _,
         histogram_error) = maps
        variants["rate"] = metrics(
            rate_maps, masks, metadata["records"], args.workers
        )
        if not args.final_only:
            variants.update({
                "official": metrics(
                    baseline, masks, metadata["records"], args.workers
                ),
                "rate_targeted_reobservation_native": metrics(
                    joint, masks, metadata["records"], args.workers
                ),
                "rate_targeted_reobservation_native_affinity": metrics(
                    feature_joint, masks, metadata["records"], args.workers
                ),
            })
    if has_semantic:
        semantics = [np.load(path, mmap_mode="r") for path in semantic_paths]
        navigated_crop = np.empty_like(crop, dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for start in range(0, metadata["count"], args.batch_size):
            stop = min(start + args.batch_size, metadata["count"])
            native_t = torch.from_numpy(
                np.array(crop[start:stop], dtype=np.float32, copy=True)
            ).to(device)
            native_rank = _percentile_ranks(native_t)
            ordering = native_rank
            for semantic in semantics:
                semantic_t = torch.from_numpy(
                    np.array(semantic[start:stop], dtype=np.float32, copy=True)
                ).to(device)
                semantic_rank = _percentile_ranks(semantic_t)
                if args.crop_navigation == "intersection":
                    ordering = torch.sqrt(ordering * semantic_rank)
                else:
                    ordering = ordering + semantic_rank
            navigated_crop[start:stop] = histogram_transport(
                native_t, ordering
            ).cpu().numpy()
        required_work_names = (
            ("baseline", "feature_joint")
            if args.promoted_only else
            ("baseline", "rate", "feature_joint")
        )
        reuse_work_dir = Path(args.reuse_work_dir or args.work_dir) if (
            args.reuse_work_dir or args.work_dir
        ) else None
        reusable = (
            args.reuse_work_maps and args.final_only and reuse_work_dir and
            all((reuse_work_dir / f"semantic_{name}.npy").exists()
                for name in required_work_names)
        )
        if reusable:
            work_dir = reuse_work_dir
            semantic_maps = (
                np.load(work_dir / "semantic_baseline.npy", mmap_mode="r"),
                None, None,
                (np.load(work_dir / "semantic_rate.npy", mmap_mode="r")
                 if (work_dir / "semantic_rate.npy").exists() else None),
                None, None,
                np.load(work_dir / "semantic_feature_joint.npy", mmap_mode="r"),
                None, None,
            )
            histogram_error = None
        else:
            semantic_maps = reconstruct_maps(
                metadata, fovea_metadata, branches, navigated_crop, crop_features,
                geometry, rate, args.batch_size, args.workers, args.feature_steps,
                minimal=args.final_only, work_dir=args.work_dir,
                work_prefix="semantic",
                direct_intermediate=args.direct_intermediate,
                store_rate=not args.promoted_only,
                component_ablation=args.paper_ablation,
                gaussian_backend=args.gaussian_backend,
            )
            histogram_error = max(histogram_error, semantic_maps[8])
        if "rate" not in variants and not args.promoted_only:
            variants["rate"] = metrics(
                semantic_maps[3], masks, metadata["records"], args.workers
            )
        if args.paper_ablation:
            variants["anomalyclip"] = metrics(
                semantic_maps[0], masks, metadata["records"], args.workers
            )
        if not args.final_only:
            variants["targeted_reobservation"] = metrics(
                semantic_maps[4], masks, metadata["records"], args.workers
            )
        if not args.promoted_only and not args.paper_ablation:
            crop_variant = (
                "targeted_reobservation_with_crop_affinity_direct"
                if args.direct_intermediate else
                "targeted_reobservation_with_crop_affinity"
            )
            variants[crop_variant] = metrics(
                semantic_maps[6], masks, metadata["records"], args.workers
            )
        saved_full_closed = (
            Path(args.work_dir) / "semantic_full_closed.npy"
            if args.work_dir else None
        )
        if (args.reuse_full_closed and saved_full_closed is not None
                and saved_full_closed.exists()):
            full_closed = np.load(saved_full_closed, mmap_mode="r")
            variant_name = (
                "remap"
                if args.full_semantic_persistence else
                "targeted_reobservation_with_multiscale_affinity"
            )
            if not args.build_full_only:
                variants[variant_name] = metrics(
                    full_closed, masks, metadata["records"], args.workers
                )
        elif args.identity_features:
            identity_features = np.load(args.identity_features, mmap_mode="r")
            final_maps = select_full_seed_maps(
                semantic_maps, args.full_seed_source
            )
            official_maps = semantic_maps[0]
            if args.work_dir:
                work_dir = Path(args.work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                full_closed = np.lib.format.open_memmap(
                    work_dir / "semantic_full_closed.npy", mode="w+",
                    dtype=np.float32, shape=final_maps.shape,
                )
            else:
                full_closed = np.empty_like(final_maps)
            token_side = int(metadata["tokens"] ** 0.5)
            crop_side = int(fovea_metadata["crop_tokens"] ** 0.5)
            image_size = metadata["image_size"]
            for start in range(0, metadata["count"], args.batch_size):
                stop = min(start + args.batch_size, metadata["count"])
                final_t = torch.from_numpy(np.array(
                    final_maps[start:stop], dtype=np.float32, copy=True
                )).to(device)[:, None]
                token_t = F.interpolate(
                    final_t, (token_side, token_side), mode="bilinear",
                    align_corners=False,
                ).flatten(1)
                feature_t = torch.from_numpy(np.array(
                    identity_features[start:stop], dtype=np.float32, copy=True
                )).to(device)
                supported = positive_feature_support(
                    token_t, feature_t, steps=args.full_feature_steps
                ).reshape(-1, 1, token_side, token_side)
                geometry_t = torch.from_numpy(np.array(
                    geometry[start:stop], dtype=np.float32, copy=True
                )).to(device)
                token_grid, token_valid = inverse_crop_grid(
                    geometry_t, token_side, device
                )
                if args.full_graph_scope == "fovea":
                    # The coarse graph may close a feature-supported gap only
                    # inside the region that triggered the extra observation.
                    # This prevents unrelated, visually similar background
                    # tokens from receiving affinity-propagated re-observation
                    # evidence.
                    supported = token_t.reshape_as(supported) + token_valid * F.relu(
                        supported - token_t.reshape_as(supported)
                    )
                if args.full_semantic_persistence:
                    semantic_crop_t = torch.stack([
                        torch.from_numpy(np.array(
                            semantic[start:stop], dtype=np.float32, copy=True
                        )).to(device).reshape(-1, 1, crop_side, crop_side)
                        for semantic in semantics
                    ]).mean(dim=0)
                    semantic_token = F.grid_sample(
                        semantic_crop_t, token_grid, mode="bilinear",
                        padding_mode="border", align_corners=True,
                    ).flatten(1)
                    base_rank = _percentile_ranks(token_t)
                    closed_rank = _percentile_ranks(supported.flatten(1))
                    semantic_rank = _percentile_ranks(semantic_token)
                    persistent_rank = (
                        base_rank * closed_rank * semantic_rank
                    ).clamp_min(0.0).pow(1.0 / 3.0)
                    bounded_rank = base_rank + token_valid.flatten(1) * F.relu(
                        persistent_rank - base_rank
                    )
                    ordering_token = bounded_rank.reshape_as(supported)
                else:
                    ordering_token = supported
                ordering = F.interpolate(
                    ordering_token, (image_size, image_size), mode="bilinear",
                    align_corners=False,
                )
                anchor_t = torch.from_numpy(np.array(
                    official_maps[start:stop], dtype=np.float32, copy=True
                )).to(device)[:, None]
                full_closed[start:stop] = dense_histogram_transport(
                    anchor_t, ordering
                )[:, 0].cpu().numpy()
            if isinstance(full_closed, np.memmap):
                full_closed.flush()
            if args.full_semantic_persistence:
                variant_name = "remap"
            else:
                variant_name = (
                    "targeted_reobservation_with_multiscale_affinity"
                    if args.full_graph_scope == "fovea" else
                    "targeted_reobservation_with_global_affinity"
                )
            if not args.build_full_only:
                variants[variant_name] = metrics(
                    full_closed, masks, metadata["records"], args.workers
                )
    if args.paper_ablation:
        required = ("anomalyclip", "rate", "remap")
        missing = [name for name in required if name not in variants]
        if missing:
            raise RuntimeError(f"paper ablation is missing variants: {missing}")
        variants = {name: variants[name] for name in required}
    promoted = args.direct_intermediate and args.full_semantic_persistence
    default_method = (
            "remap_component_ablation" if args.paper_ablation else
            ("remap_seed_control" if args.full_seed_source != "reobservation" else
             ("remap" if promoted else "remap_diagnostic"))
        )
    result = {
        "method": args.method or default_method,
        "dataset": metadata["dataset"],
        "geometry_guidance": fovea_metadata["geometry_guidance"],
        "calibration": "none",
        "target_statistics": "none",
        "full_seed_source": args.full_seed_source,
        "intermediate_transport": "none" if args.direct_intermediate else "official_histogram",
        "final_transport": "exact_official_dense_histogram",
        "dense_histogram_max_error": histogram_error,
        "variants": variants,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        name: {key: 100.0 * value for key, value in row["mean"].items()}
        for name, row in variants.items()
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fovea_cache", required=True)
    parser.add_argument(
        "--rate_scores_file",
        help="aligned external RATE scores for efficiency experiments",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", help="explicit result method label")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gaussian_backend", choices=("scipy", "torch"), default="scipy"
    )
    parser.add_argument("--final_only", action="store_true")
    parser.add_argument(
        "--crop_semantic_file", default="crop_semantic_scores.npy"
    )
    parser.add_argument("--crop_semantic_files", nargs="+")
    parser.add_argument(
        "--crop_navigation", choices=("additive", "intersection"),
        default="additive",
    )
    parser.add_argument("--crop_layers", type=int, nargs="+")
    parser.add_argument("--feature_steps", type=int, default=1)
    parser.add_argument("--identity_features")
    parser.add_argument("--full_feature_steps", type=int, default=4)
    parser.add_argument(
        "--full_graph_scope", choices=("fovea", "global"), default="fovea"
    )
    parser.add_argument("--full_semantic_persistence", action="store_true")
    parser.add_argument("--work_dir")
    parser.add_argument("--reuse_work_maps", action="store_true")
    parser.add_argument(
        "--reuse_work_dir",
        help="read reusable semantic maps here while writing new maps to --work_dir",
    )
    parser.add_argument("--promoted_only", action="store_true")
    parser.add_argument("--reuse_full_closed", action="store_true")
    parser.add_argument("--direct_intermediate", action="store_true")
    parser.add_argument("--build_full_only", action="store_true")
    parser.add_argument(
        "--full_seed_source", choices=("reobservation", "rate"),
        default="reobservation",
        help="seed the bounded identity graph from crop-enhanced or pre-crop RATE evidence",
    )
    parser.add_argument(
        "--paper_ablation", action="store_true",
        help=(
            "report only AnomalyCLIP, RATE, and full ReMAP; crop-only "
            "re-observation remains an internal diagnostic"
        ),
    )
    parsed = parser.parse_args()
    if parsed.paper_ablation:
        if parsed.final_only or parsed.promoted_only or parsed.build_full_only:
            parser.error("--paper_ablation cannot be combined with reduced-output modes")
        if parsed.reuse_work_maps or parsed.reuse_full_closed:
            parser.error("--paper_ablation requires a fresh, internally aligned run")
        if not parsed.identity_features:
            parser.error("--paper_ablation requires --identity_features")
        if not parsed.direct_intermediate or not parsed.full_semantic_persistence:
            parser.error(
                "--paper_ablation requires --direct_intermediate and "
                "--full_semantic_persistence"
            )
        if parsed.feature_steps != 4 or parsed.full_feature_steps != 1:
            parser.error(
                "--paper_ablation requires the promoted ReMAP propagation "
                "depths: --feature_steps 4 --full_feature_steps 1"
            )
        if parsed.full_graph_scope != "fovea":
            parser.error(
                "--paper_ablation requires bounded --full_graph_scope fovea"
            )
    main(parsed)
