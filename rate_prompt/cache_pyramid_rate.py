"""Cache RATE scores using a zero-forward feature-pyramid auxiliary view."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rate import adapt_rate, make_pyramid_auxiliary_features
from remap.runtime import load_frozen_model


def main(args):
    cache = Path(args.cache)
    metadata = json.loads((cache / "metadata.json").read_text())
    identity = np.load(cache / "identity_features.npy", mmap_mode="r")
    official = np.load(cache / "official_scores.npy", mmap_mode="r")
    if identity.shape[:2] != official.shape:
        raise ValueError("identity feature and official-score caches are misaligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, learned_text = load_frozen_model(args, device)
    del model

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=official.shape
    )
    diagnostics = {
        "accepted": 0.0, "rank_shift": 0.0, "histogram_max_error": 0.0
    }
    for index in tqdm(range(metadata["count"]), desc="pyramid RATE"):
        identity_t = torch.from_numpy(np.array(
            identity[index:index + 1], dtype=np.float32, copy=True
        )).to(device)
        official_t = torch.from_numpy(np.array(
            official[index:index + 1], dtype=np.float32, copy=True
        )).to(device)
        auxiliary_t = make_pyramid_auxiliary_features(
            identity_t, auxiliary_side=args.auxiliary_side,
            interpolation=args.interpolation,
        )
        result = adapt_rate(
            learned_text[0], identity_t, auxiliary_t, official_t,
            voters=tuple(args.voters),
        )
        scores[index] = result.transported_patch_scores[0].cpu().numpy()
        for key in diagnostics:
            diagnostics[key] += result.diagnostics[key]
        if index % 100 == 0:
            scores.flush()
    scores.flush()
    diagnostics = {
        key: value / metadata["count"] for key, value in diagnostics.items()
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "method": "rate_feature_pyramid_auxiliary_v1",
        "dataset": metadata["dataset"],
        "count": metadata["count"],
        "tokens": metadata["tokens"],
        "auxiliary_side": args.auxiliary_side,
        "interpolation": args.interpolation,
        "auxiliary_encoder_forwards": 0,
        "voters": list(args.voters),
        "calibration": "none",
        "target_statistics": "none",
        "diagnostics": diagnostics,
    }, indent=2) + "\n")
    print(json.dumps({"scores": str(output), "metadata": str(sidecar)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auxiliary_side", type=int, default=16)
    parser.add_argument(
        "--interpolation",
        choices=("area", "bilinear", "bicubic"),
        default="area",
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument(
        "--voters", nargs="+",
        choices=(
            "cross_scale_affinity_rank",
            "rarity_corrected_rank",
            "extrapolated_fusion_rank",
        ),
        default=(
            "cross_scale_affinity_rank",
            "rarity_corrected_rank",
            "extrapolated_fusion_rank",
        ),
    )
    main(parser.parse_args())
