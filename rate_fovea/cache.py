"""Cache the crop observation used by Affinity-Propagated Re-observation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import AnomalyCLIP_lib
from rate.core import _percentile_ranks
from dataset import Dataset
from remap.runtime import canonical_path, load_frozen_model
from rate_fovea import extract_center_fovea, extract_moment_fovea
from rate_prompt.cache_text_bank import CLINICAL_STATES, _prototype, bank_states
from utils import get_transform


def _same_record_path(observed, expected, class_name):
    """Validate cache order for both class-folder and flat medical layouts."""

    try:
        return canonical_path(observed, class_name) == canonical_path(
            expected, class_name
        )
    except ValueError:
        return Path(observed).resolve() == Path(expected).resolve()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch_cache = Path(args.branch_cache)
    with (branch_cache / "metadata.json").open() as handle:
        branch_metadata = json.load(handle)
    branch_path = branch_cache / "branch_scores.npy"
    if branch_path.exists():
        branch_scores = np.load(branch_path, mmap_mode="r")[:, -1]
    else:
        # Self-contained medical prompt caches store the exact same official
        # layer-24 evidence without the otherwise-unused branch dimension.
        branch_scores = np.load(
            branch_cache / "official_scores.npy", mmap_mode="r"
        )
    guidance_scores = None
    if args.guidance_scores:
        guidance_scores = np.load(args.guidance_scores, mmap_mode="r")
        if guidance_scores.shape != (
            branch_metadata["count"], branch_metadata["tokens"]
        ):
            raise ValueError("external guidance-score shape mismatch")
    elif args.guidance_cache:
        guidance_dir = Path(args.guidance_cache)
        with (guidance_dir / "metadata.json").open() as handle:
            guidance_metadata = json.load(handle)
        if guidance_metadata["count"] != branch_metadata["count"]:
            raise ValueError("branch/guidance count mismatch")
        guidance_scores = np.load(
            guidance_dir / "rate_scores.npy", mmap_mode="r"
        )
    semantic_scores = None
    if args.semantic_cache:
        semantic_dir = Path(args.semantic_cache)
        with (semantic_dir / "metadata.json").open() as handle:
            semantic_metadata = json.load(handle)
        if semantic_metadata["count"] != branch_metadata["count"]:
            raise ValueError("branch/semantic count mismatch")
        all_semantic_scores = np.load(
            semantic_dir / "text_bank_scores.npy", mmap_mode="r"
        )
        if args.semantic_bank in ("class_state", "route_state"):
            structural_index = semantic_metadata["bank_names"].index("structural")
            medical_bank = (
                "clinical" if args.semantic_bank == "class_state"
                else "pathological"
            )
            medical_index = semantic_metadata["bank_names"].index(medical_bank)
            semantic_scores = np.empty(
                (branch_metadata["count"], branch_metadata["tokens"]),
                dtype=np.float32,
            )
            for index, record in enumerate(branch_metadata["records"]):
                bank_index = (
                    medical_index
                    if record["class"] in CLINICAL_STATES else structural_index
                )
                semantic_scores[index] = all_semantic_scores[index, bank_index]
        else:
            bank_index = semantic_metadata["bank_names"].index(args.semantic_bank)
            semantic_scores = all_semantic_scores[:, bank_index]
    preprocess, target_transform = get_transform(args)
    dataset = Dataset(
        root=args.data_path, transform=preprocess, target_transform=target_transform,
        dataset_name=args.dataset, mode="test",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )
    if len(dataset) != branch_metadata["count"]:
        raise ValueError("dataset and branch cache counts differ")
    model, text = load_frozen_model(args, device)
    semantic_prototypes = None
    if semantic_scores is not None:
        semantic_prototypes = {}
        for class_name in sorted({row["class"] for row in branch_metadata["records"]}):
            readable = class_name.replace("_", " ")
            states = bank_states(args.semantic_bank, class_name)
            normal = _prototype(
                model,
                (prompt.format(readable) for prompt in states["normal"]),
                device,
            )
            anomaly = _prototype(
                model,
                (prompt.format(readable) for prompt in states["anomaly"]),
                device,
            )
            semantic_prototypes[class_name] = torch.stack((normal, anomaly))
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    token_side = args.crop_size // 14
    crop_scores = np.lib.format.open_memmap(
        output_dir / "crop_scores.npy", mode="w+", dtype=np.float32,
        shape=(len(dataset), token_side * token_side),
    )
    output_layers = [24] if args.final_layer_only else [6, 12, 18, 24]
    crop_branch_scores = np.lib.format.open_memmap(
        output_dir / "crop_branch_scores.npy", mode="w+", dtype=np.float32,
        shape=(len(dataset), len(output_layers), token_side * token_side),
    )
    crop_features = np.lib.format.open_memmap(
        output_dir / "crop_features.npy", mode="w+", dtype=np.float16,
        shape=(len(dataset), token_side * token_side, 768),
    )
    crop_semantic_scores = (
        np.lib.format.open_memmap(
            output_dir / "crop_semantic_scores.npy", mode="w+", dtype=np.float32,
            shape=(len(dataset), token_side * token_side),
        )
        if semantic_prototypes is not None else None
    )
    geometry_out = np.lib.format.open_memmap(
        output_dir / "geometry.npy", mode="w+", dtype=np.float32,
        shape=(len(dataset), 4 if args.anisotropic else 3),
    )
    offset = 0
    with torch.no_grad():
        for items in tqdm(loader, desc=f"{args.dataset} foveal evidence"):
            image = items["img"].to(device, non_blocking=True)
            count = image.shape[0]
            expected = branch_metadata["records"][offset:offset + count]
            for observed_path, record in zip(items["img_path"], expected):
                if not _same_record_path(
                    observed_path, record["image_path"], record["class"]
                ):
                    raise RuntimeError("dataset order differs from branch cache")
            guidance = (
                guidance_scores[offset:offset + count]
                if guidance_scores is not None
                else branch_scores[offset:offset + count]
            )
            native = torch.from_numpy(
                np.array(guidance,
                         dtype=np.float32, copy=True)
            ).to(device)
            if semantic_scores is not None:
                semantic = torch.from_numpy(np.array(
                    semantic_scores[offset:offset + count],
                    dtype=np.float32, copy=True,
                )).to(device)
                # Rank-product is a coefficient-free intersection: a site
                # attracts resolution only when both RATE and the independent
                # class-state prompt support it.
                native = torch.sqrt(
                    _percentile_ranks(native) * _percentile_ranks(semantic)
                )
            if args.geometry_mode == "center":
                crop, geometry = extract_center_fovea(image, args.crop_size)
            else:
                crop, geometry = extract_moment_fovea(
                    image, native, args.crop_size, sigma_span=args.sigma_span,
                    anisotropic=args.anisotropic,
                )
            _, layers = model.encode_image(
                crop, output_layers, DPAM_layer=20
            )
            normalized_layers = [
                F.normalize(layer[:, 1:, :].float(), dim=-1) for layer in layers
            ]
            branch_values = []
            for layer_features in normalized_layers:
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    layer_features, text[0]
                )
                branch_values.append(similarity[..., 1])
            stacked_scores = torch.stack(branch_values, dim=1)
            features = normalized_layers[-1]
            crop_scores[offset:offset + count] = stacked_scores[:, -1].cpu().numpy()
            crop_branch_scores[offset:offset + count] = stacked_scores.cpu().numpy()
            crop_features[offset:offset + count] = features.cpu().numpy().astype(np.float16)
            if crop_semantic_scores is not None:
                semantic_batch = []
                for local_index, record in enumerate(expected):
                    prototype = semantic_prototypes[record["class"]]
                    semantic_batch.append(
                        ((features[local_index] @ prototype.t()) / 0.07)
                        .softmax(dim=-1)[:, 1]
                    )
                crop_semantic_scores[offset:offset + count] = (
                    torch.stack(semantic_batch).cpu().numpy()
                )
            geometry_out[offset:offset + count] = geometry.cpu().numpy()
            offset += count
            if offset % 50 < count:
                crop_scores.flush(); crop_branch_scores.flush()
                crop_features.flush(); geometry_out.flush()
                if crop_semantic_scores is not None:
                    crop_semantic_scores.flush()
    crop_scores.flush(); crop_branch_scores.flush()
    crop_features.flush(); geometry_out.flush()
    if crop_semantic_scores is not None:
        crop_semantic_scores.flush()
    metadata = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "branch_cache": str(branch_cache.resolve()),
        "count": len(dataset),
        "crop_size": args.crop_size,
        "crop_tokens": token_side * token_side,
        "crop_features": "L2-normalized layer-24 float16",
        "crop_semantic_scores": (
            f"fixed {args.semantic_bank} class-state prompt"
            if crop_semantic_scores is not None else None
        ),
        "crop_branch_scores": [f"layer_{layer}" for layer in output_layers],
        "final_layer_only": args.final_layer_only,
        "geometry_mode": args.geometry_mode,
        "geometry": (
            f"fixed image center and exact {args.crop_size}x{args.crop_size} source footprint"
            if args.geometry_mode == "center" else
            (f"softmax-z spatial center and independent {args.sigma_span:g}-sigma x/y extents"
             if args.anisotropic else
             f"softmax-z spatial center and {args.sigma_span:g}-sigma extent")
        ),
        "geometry_guidance": (
            "fixed image center (compute-matched control)"
            if args.geometry_mode == "center" else
            (f"rank-product of RATE and {args.semantic_bank} prompt"
            if semantic_scores is not None else
            ("RATE transported scores" if guidance_scores is not None
             else "official layer-24 scores"))
        ),
        "target_statistics": "none",
        "records": branch_metadata["records"],
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--branch_cache", required=True)
    parser.add_argument("--guidance_cache")
    parser.add_argument(
        "--guidance_scores",
        help="aligned external RATE scores for efficiency experiments",
    )
    parser.add_argument("--semantic_cache")
    parser.add_argument(
        "--semantic_bank", default="pathological",
        choices=(
            "structural", "pathological", "generic", "clinical",
            "class_state", "route_state",
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop_size", type=int, default=336)
    parser.add_argument("--sigma_span", type=float, default=2.0)
    parser.add_argument("--anisotropic", action="store_true")
    parser.add_argument(
        "--geometry_mode", choices=("targeted", "center"), default="targeted"
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--final_layer_only", action="store_true",
        help="materialize only the layer-24 crop output used by final ReMAP",
    )
    parsed = parser.parse_args()
    if parsed.geometry_mode == "center" and parsed.anisotropic:
        parser.error("--geometry_mode center cannot be combined with --anisotropic")
    main(parsed)
