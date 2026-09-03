"""Run ReMAP on one image and save its anomaly map."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from remap.inference import fixed_prompt_pair, remap_anomaly_map
from remap.runtime import load_frozen_model
from utils import get_transform


def _normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    minimum = float(scores.min())
    span = float(scores.max()) - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros_like(scores)
    return (scores - minimum) / span


def _save_visualizations(image_path, scores, output_dir, alpha):
    normalized = _normalize(scores)
    four = 4.0 * normalized
    heatmap_rgb = np.stack((
        np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0),
        np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0),
        np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0),
    ), axis=-1)
    heatmap = Image.fromarray(
        np.rint(255.0 * heatmap_rgb).astype(np.uint8), mode="RGB"
    )
    image = Image.open(image_path).convert("RGB").resize(
        (scores.shape[1], scores.shape[0]), Image.Resampling.BICUBIC
    )
    overlay = Image.blend(image, heatmap, 1.0 - alpha)
    heatmap_path = output_dir / "remap_heatmap.png"
    overlay_path = output_dir / "remap_overlay.png"
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)
    return heatmap_path, overlay_path


def main(args):
    image_path = Path(args.image_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    preprocess, _ = get_transform(args)
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    model, learned_text = load_frozen_model(args, device)
    route_pair = fixed_prompt_pair(model, args.class_name, "route_state", device)
    crop_bank = "clinical" if args.domain == "medical" else "structural"
    crop_pair = fixed_prompt_pair(model, args.class_name, crop_bank, device)
    scores = remap_anomaly_map(
        model,
        learned_text,
        route_pair,
        crop_pair,
        image,
        image_size=args.image_size,
        auxiliary_size=args.aux_size,
        crop_size=args.crop_size,
    )[0]

    score_path = output_dir / "remap_score.npy"
    np.save(score_path, scores.astype(np.float32))
    heatmap_path, overlay_path = _save_visualizations(
        image_path, scores, output_dir, args.overlay_alpha
    )
    print(json.dumps({
        "device": str(device),
        "score_map": str(score_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run training-free ReMAP localization on one image."
    )
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", default="outputs/remap")
    parser.add_argument(
        "--domain", choices=("industrial", "medical"), default="industrial"
    )
    parser.add_argument(
        "--class_name", default="object",
        help="object or organ name used by the fixed prompts",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--aux_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
