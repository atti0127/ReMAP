"""Cache RATE and fixed text-bank evidence in one frozen AnomalyCLIP pass."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import AnomalyCLIP_lib
from rate import adapt_rate, align_auxiliary_patch_features, make_auxiliary_view
from dataset import Dataset
from remap.runtime import load_frozen_model
from rate_prompt.cache_text_bank import (
    BANK_NAMES,
    BANKS,
    CLINICAL_STATES,
    _prototype,
    bank_states,
)
from utils import get_transform


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess, target_transform = get_transform(args)
    dataset = Dataset(
        root=args.data_path, transform=preprocess,
        target_transform=target_transform, dataset_name=args.dataset, mode="test",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )
    model, learned_text = load_frozen_model(args, device)
    prototypes = {}
    for class_name in dataset.obj_list:
        readable = class_name.replace("_", " ")
        prototypes[class_name] = {}
        for bank_name in BANK_NAMES:
            states = bank_states(bank_name, class_name)
            normal = _prototype(
                model, (prompt.format(readable) for prompt in states["normal"]), device
            )
            anomaly = _prototype(
                model, (prompt.format(readable) for prompt in states["anomaly"]), device
            )
            prototypes[class_name][bank_name] = torch.stack((normal, anomaly))

    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    tokens = (args.image_size // 14) ** 2
    official_out = np.lib.format.open_memmap(
        output_dir / "official_scores.npy", mode="w+", dtype=np.float32,
        shape=(len(dataset), tokens),
    )
    rate_out = None
    if not args.identity_only:
        rate_out = np.lib.format.open_memmap(
            output_dir / "rate_scores.npy", mode="w+", dtype=np.float32,
            shape=(len(dataset), tokens),
        )
    bank_out = np.lib.format.open_memmap(
        output_dir / "text_bank_scores.npy", mode="w+", dtype=np.float32,
        shape=(len(dataset), len(BANK_NAMES), tokens),
    )
    auxiliary_bank_out = None
    if not args.identity_only:
        auxiliary_bank_out = np.lib.format.open_memmap(
            output_dir / "auxiliary_text_bank_scores.npy", mode="w+",
            dtype=np.float32,
            shape=(len(dataset), len(BANK_NAMES), tokens),
        )
    masks_out = np.lib.format.open_memmap(
        output_dir / "masks.npy", mode="w+", dtype=np.uint8,
        shape=(len(dataset), args.image_size, args.image_size),
    )
    feature_out = np.lib.format.open_memmap(
        output_dir / "identity_features.npy", mode="w+", dtype=np.float16,
        shape=(len(dataset), tokens, 768),
    )
    records, offset = [], 0
    bank_names = list(BANK_NAMES)
    with torch.no_grad():
        for items in tqdm(loader, desc=f"{args.dataset} RATE + text evidence"):
            image = items["img"].to(device, non_blocking=True)
            _, identity_layers = model.encode_image(image, [24], DPAM_layer=20)
            identity = F.normalize(identity_layers[0][:, 1:, :].float(), dim=-1)
            learned_similarity, _ = AnomalyCLIP_lib.compute_similarity(
                identity, learned_text[0]
            )
            official = learned_similarity[..., 1]
            auxiliary_features = None
            aligned_auxiliary = None
            if not args.identity_only:
                auxiliary = make_auxiliary_view(image, args.aux_size)
                _, auxiliary_layers = model.encode_image(
                    auxiliary, [24], DPAM_layer=20
                )
                auxiliary_features = auxiliary_layers[0][:, 1:, :].float()
                aligned_auxiliary = align_auxiliary_patch_features(
                    F.normalize(auxiliary_features, dim=-1), tokens
                )
            transported = []
            text_scores = []
            auxiliary_text_scores = []
            for index, class_name in enumerate(items["cls_name"]):
                if not args.identity_only:
                    result = adapt_rate(
                        learned_text[0], identity[index:index + 1],
                        auxiliary_features[index:index + 1],
                        official[index:index + 1],
                    )
                    transported.append(result.transported_patch_scores[0])
                per_bank = []
                auxiliary_per_bank = []
                for bank_name in bank_names:
                    pair = prototypes[class_name][bank_name]
                    per_bank.append(
                        ((identity[index] @ pair.t()) / 0.07).softmax(dim=-1)[:, 1]
                    )
                    if not args.identity_only:
                        auxiliary_per_bank.append(
                            ((aligned_auxiliary[index] @ pair.t()) / 0.07)
                            .softmax(dim=-1)[:, 1]
                        )
                text_scores.append(torch.stack(per_bank))
                if not args.identity_only:
                    auxiliary_text_scores.append(torch.stack(auxiliary_per_bank))
            count = image.shape[0]
            official_out[offset:offset + count] = official.cpu().numpy()
            if rate_out is not None:
                rate_out[offset:offset + count] = (
                    torch.stack(transported).cpu().numpy()
                )
            bank_out[offset:offset + count] = torch.stack(text_scores).cpu().numpy()
            if auxiliary_bank_out is not None:
                auxiliary_bank_out[offset:offset + count] = (
                    torch.stack(auxiliary_text_scores).cpu().numpy()
                )
            masks_out[offset:offset + count] = (
                items["img_mask"][:, 0].numpy() > 0.5
            ).astype(np.uint8)
            feature_out[offset:offset + count] = (
                identity.cpu().numpy().astype(np.float16)
            )
            for index in range(count):
                records.append({
                    "index": offset + index,
                    "class": items["cls_name"][index],
                    "image_path": items["img_path"][index],
                    "anomaly": int(items["anomaly"][index].item()),
                })
            offset += count
            if offset % 100 < count:
                official_out.flush(); bank_out.flush()
                masks_out.flush(); feature_out.flush()
                if rate_out is not None:
                    rate_out.flush()
                if auxiliary_bank_out is not None:
                    auxiliary_bank_out.flush()
    official_out.flush(); bank_out.flush(); masks_out.flush(); feature_out.flush()
    if rate_out is not None:
        rate_out.flush()
    if auxiliary_bank_out is not None:
        auxiliary_bank_out.flush()
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump({
            "method": (
                "anomalyclip_identity_fixed_text_evidence_cache_v1"
                if args.identity_only else
                "anomalyclip_rate_fixed_text_evidence_cache_v1"
            ),
            "dataset": args.dataset,
            "data_path": str(Path(args.data_path).resolve()),
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "image_size": args.image_size,
            "tokens": tokens,
            "count": len(dataset),
            "bank_names": bank_names,
            "prompt_banks": BANKS,
            "clinical_states": CLINICAL_STATES,
            "target_statistics": "none",
            "encoded_auxiliary": not args.identity_only,
            "records": records,
        }, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--aux_size", type=int, default=224)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--identity_only", action="store_true",
        help=(
            "cache only identity-view evidence needed by ReMAP; "
            "skip the retired encoded auxiliary view"
        ),
    )
    main(parser.parse_args())
