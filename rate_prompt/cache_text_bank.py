"""Cache fixed class-state text-bank evidence over AnomalyCLIP tokens."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from remap.runtime import load_frozen_model
from prompt_ensemble import tokenize


BANKS = {
    "structural": {
        "normal": (
            "a photo of a normal {}.",
            "a photo of an intact {}.",
            "a photo of an undamaged {}.",
            "a {} without defects.",
        ),
        "anomaly": (
            "a photo of a damaged {}.",
            "a photo of a defective {}.",
            "a photo of a broken {}.",
            "a {} with a defect.",
        ),
    },
    "pathological": {
        "normal": (
            "a photo of healthy {}.",
            "a photo of normal {} tissue.",
            "a {} without a lesion.",
            "a healthy medical image of {}.",
        ),
        "anomaly": (
            "a photo of diseased {}.",
            "a photo of abnormal {} tissue.",
            "a {} with a lesion.",
            "a medical image of a {} tumor.",
        ),
    },
    "generic": {
        "normal": (
            "a normal image of {}.",
            "a flawless {}.",
            "an unblemished {}.",
            "a {} without anomaly.",
        ),
        "anomaly": (
            "an abnormal image of {}.",
            "an anomalous {}.",
            "an irregular {}.",
            "a {} with anomaly.",
        ),
    },
}

CLINICAL_STATES = {
    "skin": {
        "normal": (
            "a photo of healthy skin.", "a normal skin surface.",
            "skin without a lesion.", "a benign image of skin.",
        ),
        "anomaly": (
            "a photo of a skin lesion.", "a melanoma on skin.",
            "an abnormal growth on skin.", "diseased skin tissue.",
        ),
    },
    "colon": {
        "normal": (
            "a normal colonoscopy image.", "healthy colon mucosa.",
            "an endoscopy image without a polyp.", "a colon without polyps.",
        ),
        "anomaly": (
            "a colon polyp.", "an endoscopy image with a polyp.",
            "a polyp in the colon.", "abnormal polypoid colon mucosa.",
        ),
    },
    "thyroid": {
        "normal": (
            "a normal thyroid ultrasound.", "healthy thyroid tissue.",
            "a thyroid without nodules.", "a normal thyroid gland.",
        ),
        "anomaly": (
            "a thyroid nodule on ultrasound.", "an abnormal thyroid nodule.",
            "a lesion in the thyroid.", "a thyroid tumor.",
        ),
    },
    "brain": {
        "normal": (
            "a normal brain scan.", "healthy brain tissue.",
            "a brain without a tumor.", "a normal medical image of the brain.",
        ),
        "anomaly": (
            "a brain tumor.", "a lesion in the brain.",
            "an abnormal brain scan.", "a mass in brain tissue.",
        ),
    },
    "chest": {
        "normal": (
            "a normal chest x-ray.", "healthy clear lungs.",
            "a chest radiograph without opacity.", "a normal lung scan.",
        ),
        "anomaly": (
            "an abnormal chest x-ray with opacity.", "pneumonia in the lungs.",
            "a COVID lung radiograph.", "a lesion in a chest radiograph.",
        ),
    },
}

BANK_NAMES = tuple(BANKS) + ("clinical",)


def bank_states(bank_name, class_name):
    if bank_name == "route_state":
        return (
            BANKS["pathological"]
            if class_name in CLINICAL_STATES else BANKS["structural"]
        )
    if bank_name == "class_state":
        return (
            CLINICAL_STATES[class_name]
            if class_name in CLINICAL_STATES else BANKS["structural"]
        )
    if bank_name == "clinical":
        return CLINICAL_STATES.get(class_name, BANKS["pathological"])
    return BANKS[bank_name]


@torch.no_grad()
def _prototype(model, prompts, device):
    tokens = tokenize(list(prompts)).to(device)
    # AnomalyCLIP's text transformer expects the compound-prompt carrier even
    # when no learned tokens are inserted.  Its inherited ``encode_text``
    # entry point omits that carrier, so encode the fixed phrases explicitly.
    x = model.token_embedding(tokens).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer([x, [], 0])
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    feature = (
        x[torch.arange(x.shape[0], device=device), tokens.argmax(dim=-1)]
        @ model.text_projection
    ).float()
    feature = F.normalize(feature, dim=-1).mean(dim=0)
    return F.normalize(feature, dim=0)


def main(args):
    feature_path = Path(args.identity_features)
    branch_dir = Path(args.branch_cache)
    with (branch_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[:2] != (metadata["count"], metadata["tokens"]):
        raise ValueError(f"feature/cache mismatch: {features.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_frozen_model(args, device)
    classes = sorted({record["class"] for record in metadata["records"]})
    prototypes = {}
    for class_name in classes:
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_dir / "text_bank_scores.npy", mode="w+", dtype=np.float32,
        shape=(metadata["count"], len(BANK_NAMES), metadata["tokens"]),
    )
    records = metadata["records"]
    bank_names = list(BANK_NAMES)
    with torch.no_grad():
        for start in range(0, metadata["count"], args.batch_size):
            stop = min(start + args.batch_size, metadata["count"])
            token = F.normalize(torch.from_numpy(
                np.array(features[start:stop], dtype=np.float32, copy=True)
            ).to(device), dim=-1)
            batch_scores = []
            for offset, record in enumerate(records[start:stop]):
                per_bank = []
                for bank_name in bank_names:
                    pair = prototypes[record["class"]][bank_name]
                    per_bank.append(
                        ((token[offset] @ pair.t()) / 0.07).softmax(dim=-1)[:, 1]
                    )
                batch_scores.append(torch.stack(per_bank))
            output[start:stop] = torch.stack(batch_scores).cpu().numpy()
    output.flush()
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump({
            "method": "fixed_class_state_text_banks_v1",
            "dataset": metadata["dataset"],
            "count": metadata["count"],
            "tokens": metadata["tokens"],
            "bank_names": bank_names,
            "prompt_banks": BANKS,
            "clinical_states": CLINICAL_STATES,
            "target_statistics": "none",
            "records": records,
        }, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch_cache", required=True)
    parser.add_argument("--identity_features", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
