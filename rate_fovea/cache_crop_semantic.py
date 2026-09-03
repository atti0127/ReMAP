"""Score cached foveal features with another fixed class-state prompt bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from remap.runtime import load_frozen_model
from rate_prompt.cache_text_bank import _prototype, bank_states


def main(args):
    cache = Path(args.fovea_cache)
    with (cache / "metadata.json").open() as handle:
        metadata = json.load(handle)
    features = np.load(cache / "crop_features.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_frozen_model(args, device)
    prototypes = {}
    for class_name in sorted({row["class"] for row in metadata["records"]}):
        readable = class_name.replace("_", " ")
        states = bank_states(args.bank, class_name)
        normal = _prototype(
            model, (prompt.format(readable) for prompt in states["normal"]), device
        )
        anomaly = _prototype(
            model, (prompt.format(readable) for prompt in states["anomaly"]), device
        )
        prototypes[class_name] = torch.stack((normal, anomaly))
    output = np.lib.format.open_memmap(
        cache / f"crop_semantic_scores_{args.bank}.npy", mode="w+",
        dtype=np.float32, shape=features.shape[:2],
    )
    records = metadata["records"]
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            stop = min(start + args.batch_size, len(features))
            token = F.normalize(torch.from_numpy(np.array(
                features[start:stop], dtype=np.float32, copy=True
            )).to(device), dim=-1)
            rows = []
            for local, record in enumerate(records[start:stop]):
                pair = prototypes[record["class"]]
                rows.append(
                    ((token[local] @ pair.t()) / 0.07).softmax(dim=-1)[:, 1]
                )
            output[start:stop] = torch.stack(rows).cpu().numpy()
    output.flush()
    print(json.dumps({"saved": str(cache / f"crop_semantic_scores_{args.bank}.npy")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fovea_cache", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--bank", required=True,
        choices=("structural", "pathological", "generic", "clinical"),
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    main(parser.parse_args())
