"""Shared command-line validation for dataset metadata generators."""

import argparse
from pathlib import Path


def parse_dataset_root(dataset_name):
    parser = argparse.ArgumentParser(
        description=f"Generate AnomalyCLIP metadata for {dataset_name}."
    )
    parser.add_argument(
        "--root", required=True, type=Path,
        help="dataset directory in which meta.json will be written",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"dataset directory does not exist: {root}")
    return str(root)

