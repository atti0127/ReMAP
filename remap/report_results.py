"""Render the canonical ReMAP benchmark table from stored evaluation JSON."""

import argparse
import json
from pathlib import Path


DATASETS = (
    ("industrial", "mvtec", "MVTec AD", True),
    ("industrial", "visa", "VisA", True),
    ("industrial", "btad", "BTAD", True),
    ("industrial", "mpdd", "MPDD", True),
    ("industrial", "sdd", "SDD", False),
    ("industrial", "dagm", "DAGM", True),
    ("industrial", "dtd", "DTD-Synthetic", True),
    ("medical", "isic", "ISIC", True),
    ("medical", "colondb", "CVC-ColonDB", True),
    ("medical", "clinicdb", "CVC-ClinicDB", True),
    ("medical", "kvasir", "Kvasir", True),
    ("medical", "endo", "Endo", True),
    ("medical", "tn3k", "TN3K", False),
)

METRICS = ("pixel_auroc", "pixel_aupro")


def _load(path):
    payload = json.loads(path.read_text())
    if payload.get("method") != "remap":
        raise ValueError(f"{path} is not a canonical ReMAP result")
    return payload["variants"]["remap"]


def _pair(result):
    return tuple(100.0 * result["mean"][metric] for metric in METRICS)


def _format(pair):
    return f"{pair[0]:.3f} / {pair[1]:.3f}"


def main(args):
    root = Path(args.results_root)
    rows = []
    missing = []
    for domain, slug, label, primary in DATASETS:
        path = root / f"{slug}.json"
        if not path.exists():
            missing.append(slug)
            continue
        rows.append({
            "domain": domain,
            "slug": slug,
            "dataset": label,
            "primary_mean": primary,
            "metrics": dict(zip(METRICS, _pair(_load(path)))),
        })
    if missing:
        raise ValueError("missing canonical results: " + ", ".join(missing))

    domains = {}
    for domain in ("industrial", "medical"):
        selected = [
            row for row in rows
            if row["domain"] == domain and row["primary_mean"]
        ]
        mean = tuple(
            sum(row["metrics"][metric] for row in selected) / len(selected)
            for metric in METRICS
        )
        domains[domain] = {
            "count": len(selected),
            "mean": dict(zip(METRICS, mean)),
        }

    summary = {
        "method": "ReMAP",
        "backbone": "AnomalyCLIP",
        "protocol": "official AnomalyCLIP category-macro Pixel AUROC/AUPRO",
        "calibration": "none",
        "target_statistics": "none",
        "rows": rows,
        "domains": domains,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# ReMAP Results", "",
        "All values are category-macro Pixel AUROC / AUPRO in percent. "
        "SDD and TN3K are reported as robustness datasets and excluded from "
        "the primary domain means.", "",
        "| Domain | Dataset | ReMAP | Included in mean |",
        "|---|---|---:|:---:|",
    ]
    for row in rows:
        pair = tuple(row["metrics"][metric] for metric in METRICS)
        lines.append(
            f"| {row['domain'].title()} | {row['dataset']} | {_format(pair)} | "
            f"{'yes' if row['primary_mean'] else 'no'} |"
        )
    lines.extend(("", "## Primary domain means", ""))
    for domain, result in domains.items():
        mean = tuple(result["mean"][metric] for metric in METRICS)
        lines.append(f"- {domain.title()}: {_format(mean)}.")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"summary": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/remap/main")
    parser.add_argument("--output", default="results/remap/main/summary.json")
    main(parser.parse_args())
