#!/usr/bin/env python3
"""Analyze MegaStyle metadata to estimate pre-CLIP search-base shrinkage."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_root",
        default="/scratch/eyavuz21/datasets/MegaStyle-1.4M",
        help="Path passed to datasets.load_from_disk().",
    )
    p.add_argument(
        "--output_dir",
        default="/scratch/eyavuz21/datasets/MegaStyle-1.4M_analysis",
        help="Directory for JSON/CSV outputs.",
    )
    p.add_argument(
        "--sample_top_k",
        type=int,
        default=200,
        help="How many top-style rows to save in the summary outputs.",
    )
    p.add_argument(
        "--style_caps",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500],
        help="Per-normalized-style candidate caps to simulate.",
    )
    return p.parse_args()


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text


def iter_rows(ds) -> Iterable[Tuple[str, str, str]]:
    for row in ds:
        yield row.get("id", ""), row.get("style", ""), row.get("content", "")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_from_disk

    print(f"[megastyle-meta] loading: {args.dataset_root}")
    ds_dict = load_from_disk(args.dataset_root)
    ds = ds_dict["train"] if hasattr(ds_dict, "keys") and "train" in ds_dict else ds_dict

    raw_style_counts: Counter[str] = Counter()
    norm_style_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    style_variants: Dict[str, Counter[str]] = defaultdict(Counter)

    total = 0
    for _id, raw_style, content in iter_rows(ds):
        total += 1
        raw_style = raw_style or ""
        raw_content = content or ""
        norm_style = normalize_text(raw_style)
        norm_content = normalize_text(raw_content)

        raw_style_counts[raw_style] += 1
        norm_style_counts[norm_style] += 1
        pair_counts[(norm_style, norm_content)] += 1
        style_variants[norm_style][raw_style] += 1

        if total % 100000 == 0:
            print(f"[megastyle-meta] processed {total:,} rows")

    unique_raw_styles = len(raw_style_counts)
    unique_norm_styles = len(norm_style_counts)
    unique_pairs = len(pair_counts)

    exact_pair_removed = total - unique_pairs
    exact_pair_keep = unique_pairs

    cap_results = []
    for cap in sorted(set(args.style_caps)):
        kept = sum(min(count, cap) for count in norm_style_counts.values())
        removed = total - kept
        cap_results.append(
            {
                "cap": cap,
                "kept": kept,
                "removed": removed,
                "kept_fraction": kept / total if total else 0.0,
                "removed_fraction": removed / total if total else 0.0,
            }
        )

    top_norm_styles = []
    for norm_style, count in norm_style_counts.most_common(args.sample_top_k):
        variants = style_variants[norm_style].most_common(10)
        top_norm_styles.append(
            {
                "normalized_style": norm_style,
                "count": count,
                "raw_variants": [{"style": s, "count": c} for s, c in variants],
            }
        )

    top_pairs = []
    for (norm_style, norm_content), count in pair_counts.most_common(args.sample_top_k):
        top_pairs.append(
            {
                "normalized_style": norm_style,
                "normalized_content": norm_content,
                "count": count,
            }
        )

    summary = {
        "dataset_root": args.dataset_root,
        "total_rows": total,
        "unique_raw_styles": unique_raw_styles,
        "unique_normalized_styles": unique_norm_styles,
        "unique_normalized_style_content_pairs": unique_pairs,
        "exact_pair_dedup": {
            "kept": exact_pair_keep,
            "removed": exact_pair_removed,
            "kept_fraction": exact_pair_keep / total if total else 0.0,
            "removed_fraction": exact_pair_removed / total if total else 0.0,
        },
        "cap_results": cap_results,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "top_normalized_styles.json").write_text(json.dumps(top_norm_styles, indent=2, sort_keys=True) + "\n")
    (out_dir / "top_normalized_pairs.json").write_text(json.dumps(top_pairs, indent=2, sort_keys=True) + "\n")

    with (out_dir / "cap_results.csv").open("w") as f:
        f.write("cap,kept,removed,kept_fraction,removed_fraction\n")
        for row in cap_results:
            f.write(
                f"{row['cap']},{row['kept']},{row['removed']},"
                f"{row['kept_fraction']:.8f},{row['removed_fraction']:.8f}\n"
            )

    print("[megastyle-meta] done")
    print(f"[megastyle-meta] rows={total:,}")
    print(f"[megastyle-meta] unique_raw_styles={unique_raw_styles:,}")
    print(f"[megastyle-meta] unique_normalized_styles={unique_norm_styles:,}")
    print(f"[megastyle-meta] unique_pairs={unique_pairs:,}")
    print(f"[megastyle-meta] output={out_dir}")


if __name__ == "__main__":
    main()
