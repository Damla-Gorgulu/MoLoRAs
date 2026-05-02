#!/usr/bin/env python3
"""Build a capped-per-style manifest from MegaStyle-1.4M."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text


def stable_score(*parts: str) -> int:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/scratch/eyavuz21/datasets/MegaStyle-1.4M")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cap", type=int, default=1)
    p.add_argument("--limit_styles", type=int, default=0, help="0 means no limit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_from_disk

    ds_dict = load_from_disk(args.dataset_root)
    ds = ds_dict["train"] if hasattr(ds_dict, "keys") and "train" in ds_dict else ds_dict

    per_style = {}
    counts = {}
    total = 0
    for idx, row in enumerate(ds):
        total += 1
        raw_style = row.get("style", "") or ""
        raw_content = row.get("content", "") or ""
        item_id = row.get("id", "") or str(idx)
        norm_style = normalize_text(raw_style)
        score = stable_score(item_id, raw_style, raw_content)

        chosen = per_style.setdefault(norm_style, [])
        counts[norm_style] = counts.get(norm_style, 0) + 1
        chosen.append(
            {
                "dataset_index": idx,
                "id": item_id,
                "style": raw_style,
                "style_normalized": norm_style,
                "content": raw_content,
                "score": score,
            }
        )
        chosen.sort(key=lambda x: x["score"])
        if len(chosen) > args.cap:
            del chosen[args.cap :]

    styles_sorted = sorted(per_style.keys())
    if args.limit_styles > 0:
        styles_sorted = styles_sorted[: args.limit_styles]

    entries = []
    for norm_style in styles_sorted:
        for item in per_style[norm_style]:
            item = dict(item)
            item["style_count"] = counts[norm_style]
            entries.append(item)

    entries.sort(key=lambda x: (x["style_normalized"], x["score"]))
    summary = {
        "dataset_root": args.dataset_root,
        "total_rows": total,
        "cap": args.cap,
        "limit_styles": args.limit_styles,
        "unique_styles_total": len(per_style),
        "styles_kept": len(styles_sorted),
        "entries_kept": len(entries),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    with (out_dir / "manifest.jsonl").open("w") as f:
        for row in entries:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"[cap-manifest] total_rows={total:,}")
    print(f"[cap-manifest] unique_styles_total={len(per_style):,}")
    print(f"[cap-manifest] styles_kept={len(styles_sorted):,}")
    print(f"[cap-manifest] entries_kept={len(entries):,}")
    print(f"[cap-manifest] output={out_dir}")


if __name__ == "__main__":
    main()
