#!/usr/bin/env python3
"""
Backfill reference images into existing experiment output folders.

For every *_attention.pt file found under the given root(s), this script:
  1. Copies the query / input style image  → __query.<ext>
  2. Copies the top-5 expert style thumbnails → __top1_<name>.jpg … __top5_<name>.jpg

Run on the login node — pure file I/O, no GPU needed.

Usage:
    python lora_attention/backfill_reference_images.py

Alternatively pass explicit root dirs as positional arguments:
    python lora_attention/backfill_reference_images.py /scratch/eyavuz21/lora_attention/generalization_v2
"""

import sys
import shutil
from pathlib import Path

import torch


# ── Config ────────────────────────────────────────────────────────────────────
STYLE_IMAGES_ROOT = Path(
    "/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
)

DEFAULT_ROOTS = [
    "/scratch/eyavuz21/lora_attention/generalization",
    "/scratch/eyavuz21/lora_attention/generalization_v2",
    "/scratch/eyavuz21/lora_attention/inference_s1",
    "/scratch/eyavuz21/lora_attention/inference_s2",
    "/scratch/eyavuz21/lora_attention/inference_sweep",
]

TOP_K = 5
# ─────────────────────────────────────────────────────────────────────────────


def avg_attention_per_expert(A: torch.Tensor) -> torch.Tensor:
    """Return (N,) avg attention regardless of v1 (N, r) or v2 (N, T, r) shape."""
    if A.dim() == 2:
        return A.mean(dim=1)      # v1.0: (N, rank) → (N,)
    elif A.dim() == 3:
        return A.mean(dim=(1, 2)) # v2.0: (N, T, r) → (N,)
    else:
        raise ValueError(f"Unexpected attention shape: {A.shape}")


def process_pt(pt_path: Path, dry_run: bool = False) -> dict:
    out_dir = pt_path.parent
    result = {"pt": str(pt_path), "copied": [], "skipped": [], "warnings": []}

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        result["warnings"].append(f"load error: {e}")
        return result

    # ── Query image ───────────────────────────────────────────
    query_src_str = data.get("style_image", "")
    if query_src_str:
        query_src = Path(query_src_str)
        suffix = query_src.suffix or ".jpg"
        # Use the attention file's own prefix as label so names stay consistent
        label_prefix = pt_path.stem.replace("_attention", "")
        query_dst = out_dir / f"{label_prefix}__query{suffix}"

        if query_dst.exists():
            result["skipped"].append(query_dst.name)
        elif query_src.exists():
            if not dry_run:
                shutil.copy2(query_src, query_dst)
            result["copied"].append(query_dst.name)
        else:
            result["warnings"].append(f"query src missing: {query_src}")
    else:
        label_prefix = pt_path.stem.replace("_attention", "")
        result["warnings"].append("no style_image field in .pt")

    # ── Top-K expert style images ─────────────────────────────
    A = data.get("attention")
    pool_names = data.get("pool_names", [])
    pool_indices = data.get("pool_indices", list(range(len(pool_names))))

    if A is None or len(pool_names) == 0:
        result["warnings"].append("missing attention or pool_names")
        return result

    try:
        avg_attn = avg_attention_per_expert(A)          # (N,)
        top_k_actual = min(TOP_K, len(pool_names))
        top_vals, top_idxs = avg_attn.topk(top_k_actual)
    except Exception as e:
        result["warnings"].append(f"attention processing error: {e}")
        return result

    for rank_i, expert_i in enumerate(top_idxs.tolist()):
        style_name = pool_names[expert_i]               # e.g. "style_0000_Baroque"
        src = STYLE_IMAGES_ROOT / style_name / f"{style_name}.jpg"
        attn_val = top_vals[rank_i].item()
        dst = out_dir / f"{label_prefix}__top{rank_i+1}_{style_name}.jpg"

        if dst.exists():
            result["skipped"].append(dst.name)
        elif src.exists():
            if not dry_run:
                shutil.copy2(src, dst)
            result["copied"].append(f"{dst.name} (avg_A={attn_val:.4f})")
        else:
            result["warnings"].append(f"style src missing: {src}")

    return result


def main():
    roots = [Path(r) for r in (sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ROOTS)]

    pt_files = []
    for root in roots:
        if root.exists():
            pt_files.extend(sorted(root.rglob("*_attention.pt")))
        else:
            print(f"[skip] root not found: {root}")

    print(f"Found {len(pt_files)} attention .pt files\n")

    total_copied = 0
    total_skipped = 0
    total_warnings = 0

    for pt_path in pt_files:
        rel = pt_path.relative_to(pt_path.parents[4] if len(pt_path.parts) > 4 else pt_path.parent)
        res = process_pt(pt_path)

        n_copied = len(res["copied"])
        n_skipped = len(res["skipped"])
        n_warn = len(res["warnings"])
        total_copied += n_copied
        total_skipped += n_skipped
        total_warnings += n_warn

        # Print only folders with at least one action
        if n_copied or n_warn:
            print(f"  {pt_path.parent.relative_to(pt_path.parents[len(pt_path.parts) - len(pt_path.parts)])}")
            print(f"  [{pt_path.name}]")
            for f in res["copied"]:
                print(f"    + {f}")
            for w in res["warnings"]:
                print(f"    ! {w}")
        else:
            print(f"  (already done) {pt_path.name}")

    print(f"\n{'='*60}")
    print(f"  Copied  : {total_copied} files")
    print(f"  Skipped : {total_skipped} (already present)")
    print(f"  Warnings: {total_warnings}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
