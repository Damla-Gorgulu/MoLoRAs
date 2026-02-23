#!/usr/bin/env python3
"""
Generalization Analysis: Summarise attention routing across held-out experiments.

Reads all *_attention.pt files in a results directory, prints a ranked table
of routing outcomes, and computes generalisation metrics.

Usage:
    python lora_attention/analyse_generalization.py --results_dir /scratch/.../generalization
    python lora_attention/analyse_generalization.py --results_dir ... --csv report.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch


def load_attention_files(results_dir: Path):
    """Recursively find and load all *_attention.pt files."""
    records = []
    for pt_file in sorted(results_dir.rglob("*_attention.pt")):
        try:
            d = torch.load(pt_file, map_location="cpu", weights_only=False)
            d["_file"] = pt_file
            records.append(d)
        except Exception as e:
            print(f"[warn] Could not load {pt_file}: {e}")
    return records


def analyse(records, top_k_report: int = 5, csv_path: Optional[Path] = None):
    """Print per-run routing table and aggregate metrics."""

    rows = []

    print(f"\n{'='*100}")
    print(f"{'Run':40s}  {'GT Expert':40s}  {'GT in pool':10s}  {'GT rank':8s}  "
          f"{'top-1 Expert':35s}  {'top-1 A':7s}  {'entropy':7s}")
    print(f"{'='*100}")

    gt_in_pool_count = gt_top1_count = gt_top3_count = total_with_gt = 0

    for rec in records:
        A = rec["attention"]          # (N, rank)
        pool_names = rec["pool_names"]
        gt_expert = rec.get("gt_expert") or ""
        gt_in_pool = rec.get("gt_in_pool")
        query_label = rec.get("query_label") or Path(rec["_file"]).parent.name
        exclude_experts = rec.get("exclude_experts") or []

        avg_attn = A.mean(dim=1).tolist()  # (N,)
        ranked = sorted(range(len(pool_names)), key=lambda i: avg_attn[i], reverse=True)

        top1_name = pool_names[ranked[0]] if ranked else "—"
        top1_attn = avg_attn[ranked[0]] if ranked else 0.0
        top_names = [pool_names[ranked[i]] for i in range(min(top_k_report, len(ranked)))]

        # Entropy
        import math
        N = len(pool_names)
        entropy = -sum(a * math.log(a + 1e-10) for a in avg_attn) / N
        max_entropy = math.log(N) if N > 1 else 1.0
        norm_entropy = entropy / max_entropy

        # GT rank
        gt_rank = None
        gt_attn_val = None
        if gt_expert and gt_in_pool:
            for local_i, name in enumerate(pool_names):
                if gt_expert in name:
                    gt_attn_val = avg_attn[local_i]
                    gt_rank = sum(1 for a in avg_attn if a > gt_attn_val) + 1
                    break

        # Truncate run label
        label_short = str(rec["_file"].parent)[-50:]

        print(f"{label_short:50s}  {(gt_expert or '—')[:38]:38s}  "
              f"{'yes' if gt_in_pool else 'NO':10s}  "
              f"{str(gt_rank) if gt_rank else '—':8s}  "
              f"{top1_name[:33]:33s}  "
              f"{top1_attn:.4f}  "
              f"{norm_entropy:.3f}")

        rows.append({
            "run": str(rec["_file"]),
            "query_label": query_label,
            "gt_expert": gt_expert,
            "gt_in_pool": gt_in_pool,
            "gt_rank": gt_rank,
            "gt_attn": gt_attn_val,
            "top1_expert": top1_name,
            "top1_attn": top1_attn,
            "top3_experts": " | ".join(top_names[:3]),
            "top5_experts": " | ".join(top_names[:5]),
            "norm_entropy": norm_entropy,
            "pool_size": N,
            "temperature": rec.get("temperature"),
            "top_k": rec.get("top_k"),
            "exclude_experts": " ".join(exclude_experts),
        })

        if gt_expert:
            total_with_gt += 1
            if gt_in_pool:
                gt_in_pool_count += 1
                if gt_rank == 1:
                    gt_top1_count += 1
                if gt_rank is not None and gt_rank <= 3:
                    gt_top3_count += 1

    print(f"{'='*100}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"AGGREGATE METRICS  ({len(records)} runs, {total_with_gt} with GT label)")
    print(f"{'─'*60}")
    total = len(records)
    if total_with_gt > 0:
        print(f"  GT in pool:       {gt_in_pool_count}/{total_with_gt}  "
              f"({100*gt_in_pool_count/total_with_gt:.1f}%)")
    in_pool = [r for r in rows if r["gt_in_pool"]]
    if in_pool:
        top1 = sum(1 for r in in_pool if r["gt_rank"] == 1)
        top3 = sum(1 for r in in_pool if r["gt_rank"] and r["gt_rank"] <= 3)
        print(f"  GT top-1 accuracy: {top1}/{len(in_pool)} = {100*top1/len(in_pool):.1f}%")
        print(f"  GT top-3 accuracy: {top3}/{len(in_pool)} = {100*top3/len(in_pool):.1f}%")
        mean_rank = sum(r["gt_rank"] for r in in_pool if r["gt_rank"]) / len(in_pool)
        print(f"  GT mean rank:      {mean_rank:.2f}")

    # ── Per-run top-5 detail (verbose) ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("TOP-5 ROUTING DETAIL PER RUN")
    print(f"{'─'*60}")
    for rec in records:
        A = rec["attention"]
        pool_names = rec["pool_names"]
        avg_attn = A.mean(dim=1).tolist()
        ranked = sorted(range(len(pool_names)), key=lambda i: avg_attn[i], reverse=True)
        label = rec.get("query_label") or Path(rec["_file"]).parent.name
        gt = rec.get("gt_expert") or "—"
        held_out = rec.get("exclude_experts") or []

        print(f"\n  {'Run':>16}: {str(rec['_file'].parent)[-60:]}")
        print(f"  {'GT expert':>16}: {gt}")
        print(f"  {'GT in pool?':>16}: {'YES' if rec.get('gt_in_pool') else 'HELD OUT'}")
        print(f"  {'Excluded':>16}: {held_out if held_out else '—'}")
        for rank_i in range(min(5, len(ranked))):
            i = ranked[rank_i]
            marker = " ← GT" if gt and gt in pool_names[i] else ""
            print(f"  {'':16}  #{rank_i+1}: {pool_names[i]:50s}  {avg_attn[i]:.4f}{marker}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[saved] CSV report: {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root directory to search for *_attention.pt files")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to save CSV summary table")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_attention_files(results_dir)
    print(f"[analyse] Found {len(records)} attention files in {results_dir}")

    if not records:
        print("No attention files found — nothing to analyse.")
        return

    analyse(records, top_k_report=args.top_k,
            csv_path=Path(args.csv) if args.csv else None)


if __name__ == "__main__":
    main()
