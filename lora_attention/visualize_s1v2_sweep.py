#!/usr/bin/env python3
"""
Consolidated visualization for the S1-v2 inference sweep.

Produces a single large PDF/PNG showing:
  - Row per (style × image_source) = 8 rows
  - Columns: query img | reference B-LoRA | then one column per (temp, topk) combo
  - Annotation: GT rank, top-1 expert name, entropy inside each cell
  - Bottom: bar-charts of entropy & GT-rank as functions of temperature for each top-k

Usage:
    python lora_attention/visualize_s1v2_sweep.py \\
        --sweep_dir /scratch/eyavuz21/lora_attention/s1v2_sweep \\
        --output    /scratch/eyavuz21/lora_attention/s1v2_sweep/consolidated.pdf
"""

import argparse
import math
import re
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


# ─────────────────────────────────── helpers ──────────────────────────────────

def load_attention(pt_path: Path) -> dict:
    """Load an attention .pt and compute summary stats."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    A = d["attention"]
    pool_names = d.get("pool_names", [])
    gt_expert = d.get("gt_expert", "")

    # avg attention per expert
    if A.dim() == 3:
        avg = A.mean(dim=(1, 2))  # (N,)
    else:
        avg = A.mean(dim=1)       # (N,)

    N = avg.shape[0]
    top1_idx = avg.argmax().item()
    top1_name = pool_names[top1_idx] if top1_idx < len(pool_names) else "?"
    top1_val = avg[top1_idx].item()

    entropy = -((avg * (avg + 1e-10).log()).sum()).item()
    max_entropy = math.log(N) if N > 1 else 1.0

    gt_rank = None
    if gt_expert:
        for i, nm in enumerate(pool_names):
            if gt_expert in nm:
                gt_rank = int((avg > avg[i]).sum().item()) + 1
                break

    # Top-5
    top5_vals, top5_idxs = avg.topk(min(5, N))
    top5 = [
        (pool_names[idx] if idx < len(pool_names) else "?", val.item())
        for idx, val in zip(top5_idxs.tolist(), top5_vals.tolist())
    ]

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "top1_name": top1_name,
        "top1_val": top1_val,
        "gt_rank": gt_rank,
        "gt_expert": gt_expert,
        "top5": top5,
        "N": N,
    }


def short_style(name: str) -> str:
    """style_0000_Baroque -> Baroque"""
    m = re.match(r"style_\d+_(.*)", name)
    return m.group(1).replace("_", " ") if m else name.replace("_", " ")


def load_first_jpg(folder: Path) -> Image.Image | None:
    """Load the first generated image (not __query/__top) from a folder."""
    candidates = sorted(
        f for f in folder.glob("*.jpg")
        if "__query" not in f.name and "__top" not in f.name
           and "_heatmap" not in f.name
    )
    if candidates:
        return Image.open(candidates[0]).convert("RGB")
    return None


def load_query_jpg(folder: Path) -> Image.Image | None:
    candidates = sorted(folder.glob("*__query*"))
    if candidates:
        return Image.open(candidates[0]).convert("RGB")
    return None


# ─────────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str,
                        default="/scratch/eyavuz21/lora_attention/s1v2_sweep")
    parser.add_argument("--output", type=str,
                        default="/scratch/eyavuz21/lora_attention/s1v2_sweep/consolidated.pdf")
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    out_path = Path(args.output)

    # ── Discover runs ──────────────────────────────────────────
    # Expected structure: <style>/<source>/t<temp>_k<topk>_a<alpha>/
    #                     <style>/<source>/reference_blora/
    styles = sorted([d.name for d in sweep.iterdir() if d.is_dir()])
    print(f"Styles found: {styles}")

    TEMPS = [0.005, 0.01, 0.05, 0.1, 0.5]
    TOPKS = ["none", "1", "5"]
    SOURCES = ["wikiart", "pool"]

    # Collect all data
    data = {}  # (style, source, temp, topk) -> {img, stats}
    refs = {}  # (style, source) -> {img, stats}
    alpha_data = {}  # (style, alpha) -> {img, stats}
    queries = {}  # (style, source) -> query_img

    for style in styles:
        for src in SOURCES:
            src_dir = sweep / style / src
            if not src_dir.exists():
                continue
            for run_dir in src_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Find .pt
                pts = list(run_dir.glob("*_attention.pt"))
                if not pts:
                    continue
                stats = load_attention(pts[0])
                gen_img = load_first_jpg(run_dir)
                q_img = load_query_jpg(run_dir)
                if q_img and (style, src) not in queries:
                    queries[(style, src)] = q_img

                name = run_dir.name
                if name == "reference_blora":
                    refs[(style, src)] = {"img": gen_img, "stats": stats}
                else:
                    # parse t0.01_knone_a1.0
                    m = re.match(r"t([\d.]+)_k(\w+)_a([\d.]+)", name)
                    if m:
                        temp = m.group(1)
                        topk = m.group(2)
                        alpha = m.group(3)
                        if alpha == "1.0":
                            data[(style, src, temp, topk)] = {"img": gen_img, "stats": stats}
                        else:
                            alpha_data[(style, temp, topk, alpha)] = {"img": gen_img, "stats": stats}

    print(f"Loaded {len(data)} sweep runs, {len(refs)} reference runs, {len(alpha_data)} alpha runs")

    # ════════════════════════════════════════════════════════════
    # FIGURE 1: The big comparison grid
    # Rows = styles × sources (8 rows)
    # Cols = query | ref | then temp×topk combos (5×3=15)
    # ════════════════════════════════════════════════════════════

    # Column layout: query, reference, then (temp, topk) grid
    temp_topk_combos = [(t, k) for t in TEMPS for k in TOPKS]
    n_combos = len(temp_topk_combos)
    n_cols = 2 + n_combos  # query + ref + combos
    row_specs = [(s, src) for s in styles for src in SOURCES]
    n_rows = len(row_specs)

    print(f"Grid: {n_rows} rows × {n_cols} cols")

    fig_w = 2 + n_combos * 1.5
    fig_h = n_rows * 1.8 + 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)

    # Turn off all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Column headers
    col_headers = ["Query", "Ref B-LoRA"]
    for t, k in temp_topk_combos:
        kl = "soft" if k == "none" else f"k={k}"
        col_headers.append(f"τ={t}\n{kl}")

    for j, hdr in enumerate(col_headers):
        axes[0][j].set_title(hdr, fontsize=5, pad=2)

    # Row labels
    for i, (style, src) in enumerate(row_specs):
        axes[i][0].set_ylabel(f"{style}\n({src})", fontsize=5, rotation=0,
                               labelpad=50, va="center")

    THUMB = 128

    def show_img(ax, img, annotation=""):
        if img is not None:
            img_r = img.resize((THUMB, THUMB), Image.LANCZOS)
            ax.imshow(np.array(img_r))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
        if annotation:
            ax.text(0.5, -0.02, annotation, ha="center", va="top",
                    transform=ax.transAxes, fontsize=3.5,
                    color="black", fontweight="normal",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              alpha=0.85, edgecolor="none"))

    for i, (style, src) in enumerate(row_specs):
        # Col 0: query image
        q = queries.get((style, src))
        show_img(axes[i][0], q)

        # Col 1: reference B-LoRA
        ref = refs.get((style, src))
        if ref:
            ann = f"top1={short_style(ref['stats']['top1_name'])}"
            show_img(axes[i][1], ref["img"], ann)
        else:
            show_img(axes[i][1], None)

        # Cols 2+: sweep
        for j, (t, k) in enumerate(temp_topk_combos):
            key = (style, src, str(t), k)
            entry = data.get(key)
            if entry:
                s = entry["stats"]
                top1 = short_style(s["top1_name"])
                gt_str = f"GT#{s['gt_rank']}" if s["gt_rank"] else ""
                ent_pct = s["entropy"] / s["max_entropy"] * 100
                ann = f"{top1[:18]}\nent={ent_pct:.0f}% {gt_str}"
                show_img(axes[i][2 + j], entry["img"], ann)
            else:
                show_img(axes[i][2 + j], None)

    fig.suptitle("Stage-1 v2.0 Inference Sweep: Temperature × Top-k × Image Source",
                 fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.98], h_pad=0.5, w_pad=0.3)

    grid_path = out_path.with_name("grid_temp_topk.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {grid_path}")

    # ════════════════════════════════════════════════════════════
    # FIGURE 2: Line plots — Entropy & GT-rank vs temperature
    # One subplot per style, lines per (source, topk)
    # ════════════════════════════════════════════════════════════

    fig2, axes2 = plt.subplots(2, len(styles), figsize=(4 * len(styles), 6),
                                squeeze=False)

    colors = {"none": "tab:blue", "1": "tab:red", "5": "tab:green"}
    ls_map = {"wikiart": "-", "pool": "--"}

    for si, style in enumerate(styles):
        ax_ent = axes2[0][si]
        ax_gt = axes2[1][si]
        ax_ent.set_title(style.capitalize(), fontsize=9, fontweight="bold")

        for src in SOURCES:
            for topk in TOPKS:
                ents = []
                gts = []
                valid_temps = []
                for t in TEMPS:
                    entry = data.get((style, src, str(t), topk))
                    if entry:
                        s = entry["stats"]
                        ents.append(s["entropy"] / s["max_entropy"])
                        gts.append(s["gt_rank"] if s["gt_rank"] else None)
                        valid_temps.append(t)

                if not valid_temps:
                    continue

                kl = "soft" if topk == "none" else f"k={topk}"
                label = f"{src},{kl}"
                ax_ent.plot(valid_temps, ents, marker="o", markersize=3,
                           color=colors[topk], linestyle=ls_map[src],
                           label=label, linewidth=1.2)

                gt_valid = [(t, g) for t, g in zip(valid_temps, gts) if g is not None]
                if gt_valid:
                    gt_t, gt_v = zip(*gt_valid)
                    ax_gt.plot(gt_t, gt_v, marker="s", markersize=3,
                              color=colors[topk], linestyle=ls_map[src],
                              label=label, linewidth=1.2)

        ax_ent.set_xscale("log")
        ax_ent.set_ylabel("Entropy / max" if si == 0 else "")
        ax_ent.set_xlabel("Temperature (τ)")
        ax_ent.set_ylim(0, 1.05)
        ax_ent.axhline(1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        if si == 0:
            ax_ent.legend(fontsize=5, loc="lower right", ncol=2)

        ax_gt.set_xscale("log")
        ax_gt.set_ylabel("GT Expert Rank" if si == 0 else "")
        ax_gt.set_xlabel("Temperature (τ)")
        ax_gt.invert_yaxis()
        ax_gt.set_ylim(bottom=max(110, 1))
        if si == 0:
            ax_gt.legend(fontsize=5, loc="lower right", ncol=2)

    fig2.suptitle("Entropy & GT-Rank vs Temperature (S1-v2, α=1.0)",
                  fontsize=10, fontweight="bold")
    fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    line_path = out_path.with_name("line_entropy_gtrank.png")
    fig2.savefig(line_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[saved] {line_path}")

    # ════════════════════════════════════════════════════════════
    # FIGURE 3: Alpha comparison grid (WikiArt only, τ=0.01, topk=soft)
    # ════════════════════════════════════════════════════════════

    ALPHAS = ["0.5", "1.0", "1.5", "2.0"]
    fig3, axes3 = plt.subplots(len(styles), len(ALPHAS) + 1,
                                figsize=(2 * (len(ALPHAS) + 1), 2 * len(styles)),
                                squeeze=False)
    for ax_row in axes3:
        for ax in ax_row:
            ax.axis("off")

    axes3[0][0].set_title("Query", fontsize=7)
    for j, a in enumerate(ALPHAS):
        axes3[0][j + 1].set_title(f"α={a}", fontsize=7)

    for i, style in enumerate(styles):
        q = queries.get((style, "wikiart"))
        show_img(axes3[i][0], q)
        axes3[i][0].set_ylabel(style.capitalize(), fontsize=7, rotation=0,
                                labelpad=40, va="center")

        for j, a in enumerate(ALPHAS):
            if a == "1.0":
                entry = data.get((style, "wikiart", "0.01", "none"))
            else:
                entry = alpha_data.get((style, "0.01", "none", a))
            if entry:
                s = entry["stats"]
                top1 = short_style(s["top1_name"])
                ann = f"{top1[:16]}"
                show_img(axes3[i][j + 1], entry["img"], ann)
            else:
                show_img(axes3[i][j + 1], None)

    fig3.suptitle("Style-Alpha Sweep (S1-v2, τ=0.01, soft routing, WikiArt queries)",
                  fontsize=9, fontweight="bold", y=0.995)
    fig3.tight_layout(rect=[0.06, 0.0, 1.0, 0.97])

    alpha_path = out_path.with_name("grid_alpha.png")
    fig3.savefig(alpha_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"[saved] {alpha_path}")

    # ════════════════════════════════════════════════════════════
    # FIGURE 4: Per-style focus summary (best config per style)
    # A compact 4×7 grid:  query | top1..top3 expert imgs | ref | best-gen | worst-gen
    # ════════════════════════════════════════════════════════════

    fig4, axes4 = plt.subplots(len(styles), 7,
                                figsize=(14, 2.5 * len(styles)),
                                squeeze=False)
    for ax_row in axes4:
        for ax in ax_row:
            ax.axis("off")

    col_labels = ["Query", "Top-1 Expert", "Top-2 Expert", "Top-3 Expert",
                  "Ref B-LoRA", "Best τ=0.005", "Worst τ=0.5"]
    for j, lbl in enumerate(col_labels):
        axes4[0][j].set_title(lbl, fontsize=6)

    for i, style in enumerate(styles):
        src = "wikiart"
        axes4[i][0].set_ylabel(style.capitalize(), fontsize=7, rotation=0,
                                labelpad=40, va="center")

        # Query
        q = queries.get((style, src))
        show_img(axes4[i][0], q)

        # Top-1..3 from best config
        best_key = (style, src, "0.005", "1")  # sharpest: low τ, k=1
        if best_key not in data:
            best_key = (style, src, "0.01", "none")
        entry = data.get(best_key)
        if entry:
            for rank_j in range(3):
                top_path = sorted(
                    (sweep / style / src / f"t{best_key[2]}_k{best_key[3]}_a1.0").glob(f"*__top{rank_j+1}_*")
                )
                if top_path:
                    top_img = Image.open(top_path[0]).convert("RGB")
                    sn = short_style(entry["stats"]["top5"][rank_j][0]) if rank_j < len(entry["stats"]["top5"]) else "?"
                    val = entry["stats"]["top5"][rank_j][1] if rank_j < len(entry["stats"]["top5"]) else 0
                    show_img(axes4[i][1 + rank_j], top_img, f"{sn[:18]}\n({val:.3f})")
                else:
                    show_img(axes4[i][1 + rank_j], None)

        # Reference B-LoRA
        ref = refs.get((style, src))
        show_img(axes4[i][4], ref["img"] if ref else None)

        # Best: τ=0.005, k=1
        best = data.get((style, src, "0.005", "1"))
        if best:
            s = best["stats"]
            ann = f"GT#{s['gt_rank']}" if s["gt_rank"] else f"top1={short_style(s['top1_name'])[:15]}"
            show_img(axes4[i][5], best["img"], ann)
        else:
            show_img(axes4[i][5], None)

        # Worst: τ=0.5, soft
        worst = data.get((style, src, "0.5", "none"))
        if worst:
            s = worst["stats"]
            ann = f"GT#{s['gt_rank']}" if s["gt_rank"] else f"top1={short_style(s['top1_name'])[:15]}"
            show_img(axes4[i][6], worst["img"], ann)
        else:
            show_img(axes4[i][6], None)

    fig4.suptitle("Per-Style Summary: Query → Retrieved Experts → Best/Worst Outputs",
                  fontsize=10, fontweight="bold", y=0.995)
    fig4.tight_layout(rect=[0.06, 0.0, 1.0, 0.97])

    summary_path = out_path.with_name("summary_per_style.png")
    fig4.savefig(summary_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print(f"[saved] {summary_path}")

    # ════════════════════════════════════════════════════════════
    # Also save a machine-readable CSV of all stats
    # ════════════════════════════════════════════════════════════
    csv_path = out_path.with_name("sweep_stats.csv")
    with open(csv_path, "w") as f:
        f.write("style,source,temperature,topk,alpha,entropy,max_entropy,entropy_pct,top1_name,top1_val,gt_rank\n")
        for (style, src, t, k), entry in sorted(data.items()):
            s = entry["stats"]
            ent_pct = s["entropy"] / s["max_entropy"] * 100
            f.write(f"{style},{src},{t},{k},1.0,{s['entropy']:.4f},{s['max_entropy']:.4f},{ent_pct:.1f},"
                    f"{short_style(s['top1_name'])},{s['top1_val']:.6f},{s['gt_rank'] or ''}\n")
        for (style, t, k, a), entry in sorted(alpha_data.items()):
            s = entry["stats"]
            ent_pct = s["entropy"] / s["max_entropy"] * 100
            f.write(f"{style},wikiart,{t},{k},{a},{s['entropy']:.4f},{s['max_entropy']:.4f},{ent_pct:.1f},"
                    f"{short_style(s['top1_name'])},{s['top1_val']:.6f},{s['gt_rank'] or ''}\n")
    print(f"[saved] {csv_path}")

    print(f"\nAll figures saved under: {out_path.parent}")


if __name__ == "__main__":
    main()
