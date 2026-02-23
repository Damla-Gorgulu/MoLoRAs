#!/usr/bin/env python3
"""
Generate comparison grids from the inference sweep results.

Creates one grid per style with all configurations side-by-side.
Each row = one config, each column = one of the 4 generated images.

Usage:
    python lora_attention/make_comparison_grid.py
    python lora_attention/make_comparison_grid.py --sweep_dir /path/to/inference_sweep
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def make_grid(sweep_dir: Path, output_dir: Path):
    """Create comparison grids."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all configs and styles
    configs = sorted([
        d.name for d in sweep_dir.iterdir()
        if d.is_dir()
    ])
    
    # The order we want to display (if available)
    preferred_order = [
        "vanilla_sdxl",
        "tau1.0_noTopK",
        "tau0.5_noTopK",
        "tau0.1_noTopK",
        "tau0.01_noTopK",
        "tau0.1_top1",
        "tau0.1_top3",
        "tau0.1_top5",
        "tau1.0_alpha2.0",
        "tau0.1_alpha1.5",
        "reference_blora",
    ]
    configs = [c for c in preferred_order if c in configs] + \
              [c for c in configs if c not in preferred_order]

    # Get all style dirs from the first config
    all_styles = set()
    for config in configs:
        config_dir = sweep_dir / config
        for style_dir in config_dir.iterdir():
            if style_dir.is_dir():
                all_styles.add(style_dir.name)
    styles = sorted(all_styles)

    print(f"Configs: {len(configs)}")
    print(f"Styles:  {len(styles)}")

    for style in styles:
        print(f"\n--- Grid for {style} ---")
        rows = []
        row_labels = []

        for config in configs:
            img_dir = sweep_dir / config / style
            if not img_dir.exists():
                continue

            jpgs = sorted(img_dir.glob("*.jpg"))
            if not jpgs:
                continue

            imgs = [Image.open(p) for p in jpgs[:4]]  # max 4
            rows.append(imgs)
            row_labels.append(config)

        if not rows:
            print(f"  No images found for {style}")
            continue

        n_rows = len(rows)
        n_cols = max(len(r) for r in rows)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4, n_rows * 4),
            squeeze=False,
        )

        for i, (imgs, label) in enumerate(zip(rows, row_labels)):
            for j in range(n_cols):
                ax = axes[i][j]
                if j < len(imgs):
                    ax.imshow(imgs[j])
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(label, fontsize=10, rotation=0,
                                  labelpad=120, ha="right", va="center")

        fig.suptitle(f"Style Transfer Comparison: {style}", fontsize=14, y=1.01)
        plt.tight_layout()

        out_path = output_dir / f"grid_{style}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # Also create a "best shot" grid: 1 image per (config, style)
    print("\n--- Best-of grid ---")
    fig, axes = plt.subplots(
        len(configs), len(styles),
        figsize=(len(styles) * 4.5, len(configs) * 4.5),
        squeeze=False,
    )

    for i, config in enumerate(configs):
        for j, style in enumerate(styles):
            ax = axes[i][j]
            img_dir = sweep_dir / config / style
            jpgs = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
            if jpgs:
                ax.imshow(Image.open(jpgs[0]))
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(config, fontsize=9, rotation=0,
                              labelpad=110, ha="right", va="center")
            if i == 0:
                ax.set_title(style.replace("style_", "").replace("_", " "),
                             fontsize=10)

    fig.suptitle("MoELoRA Temperature/Top-k Sweep — Best Shot Grid", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "grid_best_shot.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str,
                        default="/scratch/eyavuz21/lora_attention/inference_sweep")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/eyavuz21/lora_attention/inference_sweep/grids")
    args = parser.parse_args()

    make_grid(Path(args.sweep_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
