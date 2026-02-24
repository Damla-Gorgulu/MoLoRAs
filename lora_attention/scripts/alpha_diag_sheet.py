"""
alpha_diag_sheet.py — Build a contact sheet from alpha_diag.sh results.

Layout (single row per run, 2 columns):
  Col 0: run label
  Col 1: output image

Rows (in order):
  vanilla | ref_blora | synth alpha sweep | oracle alpha sweep | norm_match variants

Usage:
    python lora_attention/scripts/alpha_diag_sheet.py \
        --run_dir /scratch/eyavuz21/lora_attention/alpha_diag \
        --query   /path/to/query.jpg \
        --out     /scratch/eyavuz21/lora_attention/alpha_diag/contact_sheet.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Ordered run list (must match alpha_diag.sh exactly) ─────────────────────
RUN_ORDER = [
    ("vanilla",               "Vanilla SDXL (no LoRA)"),
    ("ref_blora_a1",          "Reference B-LoRA  α=1.0"),
    ("synth_t0.005_a0.5",     "Synth  τ=0.005  α=0.5"),
    ("synth_t0.005_a1.0",     "Synth  τ=0.005  α=1.0"),
    ("synth_t0.005_a2.0",     "Synth  τ=0.005  α=2.0"),
    ("synth_t0.005_a3.0",     "Synth  τ=0.005  α=3.0"),
    ("synth_t0.005_a5.0",     "Synth  τ=0.005  α=5.0"),
    ("synth_oracle_a1.0",     "Oracle top_k=1   α=1.0"),
    ("synth_oracle_a3.0",     "Oracle top_k=1   α=3.0"),
    ("synth_oracle_a5.0",     "Oracle top_k=1   α=5.0"),
    ("synth_normmatch",       "Synth  norm_match  α=1.0"),
    ("synth_oracle_normmatch","Oracle norm_match  α=1.0"),
]

THUMB_W = 512
THUMB_H = 512
LABEL_W = 300
PAD     = 8
FONT_SIZE = 20


def load_thumb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    # pad to fixed size
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (30, 30, 30))
    x_off = (THUMB_W - img.width) // 2
    y_off = (THUMB_H - img.height) // 2
    canvas.paste(img, (x_off, y_off))
    return canvas


def find_output(run_dir: Path, tag: str) -> Path | None:
    """Find the single generated image in a run dir."""
    d = run_dir / tag
    if not d.exists():
        return None
    # Skip reference/query images (prefixed with __)
    candidates = [p for p in d.glob("*.jpg") if not p.name.startswith("__")]
    if not candidates:
        return None
    # Prefer the one ending _0.jpg
    for c in candidates:
        if c.name.endswith("_0.jpg"):
            return c
    return candidates[0]


def make_label_tile(text: str, width: int, height: int) -> Image.Image:
    tile = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    # Word-wrap
    words = text.split()
    lines = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > width - 2 * PAD and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    total_h = len(lines) * (FONT_SIZE + 4)
    y = (height - total_h) // 2
    for l in lines:
        bbox = draw.textbbox((0, 0), l, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), l, fill=(240, 240, 240), font=font)
        y += FONT_SIZE + 4
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--query",   required=True)
    ap.add_argument("--out",     required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)

    query_thumb = load_thumb(Path(args.query))

    # Header: query image + label
    rows = []
    header_label = make_label_tile("QUERY IMAGE", LABEL_W, THUMB_H)
    header_row = Image.new("RGB", (LABEL_W + PAD + THUMB_W, THUMB_H), (0, 0, 0))
    header_row.paste(header_label, (0, 0))
    header_row.paste(query_thumb, (LABEL_W + PAD, 0))
    rows.append(header_row)

    # Separator
    sep = Image.new("RGB", (LABEL_W + PAD + THUMB_W, PAD), (60, 60, 60))
    rows.append(sep)

    missing = []
    for tag, label in RUN_ORDER:
        img_path = find_output(run_dir, tag)
        if img_path is None:
            missing.append(tag)
            # placeholder
            out_thumb = Image.new("RGB", (THUMB_W, THUMB_H), (50, 0, 0))
            draw = ImageDraw.Draw(out_thumb)
            draw.text((PAD, THUMB_H // 2), f"MISSING\n{tag}", fill=(255, 80, 80))
        else:
            out_thumb = load_thumb(img_path)

        label_tile = make_label_tile(label, LABEL_W, THUMB_H)
        row = Image.new("RGB", (LABEL_W + PAD + THUMB_W, THUMB_H), (0, 0, 0))
        row.paste(label_tile, (0, 0))
        row.paste(out_thumb, (LABEL_W + PAD, 0))
        rows.append(row)
        rows.append(Image.new("RGB", (LABEL_W + PAD + THUMB_W, PAD), (40, 40, 40)))

    total_h = sum(r.height for r in rows)
    total_w = LABEL_W + PAD + THUMB_W
    sheet = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)
    print(f"Saved contact sheet → {out_path}  ({sheet.width}×{sheet.height})")

    if missing:
        print(f"[warn] Missing runs: {missing}")


if __name__ == "__main__":
    main()
