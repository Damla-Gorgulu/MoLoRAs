#!/usr/bin/env python3
"""
Prepare a tiny, deterministic mini experiment for fast routing/debug cycles.

This creates:
  - a mini LoRA zoo made of symlinks to a small set of experts
  - a train/val split of WikiArt images for those same styles
  - a remapped label map that matches the mini pool order
  - a manifest describing the resulting split

The goal is not to be exhaustive. The goal is to make the pipeline fail fast
or prove that it can learn on a very small, seen-style subset.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/scratch/eyavuz21/lora_attention/mini_v1",
        help="Mini experiment root.",
    )
    p.add_argument(
        "--wikiart_src",
        default="/home/eyavuz21/datasets/wikiart",
        help="Source WikiArt tree.",
    )
    p.add_argument(
        "--zoo_src",
        default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras",
        help="Source B-LoRA zoo tree.",
    )
    p.add_argument(
        "--styles",
        nargs="+",
        default=["Baroque", "Cubism", "Impressionism", "Expressionism"],
        help="WikiArt categories to include in the mini run.",
    )
    p.add_argument(
        "--train_per_style",
        type=int,
        default=32,
        help="Training images per style.",
    )
    p.add_argument(
        "--val_per_style",
        type=int,
        default=8,
        help="Validation images per style.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refusing to overwrite directory: {dst}")
        dst.unlink()
    dst.symlink_to(src)


def list_images(style_dir: Path) -> List[Path]:
    imgs = [
        p for p in sorted(style_dir.iterdir())
        if p.is_file() and p.suffix in IMG_EXTS
    ]
    return imgs


def find_exact_zoo_style(zoo_src: Path, style: str) -> Path:
    """
    Find a B-LoRA zoo directory whose suffix exactly matches the WikiArt style.

    Zoo folders are named like `style_0010_Expressionism`.
    We match on the suffix after the second underscore so that
    `Expressionism` does not collide with `Abstract_Expressionism`.
    """
    for path in sorted(zoo_src.iterdir()):
        if not path.is_dir():
            continue
        parts = path.name.split("_", 2)
        if len(parts) == 3 and parts[2] == style:
            return path
    raise RuntimeError(f"No zoo expert found for exact style '{style}'")


def write_image_split(
    src_dir: Path,
    dst_dir: Path,
    images: Sequence[Path],
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        dst = dst_dir / img.name
        ensure_symlink(img, dst)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    root = Path(args.root)
    wikiart_src = Path(args.wikiart_src)
    zoo_src = Path(args.zoo_src)

    zoo_dst = root / "zoo" / "bloras"
    train_root = root / "wikiart_train"
    val_root = root / "wikiart_val"
    cache_dir = root / "cache"
    output_root = root / "outputs"
    log_dir = root / "logs"

    for d in [zoo_dst, train_root, val_root, cache_dir, output_root, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    selected_zoo_dirs: List[str] = []
    label_map: Dict[str, List[int]] = {}
    manifest: Dict[str, object] = {
        "root": str(root),
        "seed": args.seed,
        "styles": [],
        "train_per_style": args.train_per_style,
        "val_per_style": args.val_per_style,
        "zoo_dirs": [],
        "wikiart_splits": {},
    }

    for idx, style in enumerate(args.styles):
        zoo_style = find_exact_zoo_style(zoo_src, style)

        ensure_symlink(zoo_style, zoo_dst / zoo_style.name)
        selected_zoo_dirs.append(zoo_style.name)
        label_map[style] = [idx]

        src_style_dir = wikiart_src / style
        if not src_style_dir.is_dir():
            raise RuntimeError(f"WikiArt category not found: {src_style_dir}")
        imgs = list_images(src_style_dir)
        if len(imgs) < args.train_per_style + args.val_per_style:
            raise RuntimeError(
                f"Not enough images in {style}: have {len(imgs)}, "
                f"need {args.train_per_style + args.val_per_style}"
            )

        rng.shuffle(imgs)
        train_imgs = imgs[:args.train_per_style]
        val_imgs = imgs[args.train_per_style:args.train_per_style + args.val_per_style]

        write_image_split(src_style_dir, train_root / style, train_imgs)
        write_image_split(src_style_dir, val_root / style, val_imgs)

        manifest["styles"].append({
            "category": style,
            "zoo_dir": zoo_style.name,
            "train_count": len(train_imgs),
            "val_count": len(val_imgs),
        })
        manifest["wikiart_splits"][style] = {
            "train": [str(p) for p in train_imgs],
            "val": [str(p) for p in val_imgs],
        }

    label_map_path = root / "wikiart_label_map_mini.json"
    label_map_path.write_text(json.dumps(label_map, indent=2, sort_keys=True) + "\n")

    manifest_path = root / "manifest.json"
    manifest["selected_zoo_dirs"] = selected_zoo_dirs
    manifest["label_map_path"] = str(label_map_path)
    manifest["zoo_root"] = str(zoo_dst)
    manifest["train_root"] = str(train_root)
    manifest["val_root"] = str(val_root)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[mini-setup] root: {root}")
    print(f"[mini-setup] zoo: {zoo_dst}")
    print(f"[mini-setup] train: {train_root}")
    print(f"[mini-setup] val: {val_root}")
    print(f"[mini-setup] label map: {label_map_path}")
    print(f"[mini-setup] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
