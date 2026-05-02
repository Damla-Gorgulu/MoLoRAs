#!/usr/bin/env python3
"""
Prepare a mini exact-exemplar experiment.

Each query image is the exact B-LoRA style-image exemplar for the corresponding
expert, with mild style-preserving augmentations applied during training.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/scratch/eyavuz21/lora_attention/mini_exact_v1")
    p.add_argument("--zoo_src", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--style_images_src", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images")
    p.add_argument(
        "--styles",
        nargs="+",
        default=["Baroque", "Cubism", "Impressionism", "Expressionism"],
    )
    p.add_argument("--train_views_per_style", type=int, default=32)
    p.add_argument("--val_views_per_style", type=int, default=8)
    return p.parse_args()


def find_exact_dir(zoo_src: Path, style: str) -> Path:
    for path in sorted(zoo_src.iterdir()):
        if not path.is_dir():
            continue
        parts = path.name.split("_", 2)
        if len(parts) == 3 and parts[2] == style:
            return path
    raise RuntimeError(f"Could not find exact zoo dir for style {style}")


def main():
    args = parse_args()
    root = Path(args.root)
    zoo_src = Path(args.zoo_src)
    style_images_src = Path(args.style_images_src)

    zoo_dst = root / "zoo" / "bloras"
    sources_dst = root / "sources"
    outputs_dst = root / "outputs"
    cache_dst = root / "cache"
    logs_dst = root / "logs"

    for d in [zoo_dst, sources_dst, outputs_dst, cache_dst, logs_dst]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "root": str(root),
        "styles": [],
        "train_views_per_style": args.train_views_per_style,
        "val_views_per_style": args.val_views_per_style,
    }

    for idx, style in enumerate(args.styles):
        zoo_dir = find_exact_dir(zoo_src, style)
        src_img = style_images_src / zoo_dir.name / f"{zoo_dir.name}.jpg"
        if not src_img.exists():
            raise RuntimeError(f"Missing exemplar image: {src_img}")

        dst_zoo_dir = zoo_dst / zoo_dir.name
        if not dst_zoo_dir.exists():
            dst_zoo_dir.symlink_to(zoo_dir)

        dst_src_dir = sources_dst / zoo_dir.name
        dst_src_dir.mkdir(parents=True, exist_ok=True)
        link = dst_src_dir / src_img.name
        if not link.exists():
            link.symlink_to(src_img)

        manifest["styles"].append(
            {
                "category": style,
                "expert_name": zoo_dir.name,
                "gt_idx": idx,
                "source_image": str(src_img),
            }
        )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[mini-exact] root: {root}")
    print(f"[mini-exact] zoo: {zoo_dst}")
    print(f"[mini-exact] sources: {sources_dst}")
    print(f"[mini-exact] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
