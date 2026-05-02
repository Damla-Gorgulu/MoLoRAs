#!/usr/bin/env python3
"""Export selected MegaStyle images to disk from a selection manifest."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/scratch/eyavuz21/datasets/MegaStyle-1.4M")
    p.add_argument("--selection_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_size", type=int, default=1024)
    return p.parse_args()


def decode_image(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            return Image.open(obj["path"]).convert("RGB")
    if isinstance(obj, str):
        return Image.open(obj).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(obj)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Image as HFImage
    from datasets import load_from_disk

    selection = json.loads(Path(args.selection_path).read_text())
    indices = [int(row["dataset_index"]) for row in selection]

    ds_dict = load_from_disk(args.dataset_root)
    ds = ds_dict["train"] if hasattr(ds_dict, "keys") and "train" in ds_dict else ds_dict
    ds = ds.cast_column("image", HFImage(decode=False))
    subset = ds.select(indices)

    export_rows = []
    for row_meta, row_ds in zip(selection, subset):
        image = decode_image(row_ds["image"]).resize((args.image_size, args.image_size), Image.BICUBIC)
        filename = f"{row_meta['rank']:04d}_{row_meta['id']}.jpg"
        dst = img_dir / filename
        image.save(dst, quality=95)
        export_rows.append({**row_meta, "export_path": str(dst)})

    (out_dir / "export_manifest.json").write_text(json.dumps(export_rows, indent=2, sort_keys=True) + "\n")
    print(f"[export-subset] exported {len(export_rows)} images to {img_dir}")


if __name__ == "__main__":
    main()
