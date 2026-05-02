#!/usr/bin/env python3
"""Embed a MegaStyle manifest with CLIP image features."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/scratch/eyavuz21/datasets/MegaStyle-1.4M")
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text())


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
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Image as HFImage
    from datasets import load_from_disk
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[embed-clip] device={device}")
    manifest = load_manifest(Path(args.manifest_path))
    indices = [int(row["dataset_index"]) for row in manifest]

    ds_dict = load_from_disk(args.dataset_root)
    ds = ds_dict["train"] if hasattr(ds_dict, "keys") and "train" in ds_dict else ds_dict
    ds = ds.cast_column("image", HFImage(decode=False))
    subset = ds.select(indices)

    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embeddings = []
    ids = []
    dataset_indices = []
    styles = []
    contents = []

    for start in range(0, len(subset), args.batch_size):
        end = min(start + args.batch_size, len(subset))
        batch_rows = subset[start:end]
        images = [decode_image(x) for x in batch_rows["image"]]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().to(torch.float16))
        ids.extend(batch_rows["id"])
        styles.extend(batch_rows["style"])
        contents.extend(batch_rows["content"])
        dataset_indices.extend(indices[start:end])
        if end % (args.batch_size * 50) == 0 or end == len(subset):
            print(f"[embed-clip] processed {end:,}/{len(subset):,}")

    emb = torch.cat(embeddings, dim=0)
    torch.save(emb, out_dir / "embeddings.pt")
    meta = {
        "dataset_root": args.dataset_root,
        "manifest_path": args.manifest_path,
        "clip_model_id": args.clip_model_id,
        "count": len(dataset_indices),
        "embedding_dim": int(emb.shape[1]),
        "dataset_indices": dataset_indices,
        "ids": ids,
        "styles": styles,
        "contents": contents,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[embed-clip] saved {len(dataset_indices):,} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
