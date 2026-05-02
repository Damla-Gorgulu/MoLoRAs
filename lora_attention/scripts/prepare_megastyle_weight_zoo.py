#!/usr/bin/env python3
"""Select k=1000 diverse images from MegaStyle cap-1 pool, export them, and split into 4 training-job manifests."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_path",
                   default="/scratch/eyavuz21/datasets/MegaStyle-diverse/full/embeddings/embeddings.pt")
    p.add_argument("--metadata_path",
                   default="/scratch/eyavuz21/datasets/MegaStyle-diverse/full/embeddings/metadata.json")
    p.add_argument("--dataset_root",
                   default="/scratch/eyavuz21/datasets/MegaStyle-1.4M")
    p.add_argument("--output_root",
                   default="/scratch/eyavuz21/blora_megastyle_zoo_v1")
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--num_jobs", type=int, default=4)
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


def farthest_point_selection(emb: torch.Tensor, k: int) -> list[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = torch.nn.functional.normalize(emb.float(), dim=-1).to(device)
    n = emb.shape[0]
    k = min(k, n)

    centroid = torch.nn.functional.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
    centroid_sim = (emb @ centroid.T).squeeze(1)
    first = int(torch.argmin(centroid_sim).item())

    selected = [first]
    nearest_sim = emb @ emb[first]
    nearest_sim[first] = 1.0

    for step in range(1, k):
        candidate = int(torch.argmin(nearest_sim).item())
        selected.append(candidate)
        sim_new = emb @ emb[candidate]
        nearest_sim = torch.maximum(nearest_sim, sim_new)
        nearest_sim[selected] = 1.0
        if step % 100 == 0 or step == k - 1:
            print(f"[fps] picked {step+1}/{k}")

    return selected


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. Farthest-point selection ----
    print("[prep] loading embeddings ...")
    emb = torch.load(args.embeddings_path, map_location="cpu")
    meta = json.loads(Path(args.metadata_path).read_text())

    print(f"[prep] selecting k={args.k} from {emb.shape[0]} embeddings ...")
    selected_indices = farthest_point_selection(emb, args.k)

    selection_rows = []
    for rank, idx in enumerate(selected_indices):
        selection_rows.append({
            "rank": rank,
            "embedding_index": idx,
            "dataset_index": meta["dataset_indices"][idx],
            "id": meta["ids"][idx],
            "style": meta["styles"][idx],
            "content": meta["contents"][idx],
        })

    sel_dir = out_root / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "selection.json").write_text(json.dumps(selection_rows, indent=2, sort_keys=True) + "\n")
    print(f"[prep] saved selection: {len(selection_rows)} rows")

    # ---- 2. Export images ----
    print("[prep] loading MegaStyle dataset (arrow) ...")
    from datasets import Image as HFImage
    from datasets import load_from_disk

    indices = [int(row["dataset_index"]) for row in selection_rows]
    ds_dict = load_from_disk(args.dataset_root)
    ds = ds_dict["train"] if hasattr(ds_dict, "keys") and "train" in ds_dict else ds_dict
    ds = ds.cast_column("image", HFImage(decode=False))
    subset = ds.select(indices)

    img_dir = out_root / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for row_meta, row_ds in zip(selection_rows, subset):
        image = decode_image(row_ds["image"]).resize((args.image_size, args.image_size), Image.BICUBIC)
        filename = f"{row_meta['rank']:04d}_{row_meta['id']}.jpg"
        dst = img_dir / filename
        image.save(dst, quality=95)
        row_meta["image_path"] = str(dst)
    print(f"[prep] exported {len(selection_rows)} images to {img_dir}")

    # ---- 3. Create per-style training folders and job manifests ----
    style_dir = out_root / "style_images"
    style_dir.mkdir(parents=True, exist_ok=True)

    jobs = [[] for _ in range(args.num_jobs)]
    for i, row in enumerate(selection_rows):
        style_name = f"style_{row['rank']:04d}"
        folder = style_dir / style_name
        folder.mkdir(parents=True, exist_ok=True)

        src = Path(row["image_path"])
        dst = folder / f"{style_name}.jpg"
        import shutil
        shutil.copy2(src, dst)

        row["style_folder"] = str(folder)
        job_idx = i % args.num_jobs
        jobs[job_idx].append(row)

    manifests_dir = out_root / "job_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for ji, job_rows in enumerate(jobs):
        mf = {
            "job_index": ji,
            "total_jobs": args.num_jobs,
            "num_styles": len(job_rows),
            "styles": [{
                "style_name": f"style_{r['rank']:04d}",
                "style_folder": r["style_folder"],
                "style_desc": r["style"],
            } for r in job_rows],
        }
        (manifests_dir / f"job_{ji}.json").write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")
        print(f"[prep] job_{ji}: {len(job_rows)} styles")

    summary = {
        "k": args.k,
        "num_jobs": args.num_jobs,
        "output_root": str(out_root),
        "total_styles": len(selection_rows),
    }
    (out_root / "zoo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[prep] done. {len(selection_rows)} styles, {args.num_jobs} jobs")


if __name__ == "__main__":
    main()
