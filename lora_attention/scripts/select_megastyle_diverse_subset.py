#!/usr/bin/env python3
"""Select a diverse subset from cached CLIP embeddings using farthest-point sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_path", required=True)
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = torch.load(args.embeddings_path, map_location="cpu").float()
    emb = torch.nn.functional.normalize(emb, dim=-1).to(device)
    meta = json.loads(Path(args.metadata_path).read_text())

    n = emb.shape[0]
    k = min(args.k, n)
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
        if step % 25 == 0 or step == k - 1:
            print(f"[diverse-select] picked {step+1}/{k}")

    selected_rows = []
    for rank, idx in enumerate(selected):
        selected_rows.append(
            {
                "rank": rank,
                "embedding_index": idx,
                "dataset_index": meta["dataset_indices"][idx],
                "id": meta["ids"][idx],
                "style": meta["styles"][idx],
                "content": meta["contents"][idx],
            }
        )

    summary = {
        "count": len(selected_rows),
        "source_count": n,
        "k": k,
        "first_seed_index": first,
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "selection.json").write_text(json.dumps(selected_rows, indent=2, sort_keys=True) + "\n")
    with (out_dir / "selection.csv").open("w") as f:
        f.write("rank,embedding_index,dataset_index,id,style,content\n")
        for row in selected_rows:
            def esc(x: str) -> str:
                x = str(x).replace('"', '""')
                return f'"{x}"'
            f.write(
                f"{row['rank']},{row['embedding_index']},{row['dataset_index']},"
                f"{esc(row['id'])},{esc(row['style'])},{esc(row['content'])}\n"
            )
    print(f"[diverse-select] saved {len(selected_rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
