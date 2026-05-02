#!/usr/bin/env python3
"""Routing-only validation for MoELoRA v3 mini checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.data.dataset import ExactExemplarStage1Dataset, exact_stage1_collate_fn
from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v3 import MoELoRAv3
from lora_attention.train_stage1_v3_mini import compute_ce_loss


def parse_args():
    p = argparse.ArgumentParser(description="Validate MoELoRA v3 mini routing")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--views_per_style", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    pool = LoRAPool(
        zoo_dir=cfg["zoo_dir"],
        cache_dir=cfg["cache_dir"],
        force_rebuild=False,
        device=str(device),
    )
    model = MoELoRAv3(
        pool=pool,
        image_encoder=cfg["image_encoder"],
        rank_tokens=cfg["rank_tokens"],
        max_tensor_groups=cfg["max_tensor_groups"],
        d_model=cfg["d_model"],
        query_layers=cfg["query_layers"],
        key_layers=cfg["key_layers"],
        num_heads=cfg["num_heads"],
        clip_model_id=cfg["clip_model_id"],
        vae_model_id=cfg["vae_model_id"],
    ).to(device)
    query_state = {
        k: v
        for k, v in ckpt["query_encoder_state_dict"].items()
        if not k.startswith("_clip_model.") and not k.startswith("_vae.")
    }
    model.query_encoder.load_state_dict(query_state, strict=False)
    model.key_encoder.load_state_dict(ckpt["key_encoder_state_dict"])
    model.eval()

    dataset = ExactExemplarStage1Dataset(
        pool=pool,
        manifest_path=cfg["manifest_path"],
        min_pool_size=cfg["min_pool_size"],
        max_pool_size=cfg["max_pool_size"],
        views_per_style=args.views_per_style,
        seed=args.seed,
        deterministic_augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=exact_stage1_collate_fn,
    )

    total = 0
    top1_correct = 0
    losses = []
    entropies = []
    gt_ranks = []
    per_style = defaultdict(lambda: {"n": 0, "top1": 0, "loss": 0.0, "entropy": 0.0})
    layer_votes = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in loader:
            image = batch["images"][0]
            pool_indices = batch["pool_indices"][0]
            gt_pos = batch["gt_positions"][0]
            style_name = batch["style_names"][0]

            A, meta = model(
                image,
                pool_indices,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            loss = compute_ce_loss(A, gt_pos, device).item()
            entropy = model.attention_entropy(A).item()
            avg = A.mean(dim=(1, 2))
            pred_pos = int(avg.argmax().item())
            gt_attn = avg[gt_pos].item()
            gt_rank = int((avg > gt_attn).sum().item()) + 1

            total += 1
            top1_correct += int(pred_pos == gt_pos)
            losses.append(loss)
            entropies.append(entropy)
            gt_ranks.append(gt_rank)

            per_style[style_name]["n"] += 1
            per_style[style_name]["top1"] += int(pred_pos == gt_pos)
            per_style[style_name]["loss"] += loss
            per_style[style_name]["entropy"] += entropy

            per_tensor = A.mean(dim=2).argmax(dim=0).tolist()
            for t_idx, pos in enumerate(per_tensor):
                expert_name = pool.style_names[pool_indices[pos]]
                layer_votes[style_name][f"t{t_idx}:{expert_name}"] += 1

    summary = {
        "checkpoint": args.checkpoint,
        "image_encoder": cfg["image_encoder"],
        "temperature": args.temperature,
        "top_k": args.top_k,
        "samples": total,
        "loss": sum(losses) / max(len(losses), 1),
        "entropy": sum(entropies) / max(len(entropies), 1),
        "entropy_ratio": (sum(entropies) / max(len(entropies), 1)) / math.log(cfg["max_pool_size"]),
        "top1_acc": top1_correct / max(total, 1),
        "mean_gt_rank": sum(gt_ranks) / max(len(gt_ranks), 1),
        "per_style": {
            k: {
                "n": v["n"],
                "top1_acc": v["top1"] / max(v["n"], 1),
                "loss": v["loss"] / max(v["n"], 1),
                "entropy": v["entropy"] / max(v["n"], 1),
            }
            for k, v in sorted(per_style.items())
        },
        "layer_votes": {k: dict(v) for k, v in sorted(layer_votes.items())},
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
