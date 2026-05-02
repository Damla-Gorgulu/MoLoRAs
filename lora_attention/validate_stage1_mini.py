#!/usr/bin/env python3
"""
Lightweight routing-only validation for the mini experiment.

This is intentionally cheaper than SDXL inference. It measures whether the
router can retrieve the correct expert on held-out images from seen styles.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.data.dataset import WikiArtStage1Dataset, wikiart_stage1_collate_fn
from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.train_stage1_v2 import compute_ce_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zoo_dir", default="/scratch/eyavuz21/lora_attention/mini_v1/zoo/bloras")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention/mini_v1/cache")
    p.add_argument("--wikiart_dir", default="/scratch/eyavuz21/lora_attention/mini_v1/wikiart_val")
    p.add_argument("--label_map_path", default="/scratch/eyavuz21/lora_attention/mini_v1/wikiart_label_map_mini.json")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_attention/mini_v1/validation_stage1")
    p.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--min_pool_size", type=int, default=4)
    p.add_argument("--max_pool_size", type=int, default=4)
    p.add_argument("--max_images_per_style", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-val] device: {device}")

    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=False,
        device=str(device),
    )
    model = MoELoRAv2(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.encoder.eval()
    model._ensure_clip(device)

    dataset = WikiArtStage1Dataset(
        pool=pool,
        wikiart_dir=args.wikiart_dir,
        label_map_path=args.label_map_path,
        similarity_path=None,
        tau_label=0.3,
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
        max_images_per_style=args.max_images_per_style,
        rank=args.rank,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=wikiart_stage1_collate_fn,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0
    total_entropy = 0.0
    total_samples = 0
    top1_correct = 0
    gt_ranks = []
    per_style = defaultdict(lambda: {"n": 0, "loss": 0.0, "entropy": 0.0, "top1": 0})

    with torch.no_grad():
        for batch in loader:
            image = batch["images"][0]
            pool_indices = batch["pool_indices"][0]
            gt_pos = batch["gt_positions"][0]
            style_name = batch["style_names"][0]

            q = model.encode_image(image, device)
            A, _ = model.forward(
                q,
                pool_indices,
                temperature=args.temperature,
                top_k=None,
                product_space=True,
                synthesise=False,
            )

            loss = compute_ce_loss(A, gt_pos, device)
            entropy = model.attention_entropy(A).item()
            avg_attn_global = A.mean(dim=(1, 2))
            pred_pos = int(avg_attn_global.argmax().item())
            gt_attn = avg_attn_global[gt_pos].item()
            gt_rank = int((avg_attn_global > gt_attn).sum().item()) + 1

            total_loss += loss.item()
            total_entropy += entropy
            total_samples += 1
            top1_correct += int(pred_pos == gt_pos)
            gt_ranks.append(gt_rank)

            per_style[style_name]["n"] += 1
            per_style[style_name]["loss"] += loss.item()
            per_style[style_name]["entropy"] += entropy
            per_style[style_name]["top1"] += int(pred_pos == gt_pos)

    mean_loss = total_loss / max(total_samples, 1)
    mean_entropy = total_entropy / max(total_samples, 1)
    mean_gt_rank = mean(gt_ranks) if gt_ranks else float("nan")
    top1_acc = top1_correct / max(total_samples, 1)
    max_entropy = math.log(max(args.max_pool_size, 2))
    entropy_ratio = mean_entropy / max_entropy if max_entropy > 0 else float("nan")

    summary = {
        "checkpoint": args.checkpoint,
        "samples": total_samples,
        "pool_size": args.max_pool_size,
        "temperature": args.temperature,
        "loss": mean_loss,
        "entropy": mean_entropy,
        "entropy_ratio": entropy_ratio,
        "top1_acc": top1_acc,
        "mean_gt_rank": mean_gt_rank,
        "per_style": {
            k: {
                "n": v["n"],
                "loss": v["loss"] / max(v["n"], 1),
                "entropy": v["entropy"] / max(v["n"], 1),
                "top1_acc": v["top1"] / max(v["n"], 1),
            }
            for k, v in sorted(per_style.items())
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    verdict = "bad"
    if top1_acc >= 0.75 and mean_entropy <= 0.8:
        verdict = "good"
    elif top1_acc >= 0.50 or mean_gt_rank <= 2.0:
        verdict = "partial"

    print(f"[mini-val] samples={total_samples}")
    print(f"[mini-val] loss={mean_loss:.6f}")
    print(f"[mini-val] entropy={mean_entropy:.4f} (ratio={entropy_ratio:.3f} of max)")
    print(f"[mini-val] top1_acc={top1_acc:.3f}")
    print(f"[mini-val] mean_gt_rank={mean_gt_rank:.3f}")
    print(f"[mini-val] verdict={verdict}")
    print(f"[mini-val] summary={summary_path}")

    kb_path = Path("/home/eyavuz21/repos/MoLoRAs/lora_attention/experiment_kb.md")
    jsonl_path = Path("/home/eyavuz21/repos/MoLoRAs/lora_attention/experiment_kb.jsonl")
    try:
        import subprocess

        subprocess.run(
            [
                "/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python",
                "/home/eyavuz21/repos/MoLoRAs/lora_attention/log_experiment.py",
                "--kb_path", str(kb_path),
                "--jsonl_path", str(jsonl_path),
                "--stage", "mini_stage1_validation",
                "--job_id", "local",
                "--run_tag", out_dir.name,
                "--checkpoint", args.checkpoint,
                "--synthesis", "routing_only",
                "--prompt_type", "seen_styles",
                "--result", f"samples={total_samples} top1_acc={top1_acc:.3f} loss={mean_loss:.6f} entropy={mean_entropy:.4f}",
                "--verdict", verdict,
                "--notes", "Mini routing-only validation on held-out seen-style images.",
            ],
            check=True,
        )
    except Exception as exc:  # pragma: no cover - logging should not block the run
        print(f"[mini-val] warning: could not append experiment KB: {exc}")


if __name__ == "__main__":
    main()
