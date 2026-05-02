#!/usr/bin/env python3
"""Exact-instance post-train validation for the tiny Stage 2 follow-up."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.inference_v2 import _save_attention_heatmap_v2
from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.train_stage2_v2 import load_sdxl_pipeline
from lora_attention.utils.lora_inject import inject_lora, unload_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini exact Stage 2 validation")
    p.add_argument("--checkpoint", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/outputs/stage2_exact_followup/latest.pt")
    p.add_argument("--root", default="/scratch/eyavuz21/lora_attention/mini_exact_v1")
    p.add_argument("--manifest_path", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/manifest.json")
    p.add_argument("--zoo_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/zoo/bloras")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/cache")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/outputs/stage2_exact_followup_validation")
    p.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--style_alpha", type=float, default=1.0)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _build_prompt(category: str, prompt_kind: str) -> str:
    if prompt_kind == "style":
        return f"A painting in {category} style"
    return "A detailed painting"


def _aggregate_metrics(run_results: List[Dict]) -> Dict[str, float]:
    if not run_results:
        return {"top1_acc": 0.0, "mean_gt_rank": 0.0, "entropy": 0.0}
    top1_acc = sum(r["top1_correct"] for r in run_results) / len(run_results)
    mean_gt_rank = sum(r["gt_rank"] for r in run_results) / len(run_results)
    entropy = sum(r["entropy"] for r in run_results) / len(run_results)
    return {
        "top1_acc": top1_acc,
        "mean_gt_rank": mean_gt_rank,
        "entropy": entropy,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-exact-s2-val] device: {device}")

    with open(args.manifest_path) as f:
        manifest = json.load(f)

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

    pipeline = load_sdxl_pipeline(args, device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_results: List[Dict] = []
    per_style = defaultdict(list)
    pool_indices = list(range(pool.num_experts))
    max_entropy = math.log(max(pool.num_experts, 2))

    for entry in manifest["styles"]:
        category = entry["category"]
        expert_name = entry["expert_name"]
        gt_idx = pool.index_of(expert_name)
        style_image = Image.open(entry["source_image"]).convert("RGB")
        gt_pos = pool_indices.index(gt_idx)

        for prompt_kind in ("style", "neutral"):
            prompt = _build_prompt(category, prompt_kind)
            for routing_kind, top_k in (("soft", None), ("top1", 1)):
                run_tag = f"{_slug(category)}/{prompt_kind}_{routing_kind}"
                run_dir = out_root / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)

                with torch.no_grad():
                    q = model.encode_image(style_image, device)
                    A, synth_lora = model.forward(
                        q,
                        pool_indices,
                        temperature=args.temperature,
                        top_k=top_k,
                        product_space=True,
                    )

                avg_attn_global = A.mean(dim=(1, 2))
                pred_pos = int(avg_attn_global.argmax().item())
                gt_attn = avg_attn_global[gt_pos].item()
                gt_rank = int((avg_attn_global > gt_attn).sum().item()) + 1
                entropy = float(model.attention_entropy(A).item())
                top1_correct = int(pred_pos == gt_pos)

                inject_lora(
                    pipeline=pipeline,
                    style_state_dict={k: v.detach().cpu() for k, v in synth_lora.items()},
                    style_alpha=args.style_alpha,
                )
                generator = torch.Generator(device=device).manual_seed(args.seed)
                images = pipeline(
                    prompt,
                    num_images_per_prompt=args.num_images,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    cross_attention_kwargs={"scale": 1.0},
                ).images
                unload_lora(pipeline)

                for i, img in enumerate(images):
                    img.save(run_dir / f"{_slug(run_tag)}_{i}.jpg")

                query_dst = run_dir / "query.jpg"
                style_image.save(query_dst)

                attn_path = run_dir / "attention.pt"
                torch.save(
                    {
                        "attention": A.cpu(),
                        "pool_indices": pool_indices,
                        "pool_names": [pool.style_names[i] for i in pool_indices],
                        "prompt": prompt,
                        "style_image": entry["source_image"],
                        "step": ckpt.get("step", None),
                        "temperature": args.temperature,
                        "top_k": top_k,
                        "style_alpha": args.style_alpha,
                        "gt_expert": expert_name,
                        "gt_in_pool": True,
                        "query_label": run_tag,
                        "version": "mini_exact_stage2",
                    },
                    attn_path,
                )

                heatmap_path = run_dir / "heatmap.png"
                _save_attention_heatmap_v2(
                    A.cpu(),
                    pool_indices,
                    pool.style_names,
                    model.down_key_order,
                    heatmap_path,
                    title=f"τ={args.temperature}, top_k={top_k}, {run_tag}",
                )

                run_results.append(
                    {
                        "category": category,
                        "expert_name": expert_name,
                        "prompt_kind": prompt_kind,
                        "routing_kind": routing_kind,
                        "top_k": top_k,
                        "prompt": prompt,
                        "gt_rank": gt_rank,
                        "top1_correct": top1_correct,
                        "entropy": entropy,
                        "entropy_ratio": entropy / max_entropy if max_entropy > 0 else None,
                        "pred_expert": pool.style_names[pool_indices[pred_pos]],
                        "gt_expert": expert_name,
                        "run_dir": str(run_dir),
                    }
                )
                per_style[category].append(run_results[-1])

    agg = _aggregate_metrics(run_results)
    by_prompt = defaultdict(list)
    by_routing = defaultdict(list)
    for r in run_results:
        by_prompt[r["prompt_kind"]].append(r)
        by_routing[r["routing_kind"]].append(r)

    summary = {
        "checkpoint": args.checkpoint,
        "runs": len(run_results),
        "failures": 0,
        "top1_acc": agg["top1_acc"],
        "mean_gt_rank": agg["mean_gt_rank"],
        "entropy": agg["entropy"],
        "entropy_ratio": agg["entropy"] / max_entropy if max_entropy > 0 else None,
        "per_style": {
            k: _aggregate_metrics(v) for k, v in sorted(per_style.items())
        },
        "by_prompt_kind": {
            k: _aggregate_metrics(v) for k, v in sorted(by_prompt.items())
        },
        "by_routing_kind": {
            k: _aggregate_metrics(v) for k, v in sorted(by_routing.items())
        },
        "results": run_results,
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    verdict = "bad"
    if agg["top1_acc"] >= 0.75 and agg["mean_gt_rank"] <= 1.5:
        verdict = "good"
    elif agg["top1_acc"] >= 0.5 or agg["mean_gt_rank"] <= 2.0:
        verdict = "partial"

    print(f"[mini-exact-s2-val] runs={len(run_results)}")
    print(f"[mini-exact-s2-val] top1_acc={agg['top1_acc']:.3f}")
    print(f"[mini-exact-s2-val] mean_gt_rank={agg['mean_gt_rank']:.3f}")
    print(f"[mini-exact-s2-val] entropy={agg['entropy']:.4f}")
    print(f"[mini-exact-s2-val] verdict={verdict}")
    print(f"[mini-exact-s2-val] summary={summary_path}")

    unload_lora(pipeline)


if __name__ == "__main__":
    main()
