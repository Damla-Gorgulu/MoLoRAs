#!/usr/bin/env python3
"""
Inference: Synthesise a style LoRA for a query image and generate result images.

Loads a trained RoutingMLP checkpoint, encodes the query style image with CLIP,
routes through the expert pool, synthesises a composite LoRA, injects it into
frozen SDXL, and generates images.

Example (Stage 1 checkpoint, all experts in pool):
    python inference.py \
        --checkpoint  /scratch/eyavuz21/lora_attention/stage1/latest.pt \
        --style_image /path/to/query_style.jpg \
        --prompt      "A cat in Impressionism style" \
        --output_dir  /scratch/eyavuz21/lora_attention/inference_out

Example (specify a subset pool):
    python inference.py \
        --checkpoint  /scratch/eyavuz21/lora_attention/stage2/latest.pt \
        --style_image /path/to/query_style.jpg \
        --prompt      "A landscape painting in Romanticism style" \
        --pool_size   15 \
        --output_dir  /scratch/eyavuz21/lora_attention/inference_out
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# parents[1] = MoLoRAs/ (the directory that contains lora_attention/)
sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora import MoELoRA
from lora_attention.utils.lora_inject import inject_lora, unload_lora


# ──────────────────────────────────────────────────────────────
# Attention Heatmap Visualization
# ──────────────────────────────────────────────────────────────
def _save_attention_heatmap(
    A: torch.Tensor,                      # (N, rank)
    pool_indices: list,
    style_names: list,
    save_path: Path,
    title: str = "",
    top_n_experts: int = 15,
):
    """Save a heatmap showing attention weights over experts × ranks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A_np = A.detach().numpy()                              # (N, rank)
    N, R = A_np.shape

    # Show only the top-N experts by avg attention (for readability)
    avg_attn = A_np.mean(axis=1)                           # (N,)
    top_expert_idx = np.argsort(avg_attn)[::-1][:top_n_experts]
    A_top = A_np[top_expert_idx]                           # (top_n, rank)

    expert_labels = [
        f"{style_names[pool_indices[i]]} ({avg_attn[i]:.3f})"
        for i in top_expert_idx
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, top_n_experts * 0.4)),
                             gridspec_kw={"width_ratios": [4, 1]})

    # Left: full heatmap
    im = axes[0].imshow(A_top, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axes[0].set_yticks(range(len(expert_labels)))
    axes[0].set_yticklabels(expert_labels, fontsize=8)
    axes[0].set_xlabel("Rank position")
    axes[0].set_title(f"Attention heatmap (top {top_n_experts} experts) {title}")
    fig.colorbar(im, ax=axes[0], shrink=0.6)

    # Right: bar chart of avg attention
    axes[1].barh(range(len(expert_labels)), avg_attn[top_expert_idx],
                 color="coral", edgecolor="k", linewidth=0.5)
    axes[1].set_yticks(range(len(expert_labels)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Avg attention")
    axes[1].set_title("Avg over ranks")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA Inference")

    # Required
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a Stage 1 or Stage 2 checkpoint (latest.pt).")
    p.add_argument("--style_image", type=str, required=True,
                   help="Path to the query style image (will be encoded with CLIP).")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt for generation (e.g. 'A cat in Impressionism style').")
    p.add_argument("--output_dir", type=str, required=True)

    # Pool
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--pool_size", type=int, default=None,
                   help="Number of experts to include in the pool. "
                        "Default: all experts in zoo.")
    p.add_argument("--pool_indices", type=int, nargs="+", default=None,
                   help="Explicit expert indices to use as pool. "
                        "Overrides --pool_size.")
    p.add_argument("--exclude_experts", type=str, nargs="+", default=None,
                   help="Substrings to match against style names to exclude from pool. "
                        "E.g. --exclude_experts style_0000_Baroque style_0003_Cubism. "
                        "Applied after --pool_indices/--pool_size.")
    p.add_argument("--gt_expert", type=str, default=None,
                   help="Name of the ground-truth expert (for logging whether it is "
                        "in the pool). E.g. style_0000_Baroque.")
    p.add_argument("--query_label", type=str, default=None,
                   help="Short label embedded in output filenames and attention .pt. "
                        "Useful to distinguish held-out vs in-pool runs.")

    # Model
    p.add_argument("--clip_model_id", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--style_alpha", type=float, default=1.0)

    # Attention sharpening
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature. < 1.0 sharpens, > 1.0 smooths. "
                        "Try 0.1 or 0.01 for strong style transfer.")
    p.add_argument("--top_k", type=int, default=None,
                   help="Keep only top-k experts per rank, zero out rest. "
                        "Reduces dilution. Try 1, 3, or 5.")

    # Reference: use real B-LoRA instead of synthesised
    p.add_argument("--reference_blora", type=str, default=None,
                   help="Path to a real B-LoRA .safetensors for comparison. "
                        "Bypasses MoE routing and injects this directly.")

    # SDXL
    p.add_argument("--pretrained_model", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", type=str,
                   default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)

    # Content LoRA (optional, for style+content transfer)
    p.add_argument("--content_lora", type=str, default=None,
                   help="Optional path to a content B-LoRA .safetensors file.")
    p.add_argument("--content_alpha", type=float, default=1.0)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    # ── LoRA pool ─────────────────────────────────────────────
    pool = LoRAPool(zoo_dir=args.zoo_dir, cache_dir=args.cache_dir)

    # ── MoELoRA ───────────────────────────────────────────────
    model = MoELoRA(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.routing_mlp.load_state_dict(ckpt["routing_mlp_state_dict"])
    step = ckpt.get("step", "?")
    print(f"[Inference] Loaded checkpoint from step {step}")
    model.routing_mlp.eval()

    # ── Determine pool ────────────────────────────────────────
    if args.pool_indices is not None:
        pool_indices = args.pool_indices
    elif args.pool_size is not None:
        n = min(args.pool_size, pool.num_experts)
        pool_indices = torch.randperm(pool.num_experts)[:n].tolist()
    else:
        pool_indices = list(range(pool.num_experts))

    # Apply exclusions (substring matching on style names)
    if args.exclude_experts:
        excluded_set = {e.strip() for e in args.exclude_experts}
        before = len(pool_indices)
        pool_indices = [
            idx for idx in pool_indices
            if not any(excl in pool.style_names[idx] for excl in excluded_set)
        ]
        removed = before - len(pool_indices)
        print(f"[Inference] Excluded {removed} experts matching: {excluded_set}")

    # Log ground-truth expert status
    gt_in_pool = None
    if args.gt_expert:
        gt_in_pool = any(args.gt_expert in pool.style_names[i] for i in pool_indices)
        status = "IN pool" if gt_in_pool else "HELD OUT"
        print(f"[Inference] GT expert '{args.gt_expert}': {status}")

    print(f"[Inference] Pool size: {len(pool_indices)} experts")

    # ── Load query style image ────────────────────────────────
    style_image = Image.open(args.style_image).convert("RGB")

    # ── Route: CLIP encode + MoE forward ──────────────────────
    print(f"[Inference] Temperature: {args.temperature}, Top-k: {args.top_k}")
    with torch.no_grad():
        q = model.encode_image(style_image, device)             # (1, clip_dim)
        A, synth_lora = model.forward(
            q, pool_indices,
            temperature=args.temperature,
            top_k=args.top_k,
        )                                                       # (N, rank), {...}

    # Print top-5 most attended experts per rank (average)
    avg_attention = A.mean(dim=1)                               # (N,)
    top_show = min(5, len(pool_indices))
    top_vals, top_idxs = avg_attention.topk(top_show)
    print("[Inference] Top experts by avg attention:")
    for rank_i, expert_i in enumerate(top_idxs.tolist()):
        expert_name = pool.style_names[pool_indices[expert_i]]
        print(f"  #{rank_i+1}: {expert_name}  (avg_A={top_vals[rank_i].item():.4f})")

    # Print attention entropy for diagnostics
    import math as _math
    entropy = -(A * (A + 1e-10).log()).sum(dim=0).mean().item()
    max_entropy = _math.log(len(pool_indices))
    print(f"[Inference] Attention entropy: {entropy:.4f} / {max_entropy:.4f} (max)")

    # Log GT rank if GT expert was in pool
    if args.gt_expert and gt_in_pool:
        for local_i, global_i in enumerate(pool_indices):
            if args.gt_expert in pool.style_names[global_i]:
                gt_local = local_i
                gt_attn = avg_attention[gt_local].item()
                gt_rank = (avg_attention > gt_attn).sum().item() + 1
                print(f"[Inference] GT expert rank in pool: #{gt_rank}  (avg_A={gt_attn:.4f})")
                break

    # ── Load SDXL ─────────────────────────────────────────────
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    print(f"[Inference] Loading SDXL: {args.pretrained_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)

    # ── Inject LoRA ───────────────────────────────────────
    if args.reference_blora is not None:
        # Reference mode: load real B-LoRA directly for comparison
        from safetensors.torch import load_file as _load_file
        real_sd = _load_file(args.reference_blora)
        real_sd = {k: v * args.style_alpha for k, v in real_sd.items()
                   if "up_blocks.0.attentions.1" in k}  # style block only
        print(f"[Inference] Reference B-LoRA: {args.reference_blora}")
        print(f"[Inference] Injecting {len(real_sd)} real LoRA tensors")
        pipeline.load_lora_into_unet(real_sd, None, pipeline.unet)
    else:
        # Normal mode: inject synthesised LoRA
        synth_lora_cpu = {k: v.detach().cpu() for k, v in synth_lora.items()}
        inject_lora(
            pipeline=pipeline,
            style_state_dict=synth_lora_cpu,
            style_alpha=args.style_alpha,
            content_lora_path=args.content_lora,
            content_alpha=args.content_alpha,
        )

    # ── Generate ──────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipeline(
        args.prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,        cross_attention_kwargs={"scale": 1.0},    ).images

    # ── Save ──────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    label_prefix = f"{args.query_label}_" if args.query_label else ""
    prompt_slug = args.prompt[:40].replace(" ", "_").replace("/", "_")
    for i, img in enumerate(images):
        out_path = Path(args.output_dir) / f"{label_prefix}{prompt_slug}_{i}.jpg"
        img.save(out_path)
        print(f"  [saved] {out_path}")

    # Save the attention map as a reference
    attn_data = {
        "attention": A.cpu(),
        "pool_indices": pool_indices,
        "pool_names": [pool.style_names[j] for j in pool_indices],
        "prompt": args.prompt,
        "style_image": args.style_image,
        "step": step,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "style_alpha": args.style_alpha,
        "gt_expert": args.gt_expert,
        "gt_in_pool": gt_in_pool,
        "exclude_experts": args.exclude_experts,
        "query_label": args.query_label,
    }
    attn_filename = f"{label_prefix}{prompt_slug}_attention.pt"
    torch.save(attn_data, Path(args.output_dir) / attn_filename)
    print(f"  [saved] attention map")

    # ── Attention heatmap ─────────────────────────────────────────────────────────────────────
    try:
        gt_label = f" | GT={args.gt_expert}" if args.gt_expert else ""
        pool_status = f" ({'held-out' if not gt_in_pool else 'in-pool'})" if args.gt_expert else ""
        _save_attention_heatmap(
            A.cpu(), pool_indices, pool.style_names,
            Path(args.output_dir) / f"{label_prefix}{prompt_slug}_heatmap.png",
            title=f"τ={args.temperature}, top_k={args.top_k}{gt_label}{pool_status}",
        )
        print(f"  [saved] attention heatmap")
    except Exception as e:
        print(f"  [warn] heatmap save failed: {e}")

    # ── Copy reference images for quick visual comparison ──────
    import shutil
    try:
        # 1. Query / input style image
        query_ext = Path(args.style_image).suffix or ".jpg"
        query_dst = Path(args.output_dir) / f"{label_prefix}__query{query_ext}"
        shutil.copy2(args.style_image, query_dst)
        print(f"  [saved] query image  → {query_dst.name}")

        # 2. Top-5 pool expert reference style images
        #    Layout: <zoo_dir>/../style_images/<style_name>/<style_name>.jpg
        style_images_root = pool.zoo_dir.parent / "style_images"
        N_pool = len(pool_indices)
        top_k_show = min(5, N_pool)
        top_vals_ref, top_idxs_ref = avg_attention.topk(top_k_show)
        for rank_i, expert_i in enumerate(top_idxs_ref.tolist()):
            style_name = pool.style_names[pool_indices[expert_i]]
            src = style_images_root / style_name / f"{style_name}.jpg"
            if src.exists():
                dst = Path(args.output_dir) / (
                    f"{label_prefix}__top{rank_i+1}_{style_name}.jpg"
                )
                shutil.copy2(src, dst)
                attn_val = top_vals_ref[rank_i].item()
                print(f"  [saved] top-{rank_i+1} expert    → {dst.name}  (avg_A={attn_val:.4f})")
            else:
                print(f"  [warn]  style image not found: {src}")
    except Exception as _e:
        print(f"  [warn] could not copy reference images: {_e}")

    unload_lora(pipeline)
    print(f"\n[Inference] Done. {len(images)} images saved to {args.output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
