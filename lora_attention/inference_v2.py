#!/usr/bin/env python3
"""
v2.0 Inference: Per-tensor routing with LoRARankEncoder.

Loads a trained v2.0 encoder checkpoint, encodes a query style image with CLIP,
runs per-tensor routing through the expert pool, synthesises a composite LoRA,
injects it into frozen SDXL, and generates images.

Key differences from v1.0:
  - Uses LoRARankEncoder (not RoutingMLP)
  - Per-tensor attention A ∈ ℝ^{N×T×r} (80 independent routing decisions)
  - Saves full (N, T, r) attention tensor for analysis
  - Per-layer attention visualisation

Example:
    python inference_v2.py \\
        --checkpoint /scratch/eyavuz21/lora_attention/stage1_v2/latest.pt \\
        --style_image /path/to/query_style.jpg \\
        --prompt "A cat in Impressionism style" \\
        --output_dir /scratch/eyavuz21/lora_attention/inference_v2_out \\
        --temperature 0.1
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.utils.lora_inject import inject_lora, unload_lora


# ──────────────────────────────────────────────────────────────
# Attention Heatmap — Per-tensor version
# ──────────────────────────────────────────────────────────────
def _save_attention_heatmap_v2(
    A: torch.Tensor,               # (N, T, r)
    pool_indices: list,
    style_names: list,
    down_keys: list,               # T key names
    save_path: Path,
    title: str = "",
    top_n_experts: int = 10,
):
    """Save per-tensor attention heatmap: rows=experts, cols=tensor groups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Average over ranks → (N, T)
    A_np = A.detach().numpy()
    A_avg = A_np.mean(axis=2)  # (N, T)

    # Top experts by global average attention
    global_avg = A_avg.mean(axis=1)  # (N,)
    top_idx = np.argsort(global_avg)[::-1][:top_n_experts]
    A_show = A_avg[top_idx]  # (top_n, T)

    expert_labels = [
        f"{style_names[pool_indices[i]]} ({global_avg[i]:.3f})"
        for i in top_idx
    ]

    # Shorten tensor key names for readability
    def _short_key(k):
        # e.g. "unet.up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_v.lora.down.weight"
        # → "b3.attn2.to_v"
        parts = k.split(".")
        block_idx = [p for p in parts if p.startswith("transformer_blocks")]
        attn = [p for p in parts if p.startswith("attn")]
        proj = [p for p in parts if p.startswith("to_")]
        b = block_idx[0].split(".")[-1] if block_idx else "?"
        a = attn[0] if attn else "?"
        p_name = proj[0] if proj else "?"
        return f"b{b}.{a}.{p_name}"

    tensor_labels = [_short_key(k) for k in down_keys]

    fig, ax = plt.subplots(figsize=(max(20, len(down_keys) * 0.3),
                                     max(6, top_n_experts * 0.5)))
    im = ax.imshow(A_show, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(expert_labels)))
    ax.set_yticklabels(expert_labels, fontsize=7)
    ax.set_xticks(range(len(tensor_labels)))
    ax.set_xticklabels(tensor_labels, fontsize=5, rotation=90)
    ax.set_xlabel("Tensor group (adapter pair)")
    ax.set_ylabel("Expert (avg attention)")
    ax.set_title(f"Per-tensor attention (avg over ranks) {title}")
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA v2.0 Inference")

    # Required
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a v2.0 checkpoint (latest.pt).")
    p.add_argument("--style_image", type=str, required=True,
                   help="Path to the query style image.")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt (e.g. 'A cat in Impressionism style').")
    p.add_argument("--output_dir", type=str, required=True)

    # Pool
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--pool_size", type=int, default=None)
    p.add_argument("--pool_indices", type=int, nargs="+", default=None)
    p.add_argument("--exclude_experts", type=str, nargs="+", default=None)
    p.add_argument("--gt_expert", type=str, default=None)
    p.add_argument("--query_label", type=str, default=None)

    # Model
    p.add_argument("--clip_model_id", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--no_normalize_keys", action="store_true")
    p.add_argument("--style_alpha", type=float, default=1.0)
    p.add_argument("--norm_match", action="store_true",
                   help="Scale synth LoRA so its total Frobenius norm matches the "
                        "mean real B-LoRA norm (~50). Makes style effect visible "
                        "even when routing is near-uniform. The style direction is "
                        "still determined by routing; only magnitude is forced to "
                        "match a single real B-LoRA.")
    p.add_argument("--product_synth", action="store_true",
                   help="Use product-space synthesis (RECOMMENDED). "
                        "Computes ΔW = Σ A_i*(W_up_i @ W_down_i) then decomposes "
                        "back to LoRA via SVD. Fixes the O(N²) cross-term "
                        "cancellation bug in the default parameter-averaging mode. "
                        "Essential when routing is near-uniform over many experts.")

    # Attention
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Softmax temperature at inference.")
    p.add_argument("--top_k", type=int, default=None)

    # Reference comparison
    p.add_argument("--reference_blora", type=str, default=None)

    # SDXL
    p.add_argument("--pretrained_model", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", type=str,
                   default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)

    # Content LoRA
    p.add_argument("--content_lora", type=str, default=None)
    p.add_argument("--content_alpha", type=float, default=1.0)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference-v2] Device: {device}")

    # ── Pool ──────────────────────────────────────────────────
    pool = LoRAPool(zoo_dir=args.zoo_dir, cache_dir=args.cache_dir)

    # ── MoELoRAv2 ─────────────────────────────────────────────
    model = MoELoRAv2(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
        normalize_keys=not args.no_normalize_keys,
    ).to(device)

    # Load v2.0 checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    step = ckpt.get("step", "?")
    print(f"[Inference-v2] Loaded checkpoint from step {step}")
    model.encoder.eval()

    T = model.num_tensor_groups
    print(f"[Inference-v2] Tensor groups: {T}")

    # ── Determine pool ────────────────────────────────────────
    if args.pool_indices is not None:
        pool_indices = args.pool_indices
    elif args.pool_size is not None:
        n = min(args.pool_size, pool.num_experts)
        pool_indices = torch.randperm(pool.num_experts)[:n].tolist()
    else:
        pool_indices = list(range(pool.num_experts))

    if args.exclude_experts:
        excluded_set = {e.strip() for e in args.exclude_experts}
        before = len(pool_indices)
        pool_indices = [
            idx for idx in pool_indices
            if not any(excl in pool.style_names[idx] for excl in excluded_set)
        ]
        print(f"[Inference-v2] Excluded {before - len(pool_indices)} experts")

    gt_in_pool = None
    if args.gt_expert:
        gt_in_pool = any(args.gt_expert in pool.style_names[i] for i in pool_indices)
        status = "IN pool" if gt_in_pool else "HELD OUT"
        print(f"[Inference-v2] GT expert '{args.gt_expert}': {status}")

    N = len(pool_indices)
    print(f"[Inference-v2] Pool size: {N} experts")

    # ── Load query image ──────────────────────────────────────
    style_image = Image.open(args.style_image).convert("RGB")

    # ── Route ─────────────────────────────────────────────────
    print(f"[Inference-v2] τ={args.temperature}, top_k={args.top_k}")
    if args.product_synth:
        print("[Inference-v2] Using PRODUCT-SPACE synthesis (correct, SVD-based)")
    else:
        print("[Inference-v2] Using PARAMETER-AVERAGING synthesis (legacy; cross-term bug)")

    with torch.no_grad():
        q = model.encode_image(style_image, device)
        A, synth_lora = model.forward(
            q, pool_indices,
            temperature=args.temperature,
            top_k=args.top_k,
            product_space=args.product_synth,
        )  # A: (N, T, r)

    # ── Diagnostics ───────────────────────────────────────────
    # Global average attention per expert (avg over T and r)
    avg_attn_global = A.mean(dim=(1, 2))  # (N,)
    top_show = min(5, N)
    top_vals, top_idxs = avg_attn_global.topk(top_show)
    print("[Inference-v2] Top experts by global avg attention:")
    for rank_i, expert_i in enumerate(top_idxs.tolist()):
        name = pool.style_names[pool_indices[expert_i]]
        print(f"  #{rank_i+1}: {name}  (avg_A={top_vals[rank_i].item():.4f})")

    # Per-tensor entropy
    entropy = model.attention_entropy(A).item()
    max_entropy = math.log(N)
    print(f"[Inference-v2] Attention entropy: {entropy:.4f} / {max_entropy:.4f} (max)")

    # Per-tensor routing diversity: how many experts dominate each tensor
    A_per_tensor = A.mean(dim=2)  # (N, T) — avg over ranks
    dominant = (A_per_tensor > 1.0 / N).sum(dim=0).float()  # per-tensor: how many above uniform
    print(
        f"[Inference-v2] Per-tensor active experts: "
        f"min={dominant.min().item():.0f}, "
        f"mean={dominant.mean().item():.1f}, "
        f"max={dominant.max().item():.0f}"
    )

    # Check if different tensors route to different top experts
    top1_per_tensor = A_per_tensor.argmax(dim=0)  # (T,) — top-1 expert per tensor
    n_unique = top1_per_tensor.unique().numel()
    print(
        f"[Inference-v2] Unique top-1 experts across {T} tensors: {n_unique}"
    )

    # GT rank if GT in pool
    if args.gt_expert and gt_in_pool:
        for local_i, global_i in enumerate(pool_indices):
            if args.gt_expert in pool.style_names[global_i]:
                gt_attn = avg_attn_global[local_i].item()
                gt_rank = (avg_attn_global > gt_attn).sum().item() + 1
                print(f"[Inference-v2] GT rank: #{gt_rank} (avg_A={gt_attn:.4f})")
                break

    # ── Load SDXL ─────────────────────────────────────────────
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

    print(f"[Inference-v2] Loading SDXL: {args.pretrained_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model, vae=vae, torch_dtype=torch.float16,
    ).to(device)

    # ── Inject LoRA ───────────────────────────────────────────
    if args.reference_blora is not None:
        from safetensors.torch import load_file as _load_file
        real_sd = _load_file(args.reference_blora)
        real_sd = {
            k: v * args.style_alpha
            for k, v in real_sd.items()
            if "up_blocks.0.attentions.1" in k
        }
        print(f"[Inference-v2] Reference B-LoRA: {len(real_sd)} tensors")
        pipeline.load_lora_into_unet(real_sd, None, pipeline.unet)
    else:
        synth_cpu = {k: v.detach().cpu() for k, v in synth_lora.items()}

        # Optional norm-matching: rescale synth so total Frobenius norm equals a
        # single real B-LoRA (~50). This makes the visual effect comparable to
        # injecting a real B-LoRA. Style DIRECTION is still determined by routing;
        # only magnitude is normalised. Useful to confirm injection works and see
        # what style direction the routing is pushing toward.
        alpha = args.style_alpha
        if args.norm_match:
            synth_total_norm = sum(
                v.norm().item() ** 2 for v in synth_cpu.values()
            ) ** 0.5
            TARGET_NORM = 50.0  # empirical mean Frobenius norm of a real B-LoRA style block
            if synth_total_norm > 1e-6:
                nm_scale = TARGET_NORM / synth_total_norm
                alpha = alpha * nm_scale
                print(f"[Inference-v2] norm_match: synth_norm={synth_total_norm:.3f}, "
                      f"scale={nm_scale:.3f}x, effective_alpha={alpha:.3f}")
            else:
                print("[Inference-v2] norm_match: synth_norm≈0, skipping scale")

        if alpha > 0.0:
            inject_lora(
                pipeline=pipeline,
                style_state_dict=synth_cpu,
                style_alpha=alpha,
                content_lora_path=args.content_lora,
                content_alpha=args.content_alpha,
            )
        else:
            print("[Inference-v2] style_alpha=0 — skipping LoRA injection (vanilla SDXL)")

    # ── Generate ──────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipeline(
        args.prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
    ).images

    # ── Save outputs ──────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Sanitize query_label for use as filename prefix (strip path separators)
    label_prefix = f"{args.query_label.replace('/', '_').replace(' ', '_')}_" if args.query_label else ""
    prompt_slug = args.prompt[:40].replace(" ", "_").replace("/", "_")

    for i, img in enumerate(images):
        out_path = Path(args.output_dir) / f"{label_prefix}{prompt_slug}_{i}.jpg"
        img.save(out_path)
        print(f"  [saved] {out_path}")

    # Save full per-tensor attention
    attn_data = {
        "attention": A.cpu(),           # (N, T, r) — full per-tensor
        "pool_indices": pool_indices,
        "pool_names": [pool.style_names[j] for j in pool_indices],
        "down_keys": model.down_key_order,
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
        "version": "v2.0",
    }
    attn_file = f"{label_prefix}{prompt_slug}_attention.pt"
    torch.save(attn_data, Path(args.output_dir) / attn_file)
    print(f"  [saved] per-tensor attention ({A.shape})")

    # ── Heatmap ───────────────────────────────────────────────
    try:
        gt_label = f" | GT={args.gt_expert}" if args.gt_expert else ""
        pool_status = f" ({'held-out' if not gt_in_pool else 'in-pool'})" if args.gt_expert else ""
        _save_attention_heatmap_v2(
            A.cpu(), pool_indices, pool.style_names,
            model.down_key_order,
            Path(args.output_dir) / f"{label_prefix}{prompt_slug}_heatmap.png",
            title=f"τ={args.temperature}, top_k={args.top_k}{gt_label}{pool_status}",
        )
        print(f"  [saved] per-tensor heatmap")
    except Exception as e:
        print(f"  [warn] heatmap failed: {e}")

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
        top_k_show = min(5, N)
        top_vals_ref, top_idxs_ref = avg_attn_global.topk(top_k_show)
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
    print(f"\n[Inference-v2] Done. {len(images)} images → {args.output_dir}")


if __name__ == "__main__":
    main()
