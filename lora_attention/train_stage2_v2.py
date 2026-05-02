#!/usr/bin/env python3
"""
v2.0 Stage 2 Training: LDM Loss + Entropy Regularisation

Fine-tunes the LoRARankEncoder via latent diffusion loss,
with the ground-truth LoRA excluded from the pool.

Key differences from v1.0:
  - LoRARankEncoder replaces RoutingMLP
  - Per-tensor attention A ∈ ℝ^{N×T×r} (80 independent routing decisions)
  - Entropy regularisation: L = L_LDM - λ·H̄(A)
  - λ annealing: linear from λ_start → λ_end over training
  - WikiArt dataset (~80k images)
  - τ = 1.0 at training

Gradients flow:
    L_LDM → UNet (frozen hooks) → synth_lora (W_down, W_up per tensor)
           → per-tensor attention A → LoRARankEncoder (trainable)

Requires a GPU with ≥ 16 GB VRAM for SDXL in fp16.

Example:
    python train_stage2_v2.py \\
        --stage1_ckpt /scratch/eyavuz21/lora_attention/stage1_v2/latest.pt \\
        --output_dir /scratch/eyavuz21/lora_attention/stage2_v2 \\
        --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \\
        --cache_dir /scratch/eyavuz21/lora_attention \\
        --wikiart_dir /home/eyavuz21/datasets/wikiart \\
        --label_map_path /scratch/eyavuz21/lora_attention/wikiart_label_map.json \\
        --max_steps 8000 \\
        --lr 5e-5
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# ── Path setup ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.data.dataset import (
    ExactExemplarStage2Dataset,
    WikiArtStage2Dataset,
    exact_stage2_collate_fn,
    wikiart_stage2_collate_fn,
)
from lora_attention.utils.lora_inject import (
    apply_lora_hooks_with_grad,
    remove_hooks,
)


# Image normalisation for VAE input (SDXL expects [-1, 1], 1024×1024)
_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # [0,1] → [-1,1]
])


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA v2.0 Stage 2 Training")

    # Paths
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--output_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention/stage2_v2")
    p.add_argument("--wikiart_dir", type=str,
                   default="/home/eyavuz21/datasets/wikiart")
    p.add_argument("--label_map_path", type=str,
                   default="/scratch/eyavuz21/lora_attention/wikiart_label_map.json")
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--exact_manifest_path", type=str, default=None,
                   help="Optional exact-exemplar manifest for tiny Stage 2 follow-up runs.")
    p.add_argument("--exact_views_per_style", type=int, default=32,
                   help="Number of augmented views per exact exemplar style for Stage 2.")
    p.add_argument("--exact_prompt_mode", type=str, default="neutral",
                   choices=["neutral", "minimal", "style"],
                   help="Prompt type for exact-exemplar Stage 2 runs.")
    p.add_argument("--stage1_ckpt", type=str, default=None,
                   help="v2.0 Stage 1 checkpoint to initialise encoder.")
    p.add_argument("--resume_from", type=str, default=None)

    # Base model
    p.add_argument("--pretrained_model", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", type=str,
                   default="madebyollin/sdxl-vae-fp16-fix")

    # Routing model
    p.add_argument("--clip_model_id", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--no_normalize_keys", action="store_true")
    p.add_argument("--force_rebuild_cache", action="store_true")
    p.add_argument("--product_synth", action="store_true",
                   help="Use product-space synthesis during Stage 2 training. "
                        "This is the recommended default.")
    p.add_argument("--legacy_synth", action="store_false", dest="product_synth",
                   help="Use legacy parameter averaging during Stage 2 training.")
    p.set_defaults(product_synth=True)

    # Dataset
    p.add_argument("--min_pool_size", type=int, default=5)
    p.add_argument("--max_pool_size", type=int, default=20)
    p.add_argument("--max_images_per_style", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--max_steps", type=int, default=8_000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])

    # Entropy regularisation
    p.add_argument("--lb_start_step", type=int, default=2000,
                   help="Step when load-balancing loss starts (0 to disable).")
    p.add_argument("--lb_end_step", type=int, default=6000,
                   help="Step when load-balancing loss ends.")
    p.add_argument("--lambda_start", type=float, default=0.1,
                   help="Starting load-balancing loss weight.")
    p.add_argument("--lambda_end", type=float, default=0.01,
                   help="Final load-balancing loss weight.")
    # Temperature: start sharp, cool for diversity in early training, then sharpen again
    p.add_argument("--tau_start", type=float, default=1.0,
                   help="Initial softmax temperature.")
    p.add_argument("--tau_mid", type=float, default=2.0,
                   help="Mid-training temperature (peak diversity).")
    p.add_argument("--tau_end", type=float, default=0.3,
                   help="Final softmax temperature (sharper routing).")
    p.add_argument("--tau_mid_step", type=int, default=3000,
                   help="Step when temperature reaches mid point.")
    p.add_argument("--ema_beta", type=float, default=0.99,
                   help="EMA decay for expert usage tracking in load-balancing loss.")

    # Logging / saving
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=500)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def get_lambda_entropy(
    step: int, max_steps: int, lam_start: float, lam_end: float
) -> float:
    """Linear annealing of load-balancing loss weight."""
    progress = min(step / max(1, max_steps), 1.0)
    return lam_start + (lam_end - lam_start) * progress


def get_temperature(step: int, tau_start: float, tau_end: float,
                    tau_warmup_steps: int) -> float:
    """
    Anneal softmax temperature from tau_start → tau_end over tau_warmup_steps.
    High temperature (τ >> 1) forces near-uniform routing, preventing
    snap-collapse in early training.  After warmup, τ = tau_end (typically 1.0).
    """
    if tau_warmup_steps <= 0 or tau_start == tau_end:
        return tau_end
    progress = min(step / tau_warmup_steps, 1.0)
    return tau_start + (tau_end - tau_start) * progress

def get_temperature_v3(step: int, tau_start: float, tau_mid: float,
                      tau_end: float, tau_mid_step: int, max_steps: int) -> float:
    if step < tau_mid_step:
        progress = step / max(1, tau_mid_step)
        return tau_start + (tau_mid - tau_start) * progress
    else:
        progress = (step - tau_mid_step) / max(1, max_steps - tau_mid_step)
        return tau_mid + (tau_end - tau_mid) * progress


def get_lambda_lb(step: int, lb_start: int, lb_end: int,
                  lam_start: float, lam_end: float) -> float:
    if step < lb_start:
        return 0.0
    if step >= lb_end:
        return 0.0
    progress = (step - lb_start) / max(1, lb_end - lb_start)
    return lam_start + (lam_end - lam_start) * progress



def load_sdxl_pipeline(args, device):
    """Load frozen SDXL pipeline with fp16-fix VAE."""
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

    print(f"[Stage2-v2] Loading SDXL: {args.pretrained_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)

    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.eval()
    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()

    return pipeline


def encode_prompt(pipeline, prompt: str, device) -> tuple:
    """Encode a text prompt with the SDXL dual text encoders."""
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]

    prompt_embeds_list = []
    for tok, enc in zip(tokenizers, text_encoders):
        input_ids = tok(
            [prompt],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            out = enc(input_ids, output_hidden_states=True)
            prompt_embeds_list.append(out.hidden_states[-2])

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)  # (1, seq, 2048)

    with torch.no_grad():
        input_ids_2 = pipeline.tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        pooled_embeds = pipeline.text_encoder_2(
            input_ids_2, output_hidden_states=True
        ).text_embeds  # (1, 1280)

    return prompt_embeds, pooled_embeds


def save_checkpoint(output_dir, step, model, optimizer, loss):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "version": "v2.0",
    }
    torch.save(payload, ckpt_dir / "checkpoint.pt")
    torch.save(payload, Path(output_dir) / "latest.pt")
    print(f"  [save] {ckpt_dir}")


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
from accelerate import Accelerator

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    print(f"[Stage2-v2] Device: {device}")

    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }[args.mixed_precision]

    # ── Pool ──────────────────────────────────────────────────
    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    # ── MoELoRAv2 (encoder trainable) ────────────────────────
    model = MoELoRAv2(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
        normalize_keys=not args.no_normalize_keys,
    ).to(device)

    # Load Stage 1 v2.0 weights
    if args.stage1_ckpt is not None:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"[Stage2-v2] Loaded Stage 1 encoder from {args.stage1_ckpt}")

    model._ensure_clip(device)

    n_params = sum(p.numel() for p in model.encoder.parameters())
    T = model.num_tensor_groups

    # ── SDXL pipeline ─────────────────────────────────────────
    pipeline = load_sdxl_pipeline(args, device)
    noise_scheduler = pipeline.scheduler

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Dataset ───────────────────────────────────────────────
    if args.exact_manifest_path is not None:
        dataset = ExactExemplarStage2Dataset(
            pool=pool,
            manifest_path=args.exact_manifest_path,
            min_pool_size=args.min_pool_size,
            max_pool_size=args.max_pool_size,
            views_per_style=args.exact_views_per_style,
            seed=args.seed,
            prompt_mode=args.exact_prompt_mode,
        )
        collate_fn = exact_stage2_collate_fn
    else:
        dataset = WikiArtStage2Dataset(
            pool=pool,
            wikiart_dir=args.wikiart_dir,
            label_map_path=args.label_map_path,
            min_pool_size=args.min_pool_size,
            max_pool_size=args.max_pool_size,
            max_images_per_style=args.max_images_per_style,
            rank=args.rank,
            seed=args.seed,
            prompts_file=args.prompts_file,
        )
        collate_fn = wikiart_stage2_collate_fn
    loader = DataLoader(
        dataset,
        batch_size=1,  # One sample at a time (variable N + GPU memory)
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"[Stage2-v2] Resumed from step {ckpt['step']}")

    # ── Output / logging ──────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train_log.txt"
    log_fh = open(log_path, "a")

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*60}")
    log(f"Stage 2 v2.0 Training  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pool: {pool.num_experts} experts (GT excluded)")
    log(f"Encoder params: {n_params:,}")
    log(f"Tensor groups: {T}")
    log(f"max_steps={args.max_steps}, lr={args.lr}")
    log(f"mixed_precision={args.mixed_precision}")
    log(f"λ_entropy: {args.lambda_start} → {args.lambda_end} (linear)")
    log(f"Dataset size: {len(dataset)}")
    log(f"{'='*60}")

    # ── Training ──────────────────────────────────────────────
    model.encoder.train()
    step = start_step
    running_loss_ldm = 0.0
    running_loss_ent = 0.0
    running_loss_total = 0.0
    running_skipped = 0  # steps where Inf/NaN gradients forced skip

    # EMA of mean router probability per expert  (N,).
    # Used by the Switch-style load-balancing loss:
    #   L_lb = λ · N · Σ_i  f_ema_i · P_i
    # where f_ema_i is the EMA fraction (no gradient) and P_i = A.mean()
    # has gradient.  This is ALWAYS non-zero when routing is imbalanced,
    # including when fully collapsed — unlike entropy which has zero gradient
    # at a one-hot distribution.
    N_experts = model.pool.num_experts   # number of LoRA experts
    ema_expert_usage = torch.full((N_experts,), 1.0 / N_experts, device=device)
    data_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Update LR
        lr = get_lr(step, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Entropy regularisation weight
        lam = get_lambda_lb(
            step, args.lb_start_step, args.lb_end_step, args.lambda_start, args.lambda_end
        )

        optimizer.zero_grad()

        image = batch["images"][0]          # PIL.Image
        prompt = batch["prompts"][0]
        pool_indices = batch["pool_indices"][0]

        # ── CLIP encode (expects PIL) ──────────────────────
        q = model.encode_image(image, device)  # (1, clip_dim)

        # ── Temperature-annealed forward ──────────────────
        tau = get_temperature_v3(step, args.tau_start, args.tau_mid,
                                 args.tau_end, args.tau_mid_step, args.max_steps)
        # Use product-space synthesis so training and inference share the same
        # composition rule. `--legacy_synth` is kept only for ablations.
        A, synth_lora = model.forward(q, pool_indices, temperature=tau,
                                      product_space=args.product_synth)
        # A: (N, T, r) with grad_fn

        # ── Switch-style load-balancing loss ──────────────
        # pool_indices is the subset of experts in this batch (variable N_batch).
        # ema_expert_usage is (N_experts=109,) — full-pool EMA indexed by expert id.
        pool_idx_t = torch.tensor(pool_indices, dtype=torch.long, device=device)
        N_batch = len(pool_indices)

        # Sanitize A: fp16 softmax can produce NaN/Inf on edge cases.
        # (The synthesis path already does nan_to_num in _synthesise_product_space,
        #  but LB-loss uses raw A — must sanitize here too.)
        A_safe = torch.nan_to_num(A.float(), nan=0.0, posinf=1.0, neginf=0.0)

        with torch.no_grad():
            P_detached = A_safe.detach().mean(dim=(1, 2))  # (N_batch,)
            # Scatter into full EMA buffer: decay all, then add this batch's slice
            ema_expert_usage.mul_(args.ema_beta)
            ema_expert_usage.scatter_add_(
                0, pool_idx_t, P_detached * (1.0 - args.ema_beta)
            )
        # Gradient-carrying term for this batch's experts only
        P_live = A_safe.mean(dim=(1, 2))                      # (N_batch,), has grad
        ema_subset = ema_expert_usage[pool_idx_t].detach()    # (N_batch,)
        # L_lb = λ · N_full · Σ_i f_ema_i · P_i  (larger when one expert dominates)
        loss_entropy = lam * N_experts * (ema_subset * P_live).sum()

        # ── VAE encode → latents ──────────────────────────
        image_tensor = _IMG_TRANSFORM(image).unsqueeze(0)
        image_tensor = image_tensor.to(device=device, dtype=weight_dtype)

        with torch.no_grad():
            latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

        # ── Noise + timestep ──────────────────────────────
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (1,), device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ── SDXL conditioning ─────────────────────────────
        prompt_embeds, pooled_embeds = encode_prompt(pipeline, prompt, device)
        add_time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            dtype=weight_dtype, device=device,
        )
        added_cond_kwargs = {
            "text_embeds": pooled_embeds.to(dtype=weight_dtype),
            "time_ids": add_time_ids,
        }

        # ── Inject LoRA hooks (gradient-compatible) ───────
        hooks = apply_lora_hooks_with_grad(
            pipeline.unet, synth_lora, alpha=args.lora_alpha
        )

        # ── UNet forward ──────────────────────────────────
        noise_pred = pipeline.unet(
            noisy_latents.to(dtype=weight_dtype),
            timesteps,
            encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        remove_hooks(hooks)

        # ── Total loss = LDM + entropy reg ────────────────
        loss_ldm = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss = loss_ldm + loss_entropy

        # Guard: skip update on NaN/Inf loss OR NaN/Inf gradients.
        # fp16 UNet backward can produce Inf gradients; clip_grad_norm_ then
        # returns Inf, and Inf*0=NaN (IEEE 754) would corrupt encoder params.
        # Fix: sanitize gradient elements individually with nan_to_num_ BEFORE
        # clipping.  Finite elements still contribute; only the overflowed
        # positions are zeroed.  The step always proceeds when loss is finite.
        if torch.isfinite(loss):
            loss.backward()
            # Zero any Inf/NaN gradient elements in-place before clipping
            had_bad_grad = False
            for p in model.encoder.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        had_bad_grad = True
            if had_bad_grad:
                running_skipped += 1
            torch.nn.utils.clip_grad_norm_(
                model.encoder.parameters(), args.gradient_clip
            )
            optimizer.step()
        else:
            running_skipped += 1

        running_loss_ldm += loss_ldm.item()
        running_loss_ent += loss_entropy.item()
        running_loss_total += loss.item()
        step += 1

        if step % args.log_every == 0:
            avg_ldm = running_loss_ldm / args.log_every
            avg_ent = running_loss_ent / args.log_every
            avg_total = running_loss_total / args.log_every
            n_skip = running_skipped
            running_loss_ldm = 0.0
            running_loss_ent = 0.0
            running_loss_total = 0.0
            running_skipped = 0
            log(
                f"step={step:6d}/{args.max_steps}  "
                f"total={avg_total:.6f}  "
                f"ldm={avg_ldm:.6f}  "
                f"lb={avg_ent:.6f}  "
                f"λ={lam:.4f}  "
                f"τ={get_temperature_v3(step, args.tau_start, args.tau_mid, args.tau_end, args.tau_mid_step, args.max_steps):.3f}  "
                f"top1={int(ema_expert_usage.argmax())}({ema_expert_usage.max():.3f})  "
                f"skip={n_skip}  "
                f"lr={lr:.2e}"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.output_dir, step, model, optimizer, loss.item())

    log_fh.close()
    print(f"\n[Stage2-v2] Training complete. Outputs at: {args.output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
