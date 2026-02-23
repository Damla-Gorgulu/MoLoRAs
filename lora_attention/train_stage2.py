#!/usr/bin/env python3
"""
Stage 2 Training: Hold-out Reconstruction via Diffusion Loss

Fine-tunes the RoutingMLP to reconstruct a query style from a pool that
EXCLUDES the ground-truth LoRA, forcing the model to learn combinations.

Loss: Standard LDM noise prediction loss (L_simple).

Only the RoutingMLP is updated. SDXL base + VAE + CLIP + LoRA zoo = frozen.

Gradients flow:
    L_LDM → UNet (frozen hooks) → synth_lora (W_down, W_up)
           → attention A → RoutingMLP (trainable)

Requires a GPU with ≥ 24 GB VRAM for SDXL in fp16.

Example:
    python train_stage2.py \
        --stage1_ckpt /scratch/eyavuz21/lora_attention/stage1/latest.pt \
        --output_dir  /scratch/eyavuz21/lora_attention/stage2 \
        --zoo_dir     /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
        --cache_dir   /scratch/eyavuz21/lora_attention \
        --max_steps   5000
"""

import argparse
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
# parents[1] = MoLoRAs/ (the directory that contains lora_attention/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora import MoELoRA
from lora_attention.data.dataset import Stage2Dataset, stage2_collate_fn
from lora_attention.utils.lora_inject import (
    apply_lora_hooks_with_grad,
    remove_hooks,
)

# Image normalisation for VAE input (SDXL expects [-1, 1], 1024×1024)
_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # [0,1] → [-1,1]
])


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA Stage 2 Training")

    # Paths
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--output_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention/stage2")
    p.add_argument("--image_dirs", type=str, nargs="+", default=None)
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Optional JSON list of text prompts for conditioning.")
    p.add_argument("--stage1_ckpt", type=str, default=None,
                   help="Stage 1 checkpoint to initialise RoutingMLP weights.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Stage 2 checkpoint to resume from.")

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
    p.add_argument("--force_rebuild_cache", action="store_true")

    # Dataset
    p.add_argument("--min_pool_size", type=int, default=3)
    p.add_argument("--max_pool_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--max_steps", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])

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


def load_sdxl_pipeline(args, device):
    """Load frozen SDXL pipeline with fp16-fix VAE."""
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

    print(f"[Stage2] Loading SDXL: {args.pretrained_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)

    # Freeze everything
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

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)           # (1, seq, 2048)
    # Pooled embed from the second encoder
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
    torch.save({
        "step": step,
        "routing_mlp_state_dict": model.routing_mlp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, ckpt_dir / "checkpoint.pt")
    torch.save({
        "step": step,
        "routing_mlp_state_dict": model.routing_mlp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, Path(output_dir) / "latest.pt")
    print(f"  [save] {ckpt_dir}")


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Stage 2 requires a GPU."
    print(f"[Stage2] Device: {device}")

    weight_dtype = {"fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "no":   torch.float32}[args.mixed_precision]

    # ── Pool ──────────────────────────────────────────────────
    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    # ── MoELoRA (RoutingMLP trainable) ────────────────────────
    model = MoELoRA(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Load Stage 1 weights if provided
    if args.stage1_ckpt is not None:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        model.routing_mlp.load_state_dict(ckpt["routing_mlp_state_dict"])
        print(f"[Stage2] Loaded Stage 1 weights from {args.stage1_ckpt}")

    # Ensure CLIP is initialised
    model._ensure_clip(device)

    # ── SDXL pipeline ─────────────────────────────────────────
    pipeline = load_sdxl_pipeline(args, device)
    noise_scheduler = pipeline.scheduler

    # ── Optimizer (only RoutingMLP) ───────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Dataset ───────────────────────────────────────────────
    dataset = Stage2Dataset(
        pool=pool,
        image_dirs=args.image_dirs,
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
        rank=args.rank,
        seed=args.seed,
        prompts_file=args.prompts_file,
        # No image_transform: keep images as PIL.
        # VAE transform is applied in the training loop; CLIP uses raw PIL.
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time (variable N)
        shuffle=True,
        num_workers=0,  # Must be 0 when images are already tensors and pool is in-memory
        collate_fn=stage2_collate_fn,
        drop_last=True,
    )

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.routing_mlp.load_state_dict(ckpt["routing_mlp_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"[Stage2] Resumed from step {ckpt['step']}")

    # ── Output / logging ──────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train_log.txt"
    log_fh = open(log_path, "a")

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*60}")
    log(f"Stage 2 Training  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pool: {pool.num_experts} experts (GT excluded from pool)")
    log(f"Routing MLP params: {sum(p.numel() for p in model.routing_mlp.parameters()):,}")
    log(f"max_steps={args.max_steps}, lr={args.lr}, mixed_precision={args.mixed_precision}")
    log(f"{'='*60}")

    # ── Training ──────────────────────────────────────────────
    model.routing_mlp.train()
    step = start_step
    running_loss = 0.0
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

        optimizer.zero_grad()

        # Single sample per step (batch_size=1, but loop is explicit)
        image = batch["images"][0]          # PIL.Image (RGB)
        prompt = batch["prompts"][0]
        pool_indices = batch["pool_indices"][0]

        # Prepare VAE tensor from PIL image
        image_tensor = _IMG_TRANSFORM(image).unsqueeze(0)  # (1, 3, 1024, 1024)
        image_tensor = image_tensor.to(device=device, dtype=weight_dtype)

        # ── CLIP encode image (expects PIL) ────────────────
        q = model.encode_image(image, device)  # (1, clip_dim)

        # ── MoELoRA forward (synth_lora has grad_fn) ──────
        A, synth_lora = model.forward(q, pool_indices)

        # ── Prepare text conditioning ──────────────────────
        prompt_embeds, pooled_embeds = encode_prompt(pipeline, prompt, device)

        # ── VAE encode → latents ──────────────────────────
        with torch.no_grad():
            latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

        # ── Sample noise & timestep ───────────────────────
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (1,),
            device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ── Build SDXL time ids ───────────────────────────
        add_time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            dtype=weight_dtype, device=device
        )
        added_cond_kwargs = {
            "text_embeds": pooled_embeds.to(dtype=weight_dtype),
            "time_ids": add_time_ids,
        }

        # ── Inject LoRA via hooks (gradient-compatible) ───
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

        # ── Remove hooks ──────────────────────────────────
        remove_hooks(hooks)

        # ── LDM loss ──────────────────────────────────────
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.routing_mlp.parameters(), args.gradient_clip
        )
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            log(
                f"step={step:6d}/{args.max_steps}  "
                f"loss={avg_loss:.6f}  lr={lr:.2e}"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.output_dir, step, model, optimizer, loss.item())

    log_fh.close()
    print(f"\n[Stage2] Training complete. Outputs at: {args.output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
