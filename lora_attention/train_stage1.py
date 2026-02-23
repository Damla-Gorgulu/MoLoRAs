#!/usr/bin/env python3
"""
Stage 1 Training: Ground Truth Mapping

Train the RoutingMLP to map a style image to its corresponding
expert LoRA via rank-level attention.

Loss: MSE between attention matrix A ∈ ℝ^{N×rank} and one-hot
      target (GT expert row = 1, all others = 0).

Only the RoutingMLP is updated; CLIP and LoRA pool are frozen.

Example:
    python train_stage1.py \
        --output_dir /scratch/eyavuz21/lora_attention/stage1 \
        --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
        --cache_dir /scratch/eyavuz21/lora_attention \
        --max_steps 10000 \
        --lr 1e-4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ──────────────────────────────────────────────
# parents[1] = MoLoRAs/ (the directory that contains lora_attention/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora import MoELoRA
from lora_attention.data.dataset import Stage1Dataset, stage1_collate_fn


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA Stage 1 Training")

    # Paths
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--output_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention/stage1")
    p.add_argument("--image_dirs", type=str, nargs="+", default=None,
                   help="Dirs to search for style images. "
                        "Defaults to blora_zoo/style_images/")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a checkpoint to resume from.")

    # Model
    p.add_argument("--clip_model_id", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--force_rebuild_cache", action="store_true")

    # Dataset
    p.add_argument("--min_pool_size", type=int, default=3)
    p.add_argument("--max_pool_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of (image, pool) samples per gradient step. "
                        "Each sample has its own N; loss is averaged.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. Use 0 to keep everything in main "
                        "process (avoids worker crashes from large LoRAPool "
                        "pickles). Set 4 on SLURM GPU nodes.")

    # Logging / saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Learning rate warmup scheduler
# ──────────────────────────────────────────────────────────────
def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────
def save_checkpoint(
    output_dir: str,
    step: int,
    model: MoELoRA,
    optimizer: torch.optim.Optimizer,
    loss: float ) -> None:
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "routing_mlp_state_dict": model.routing_mlp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, ckpt_dir / "checkpoint.pt")

    # Also save a "latest" pointer for easy resuming
    torch.save({
        "step": step,
        "routing_mlp_state_dict": model.routing_mlp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, Path(output_dir) / "latest.pt")

    print(f"  [save] {ckpt_dir}")


def load_checkpoint(
    resume_from: str,
    model: MoELoRA,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    model.routing_mlp.load_state_dict(ckpt["routing_mlp_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"  [resume] Resuming from step {ckpt['step']}, loss={ckpt['loss']:.6f}")
    return start_step


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage1] Device: {device}")

    # ── Pool ──────────────────────────────────────────────────
    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
        device=str(device),
    )

    # ── Model ─────────────────────────────────────────────────
    model = MoELoRA(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Ensure CLIP is loaded (lazy init) before training loop
    dummy = torch.zeros(1, 3, 224, 224)
    model._ensure_clip(device)

    # ── Optimizer (only RoutingMLP params) ────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Dataset ───────────────────────────────────────────────
    dataset = Stage1Dataset(
        pool=pool,
        image_dirs=args.image_dirs,
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
        rank=args.rank,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=stage1_collate_fn,
        drop_last=True,
    )

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer)

    # ── Output dir & log file ─────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train_log.txt"
    log_fh = open(log_path, "a")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*60}")
    log(f"Stage 1 Training  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pool: {pool.num_experts} experts, feature_dim={pool.feature_dim}")
    log(f"Routing MLP params: {sum(p.numel() for p in model.routing_mlp.parameters()):,}")
    log(f"max_steps={args.max_steps}, lr={args.lr}, batch={args.batch_size}")
    log(f"pool_size ∈ [{args.min_pool_size}, {args.max_pool_size}]")
    log(f"{'='*60}")

    # ── Training ──────────────────────────────────────────────
    model.routing_mlp.train()
    step = start_step
    running_loss = 0.0

    data_iter = iter(loader)

    while step < args.max_steps:
        # Refill iterator when exhausted
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
        batch_loss = torch.tensor(0.0, device=device)
        n_samples = len(batch["images"])

        for i in range(n_samples):
            image = batch["images"][i]
            pool_indices = batch["pool_indices"][i]
            target = batch["targets"][i].to(device)        # (N, rank)

            # Encode with frozen CLIP
            q = model.encode_image(image, device)           # (1, clip_dim)

            # Forward
            A, _ = model.forward(q, pool_indices)           # A: (N, rank)

            # MSE loss against one-hot GT target
            loss_i = F.mse_loss(A, target)
            batch_loss = batch_loss + loss_i

        batch_loss = batch_loss / n_samples
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.routing_mlp.parameters(), args.gradient_clip
        )
        optimizer.step()

        running_loss += batch_loss.item()
        step += 1

        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            log(
                f"step={step:6d}/{args.max_steps}  "
                f"loss={avg_loss:.6f}  lr={lr:.2e}"
            )

        # Saving
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                args.output_dir, step, model, optimizer, batch_loss.item()
            )

    log_fh.close()
    print(f"\n[Stage1] Training complete. Outputs at: {args.output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
