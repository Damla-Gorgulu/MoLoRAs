#!/usr/bin/env python3
"""
v2.1 Stage 1 Training: One-Hot Cross-Entropy Routing (default) or Soft-KL (legacy)

Trains the LoRARankEncoder to map WikiArt style images to per-tensor
rank-level attention distributions.

Target modes:
  ce  (default) — one-hot cross-entropy on the GT expert label.
                  Forces encoder to learn *style* invariants; no CLIP similarity
                  file required.  Generally preferred.
  kl  (legacy)  — KL divergence against CLIP-similarity soft targets.
                  Requires --similarity_path.  Prone to content-based confusion
                  because CLIP clusters by subject matter, not artistic style.

Key differences from v1.0:
  - LoRARankEncoder replaces RoutingMLP (2.2M vs 17.3M params)
  - Per-tensor attention: A ∈ ℝ^{N×T×r} instead of A ∈ ℝ^{N×r}
  - Product-space synthesis (default): W_synth = Σ_i A_i(W_up_i @ W_down_i)
  - WikiArt dataset (~80k images) instead of 109 zoo images
  - τ = 1.0 at training (no temperature hack)

Loss (CE mode, default):
  A_avg = A.mean(dim=2)            # (N, T) — average over rank
  L = CrossEntropy(A_avg.T, gt_pos_broadcast)

Example (CE, recommended — no CLIP similarity file needed):
    python train_stage1_v2.py \\
        --output_dir /scratch/eyavuz21/lora_attention/stage1_v21 \\
        --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \\
        --cache_dir /scratch/eyavuz21/lora_attention \\
        --wikiart_dir /home/eyavuz21/datasets/wikiart \\
        --label_map_path /scratch/eyavuz21/lora_attention/wikiart_label_map.json \\
        --target_mode ce \\
        --max_steps 15000 \\
        --lr 3e-4

Example (KL legacy mode):
    python train_stage1_v2.py \\
        --output_dir /scratch/eyavuz21/lora_attention/stage1_v2_kl \\
        --similarity_path /scratch/eyavuz21/lora_attention/clip_similarity.pt \\
        --target_mode kl \\
        --max_steps 15000 \\
        --lr 3e-4
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
sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.data.dataset import (
    WikiArtStage1Dataset,
    wikiart_stage1_collate_fn,
)


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA v2.0 Stage 1 Training")

    # Paths
    p.add_argument("--zoo_dir", type=str,
                   default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--output_dir", type=str,
                   default="/scratch/eyavuz21/lora_attention/stage1_v2")
    p.add_argument("--wikiart_dir", type=str,
                   default="/home/eyavuz21/datasets/wikiart")
    p.add_argument("--similarity_path", type=str, default=None,
                   help="Path to clip_similarity.pt. Required only for --target_mode kl.")
    p.add_argument("--target_mode", type=str, default="ce", choices=["ce", "kl"],
                   help="'ce': one-hot cross-entropy (default, no CLIP sim needed). "
                        "'kl': soft KL vs CLIP-similarity targets (legacy).")
    p.add_argument("--label_map_path", type=str,
                   default="/scratch/eyavuz21/lora_attention/wikiart_label_map.json")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a v2.0 checkpoint to resume from.")

    # Model
    p.add_argument("--clip_model_id", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--clip_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--no_normalize_keys", action="store_true",
                   help="Disable L2 normalisation of encoder keys.")
    p.add_argument("--force_rebuild_cache", action="store_true")

    # Dataset
    p.add_argument("--tau_label", type=float, default=0.3,
                   help="Soft target temperature. Lower=sharper.")
    p.add_argument("--min_pool_size", type=int, default=5)
    p.add_argument("--max_pool_size", type=int, default=20)
    p.add_argument("--max_images_per_style", type=int, default=500,
                   help="Cap WikiArt images per style for balanced training.")
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--max_steps", type=int, default=15_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)

    # Logging / saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────
def get_lr(step: int, warmup_steps: int, base_lr: float, max_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


import math


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────
def save_checkpoint(
    output_dir: str,
    step: int,
    model: MoELoRAv2,
    optimizer: torch.optim.Optimizer,
    loss: float,
) -> None:
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


def load_checkpoint(
    resume_from: str,
    model: MoELoRAv2,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"  [resume] Step {ckpt['step']}, loss={ckpt['loss']:.6f}")
    return start_step


# ──────────────────────────────────────────────────────────────
# Per-tensor CE loss (default, v2.1)
# ──────────────────────────────────────────────────────────────
def compute_ce_loss(
    A: torch.Tensor,  # (N, T, r) — predicted attention
    gt_pos: int,      # index of GT expert within pool
    device: torch.device,
) -> torch.Tensor:
    """
    One-hot cross-entropy routing loss.

    For each tensor group t, we want A[gt_pos, t, :] >> A[other, t, :].
    We average over rank dimension first, then apply NLL loss over T samples.

    Args:
        A:      (N, T, r) — predicted attention (already softmax-normalised over N).
        gt_pos: index of the GT expert in the sampled pool.
        device: torch device.

    Returns:
        Scalar loss — mean NLL over T tensor groups.
    """
    N, T, r = A.shape
    A_avg = A.mean(dim=2)                                          # (N, T)
    log_A = (A_avg + 1e-8).log()                                   # (N, T)
    gt_targets = torch.full((T,), gt_pos, dtype=torch.long, device=device)
    return F.nll_loss(log_A.T, gt_targets)                         # mean over T


# ──────────────────────────────────────────────────────────────
# Per-tensor KL loss (legacy, --target_mode kl)
# ──────────────────────────────────────────────────────────────
def compute_kl_loss(
    A: torch.Tensor,          # (N, T, r) — predicted attention
    soft_target: torch.Tensor, # (N,) — expert-level soft distribution
) -> torch.Tensor:
    """
    KL divergence between predicted per-tensor attention and soft target.

    The soft target is expert-level (N,) and is broadcast to all T tensors
    and all r rank positions. This is correct because the soft target says
    "how much of expert i's style is relevant" — the same question regardless
    of which layer or rank we're looking at.

    Args:
        A: (N, T, r) — predicted attention from MoELoRAv2.
        soft_target: (N,) — soft distribution from CLIP similarity.

    Returns:
        Scalar loss — average KL divergence across all T×r positions.
    """
    N, T, r = A.shape

    # Broadcast soft_target to (N, T, r)
    target = soft_target.view(N, 1, 1).expand(N, T, r)  # (N, T, r)

    # KL(target || predicted) = Σ target * log(target / predicted)
    # = Σ target * (log_target - log_predicted)
    log_A = (A + 1e-8).log()       # (N, T, r)
    log_target = (target + 1e-8).log()

    kl = (target * (log_target - log_A)).sum(dim=0)  # (T, r)
    return kl.mean()


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage1-v2] Device: {device}")

    # ── Pool ──────────────────────────────────────────────────
    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
        device=str(device),
    )

    # ── Model ─────────────────────────────────────────────────
    model = MoELoRAv2(
        pool=pool,
        clip_model_id=args.clip_model_id,
        rank=args.rank,
        clip_dim=args.clip_dim,
        hidden_dim=args.hidden_dim,
        normalize_keys=not args.no_normalize_keys,
    ).to(device)

    # Ensure CLIP is loaded
    model._ensure_clip(device)

    n_params = sum(p.numel() for p in model.encoder.parameters())
    T = model.num_tensor_groups
    print(f"[Stage1-v2] Encoder params: {n_params:,}")
    print(f"[Stage1-v2] Tensor groups (T): {T}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Validate args ─────────────────────────────────────────
    if args.target_mode == "kl" and args.similarity_path is None:
        raise ValueError(
            "--similarity_path is required when --target_mode kl. "
            "Use --target_mode ce (default) to train without a CLIP similarity file."
        )

    # ── Dataset ───────────────────────────────────────────────
    sim_path = args.similarity_path if args.target_mode == "kl" else None
    dataset = WikiArtStage1Dataset(
        pool=pool,
        wikiart_dir=args.wikiart_dir,
        label_map_path=args.label_map_path,
        similarity_path=sim_path,
        tau_label=args.tau_label,
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
        max_images_per_style=args.max_images_per_style,
        rank=args.rank,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=wikiart_stage1_collate_fn,
        drop_last=True,
    )

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer)

    # ── Output dir & log ──────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train_log.txt"
    log_fh = open(log_path, "a")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*60}")
    log(f"Stage 1 v2.1 Training  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pool: {pool.num_experts} experts")
    log(f"Encoder params: {n_params:,}")
    log(f"Tensor groups: {T}")
    log(f"target_mode={args.target_mode}, max_steps={args.max_steps}, lr={args.lr}, batch={args.batch_size}")
    log(f"τ_label={args.tau_label}, pool ∈ [{args.min_pool_size}, {args.max_pool_size}]")
    log(f"Dataset size: {len(dataset)}")
    log(f"{'='*60}")

    # ── Training ──────────────────────────────────────────────
    model.encoder.train()
    step = start_step
    running_loss = 0.0
    running_entropy = 0.0

    data_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Update LR
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_entropy = 0.0
        n_samples = len(batch["images"])

        for i in range(n_samples):
            image = batch["images"][i]
            pool_indices = batch["pool_indices"][i]

            # Encode with frozen CLIP
            q = model.encode_image(image, device)  # (1, clip_dim)

            # Forward: per-tensor attention (τ=1.0 at training)
            A, _ = model.forward(q, pool_indices, temperature=1.0, product_space=True)  # A: (N, T, r)

            # Loss — CE (default) or KL (legacy)
            if args.target_mode == "ce":
                gt_pos = batch["gt_positions"][i]
                loss_i = compute_ce_loss(A, gt_pos, device)
            else:
                soft_target = batch["soft_targets"][i].to(device)  # (N,)
                loss_i = compute_kl_loss(A, soft_target)
            batch_loss = batch_loss + loss_i

            # Track entropy for monitoring
            with torch.no_grad():
                batch_entropy += model.attention_entropy(A).item()

        batch_loss = batch_loss / n_samples
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.encoder.parameters(), args.gradient_clip
        )
        optimizer.step()

        running_loss += batch_loss.item()
        running_entropy += batch_entropy / n_samples
        step += 1

        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_entropy = running_entropy / args.log_every
            running_loss = 0.0
            running_entropy = 0.0
            log(
                f"step={step:6d}/{args.max_steps}  "
                f"loss={avg_loss:.6f}  "
                f"entropy={avg_entropy:.4f}  "
                f"lr={lr:.2e}"
            )

        # Saving
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                args.output_dir, step, model, optimizer, batch_loss.item()
            )

    log_fh.close()
    print(f"\n[Stage1-v2] Training complete. Outputs at: {args.output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
