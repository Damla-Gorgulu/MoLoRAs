#!/usr/bin/env python3
"""Mini Stage-1 training for MoELoRA v3 query/key routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.data.dataset import ExactExemplarStage1Dataset, exact_stage1_collate_fn
from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v3 import MoELoRAv3


def parse_args():
    p = argparse.ArgumentParser(description="MoELoRA v3 mini Stage-1 routing canary")
    p.add_argument("--zoo_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/zoo/bloras")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/cache_v3")
    p.add_argument("--manifest_path", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/manifest.json")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_attention/mini_v3_clip")
    p.add_argument("--image_encoder", choices=["clip", "vae"], default="clip")
    p.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    p.add_argument("--vae_model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--rank_tokens", type=int, default=16)
    p.add_argument("--max_tensor_groups", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--query_layers", type=int, default=2)
    p.add_argument("--key_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--views_per_style", type=int, default=32)
    p.add_argument("--min_pool_size", type=int, default=4)
    p.add_argument("--max_pool_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_rebuild_cache", action="store_true")
    return p.parse_args()


def compute_ce_loss(A: torch.Tensor, gt_pos: int, device: torch.device) -> torch.Tensor:
    """CE over experts, averaged over tensor/rank routing slots."""
    N, T, R = A.shape
    log_A = (A + 1e-8).log().permute(1, 2, 0).reshape(T * R, N)
    targets = torch.full((T * R,), gt_pos, dtype=torch.long, device=device)
    return F.nll_loss(log_A, targets)


def save_checkpoint(args, step: int, model: MoELoRAv3, optimizer, loss: float) -> None:
    out = Path(args.output_dir)
    ckpt_dir = out / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    query_state = {
        k: v
        for k, v in model.query_encoder.state_dict().items()
        if not k.startswith("_clip_model.") and not k.startswith("_vae.")
    }
    payload = {
        "step": step,
        "loss": loss,
        "version": "v3-mini",
        "query_encoder_state_dict": query_state,
        "key_encoder_state_dict": model.key_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }
    torch.save(payload, ckpt_dir / "checkpoint.pt")
    torch.save(payload, out / "latest.pt")
    print(f"[save] {ckpt_dir}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v3-mini] device={device}")
    print(f"[v3-mini] image_encoder={args.image_encoder}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    pool = LoRAPool(
        zoo_dir=args.zoo_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
        device=str(device),
    )
    model = MoELoRAv3(
        pool=pool,
        image_encoder=args.image_encoder,
        rank_tokens=args.rank_tokens,
        max_tensor_groups=args.max_tensor_groups,
        d_model=args.d_model,
        query_layers=args.query_layers,
        key_layers=args.key_layers,
        num_heads=args.num_heads,
        clip_model_id=args.clip_model_id,
        vae_model_id=args.vae_model_id,
    ).to(device)

    dataset = ExactExemplarStage1Dataset(
        pool=pool,
        manifest_path=args.manifest_path,
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
        views_per_style=args.views_per_style,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=exact_stage1_collate_fn,
    )
    iterator = iter(loader)

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log_path = out / "train_log.txt"
    running_loss = 0.0
    running_entropy = 0.0
    running_top1 = 0
    running_count = 0

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        losses = []
        entropies = []
        top1_correct = 0

        for image, pool_indices, gt_pos in zip(
            batch["images"], batch["pool_indices"], batch["gt_positions"]
        ):
            A, _ = model(image, pool_indices, temperature=args.temperature)
            loss = compute_ce_loss(A, gt_pos, device)
            losses.append(loss)
            entropies.append(model.attention_entropy(A))
            avg = A.mean(dim=(1, 2))
            top1_correct += int(avg.argmax().item() == gt_pos)

        loss = torch.stack(losses).mean()
        loss.backward()
        for p in model.trainable_parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), args.gradient_clip)
        optimizer.step()

        batch_n = len(losses)
        running_loss += loss.item() * batch_n
        running_entropy += torch.stack(entropies).mean().item() * batch_n
        running_top1 += top1_correct
        running_count += batch_n

        if step % args.log_every == 0 or step == 1:
            mean_loss = running_loss / max(running_count, 1)
            mean_entropy = running_entropy / max(running_count, 1)
            top1 = running_top1 / max(running_count, 1)
            line = (
                f"step={step:6d}/{args.max_steps} loss={mean_loss:.6f} "
                f"entropy={mean_entropy:.4f} top1={top1:.3f} "
                f"max_ent={math.log(args.max_pool_size):.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")
            running_loss = running_entropy = 0.0
            running_top1 = running_count = 0

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args, step, model, optimizer, loss.item())


if __name__ == "__main__":
    main()
