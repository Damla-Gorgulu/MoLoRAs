#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.lora_autoencoder.dataset import (  # noqa: E402
    StyleLoRAAutoencoderDataset,
    make_metadata_tensors,
)
from lora_attention.lora_autoencoder.model import (  # noqa: E402
    StyleLoRAAutoencoder,
    reconstruction_losses,
)


def parse_args():
    p = argparse.ArgumentParser(description="Style-only B-LoRA reconstructive autoencoder")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/overfit16_v1")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=65536)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--tensor_weight", type=float, default=1.0)
    p.add_argument("--delta_weight", type=float, default=0.1)
    p.add_argument("--cos_weight", type=float, default=0.01)
    p.add_argument("--rel_weight", type=float, default=1.0)
    p.add_argument("--norm_weight", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lora-autoencoder")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    wandb = None
    if args.wandb:
        import wandb as wandb_lib
        if os.environ.get("WANDB_API_KEY"):
            wandb_lib.login(key=os.environ["WANDB_API_KEY"])
        wandb = wandb_lib.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    meta = make_metadata_tensors(dataset.specs)
    loader = DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    iterator = iter(loader)

    model = StyleLoRAAutoencoder(
        num_pairs=dataset.num_pairs,
        rank=dataset.rank,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    print(
        f"[ae] dataset={len(dataset)} num_pairs={dataset.num_pairs} rank={dataset.rank} "
        f"latent_dim={args.latent_dim} device={device}",
        flush=True,
    )

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        meta_dev = {k: v.to(device) for k, v in meta.items()}

        opt.zero_grad(set_to_none=True)
        pred = model(batch, meta_dev)
        metrics = reconstruction_losses(
            pred,
            batch,
            tensor_weight=args.tensor_weight,
            delta_weight=args.delta_weight,
            cos_weight=args.cos_weight,
            rel_weight=args.rel_weight,
            norm_weight=args.norm_weight,
        )
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            line = (
                f"step={step:05d}/{args.max_steps} loss={metrics['loss'].item():.6f} "
                f"tensor_mse={metrics['tensor_mse'].item():.6f} "
                f"delta_mse={metrics['delta_mse'].item():.6f} "
                f"delta_rel={metrics['delta_rel'].item():.4f} "
                f"delta_rel_loss={metrics['delta_rel_loss'].item():.4f} "
                f"cos={metrics['cos'].item():.4f} rel={metrics['rel'].item():.4f} "
                f"norm_ratio={metrics['norm_ratio'].item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")
            if wandb is not None:
                wandb.log({k: float(v.item()) for k, v in metrics.items()}, step=step)

        if step % args.save_every == 0 or step == args.max_steps:
            payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": vars(args),
                "style_names": dataset.style_names,
                "pair_specs": [s.__dict__ for s in dataset.specs],
            }
            ckpt_dir = out / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, ckpt_dir / "checkpoint.pt")
            torch.save(payload, out / "latest.pt")
            print(f"[save] {ckpt_dir}", flush=True)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
