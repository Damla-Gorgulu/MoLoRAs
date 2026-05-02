#!/usr/bin/env python3
"""Residual LoRA autoencoder: learns target - mean_template for N styles."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.lora_autoencoder.dataset import (
    StyleLoRAAutoencoderDataset,
    make_metadata_tensors,
)
from lora_attention.lora_autoencoder.model import (
    StyleLoRAAutoencoder,
    reconstruction_losses,
)


def parse_args():
    p = argparse.ArgumentParser(description="Residual style-only B-LoRA autoencoder")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/residual16_v1")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=41472)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--tensor_weight", type=float, default=1.0)
    p.add_argument("--delta_weight", type=float, default=0.0)
    p.add_argument("--cos_weight", type=float, default=0.0)
    p.add_argument("--rel_weight", type=float, default=0.0)
    p.add_argument("--norm_weight", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lora-autoencoder")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def log_metrics(metrics, step, max_steps, log_path, wandb):
    line = (
        f"step={step:05d}/{max_steps} loss={metrics['loss'].item():.6f} "
        f"mse={metrics['tensor_mse'].item():.6f} "
        f"cos={metrics['cos'].item():.4f} rel={metrics['rel'].item():.4f} "
        f"nr={metrics['norm_ratio'].item():.4f}"
    )
    if "full_cos" in metrics:
        line += f" fcos={metrics['full_cos'].item():.4f} frel={metrics['full_rel'].item():.4f} fnr={metrics['full_nr'].item():.4f}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")
    if wandb is not None:
        wandb.log({k: float(v.item()) for k, v in metrics.items()}, step=step)


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
        wandb = wandb_lib.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    meta = make_metadata_tensors(dataset.specs)
    meta_dev = {k: v.to(device) for k, v in meta.items()}
    log_path = out / "train_log.txt"

    # Compute mean template from ALL loaded LoRAs
    print(f"[residual] loading {len(dataset)} LoRAs to compute mean template...", flush=True)
    sum_down1280 = torch.zeros(1, dataset.num_pairs, dataset.rank, 1280)
    sum_down2048 = torch.zeros(1, dataset.num_pairs, dataset.rank, 2048)
    sum_up = torch.zeros(1, dataset.num_pairs, dataset.rank, 1280)
    for i in range(len(dataset)):
        s = dataset[i]
        sum_down1280 += s["down1280"].unsqueeze(0)
        sum_down2048 += s["down2048"].unsqueeze(0)
        sum_up += s["up"].unsqueeze(0)
    mean_template = {
        "down1280": (sum_down1280 / len(dataset)).to(device),
        "down2048": (sum_down2048 / len(dataset)).to(device),
        "up": (sum_up / len(dataset)).to(device),
        "d_in": dataset[0]["d_in"].unsqueeze(0).to(device),
    }

    # Report mean-template baseline on first LoRA
    with torch.no_grad():
        target0 = {k: dataset[0][k].unsqueeze(0).to(device) for k in ["down1280", "down2048", "up", "d_in"]}
        mean_pred = {k: mean_template[k].clone() for k in ["down1280", "down2048", "up"]}
        mean_pred["d_in"] = target0["d_in"]
        mean_metrics = reconstruction_losses(mean_pred, target0)
        print(f"[mean_template_baseline] cos={mean_metrics['cos'].item():.4f} "
              f"rel={mean_metrics['rel'].item():.4f} mse={mean_metrics['tensor_mse'].item():.6f}", flush=True)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    iterator = iter(loader)

    model = StyleLoRAAutoencoder(
        num_pairs=dataset.num_pairs, rank=dataset.rank,
        latent_dim=args.latent_dim, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[ae_residual] dataset={len(dataset)} num_pairs={dataset.num_pairs} rank={dataset.rank} "
          f"latent_dim={args.latent_dim} device={device}", flush=True)

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)

        residual = {
            "down1280": batch["down1280"].unsqueeze(0) if batch["down1280"].ndim == 3 else batch["down1280"],
            "down2048": batch["down2048"].unsqueeze(0) if batch["down2048"].ndim == 3 else batch["down2048"],
            "up": batch["up"].unsqueeze(0) if batch["up"].ndim == 3 else batch["up"],
            "d_in": batch["d_in"].unsqueeze(0) if batch["d_in"].ndim == 1 else batch["d_in"],
        }

        # fix shapes for unsqueeze above if batch_size=1
        for k in ["down1280", "down2048", "up", "d_in"]:
            if batch[k].dim() == residual[k].dim() - 1:
                residual[k] = batch[k].unsqueeze(0)
            else:
                residual[k] = batch[k]

        residual_target = {
            "down1280": residual["down1280"] - mean_template["down1280"],
            "down2048": residual["down2048"] - mean_template["down2048"],
            "up": residual["up"] - mean_template["up"],
            "d_in": residual["d_in"],
        }

        opt.zero_grad(set_to_none=True)
        pred_residual = model(residual_target, meta_dev)
        metrics = reconstruction_losses(
            pred_residual, residual_target,
            tensor_weight=args.tensor_weight, delta_weight=args.delta_weight,
            cos_weight=args.cos_weight, rel_weight=args.rel_weight,
            norm_weight=args.norm_weight,
        )
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                pred_full = {
                    "down1280": pred_residual["down1280"].detach() + mean_template["down1280"],
                    "down2048": pred_residual["down2048"].detach() + mean_template["down2048"],
                    "up": pred_residual["up"].detach() + mean_template["up"],
                    "d_in": residual["d_in"],
                }
                full_metrics = reconstruction_losses(pred_full, residual)
                metrics["full_cos"] = full_metrics["cos"]
                metrics["full_rel"] = full_metrics["rel"]
                metrics["full_nr"] = full_metrics["norm_ratio"]
            log_metrics(metrics, step, args.max_steps, log_path, wandb)

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_dir = out / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(), "config": vars(args),
                "style_names": dataset.style_names,
                "pair_specs": [s.__dict__ for s in dataset.specs],
                "mean_template": {k: v.cpu() for k, v in mean_template.items()},
            }, ckpt_dir / "checkpoint.pt")
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "mean_template": {k: v.cpu() for k, v in mean_template.items()},
            }, out / "latest.pt")
            print(f"[save] {ckpt_dir}", flush=True)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
