#!/usr/bin/env python3
"""Residual learning: train the model to predict target minus the mean of 2 LoRAs.
Tests whether removing the common component enables full memorization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import StyleLoRAAutoencoder, reconstruction_losses


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual learning: target - mean_LoRA.")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--other_idx", type=int, default=1)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/residual_baseline")
    p.add_argument("--latent_dim", type=int, default=41472)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_to_batch(sample: dict) -> dict:
    return {k: sample[k].unsqueeze(0) for k in ["down1280", "down2048", "up", "d_in"]}


def compute_mean(dataset, indices, device) -> dict[str, torch.Tensor]:
    tensors = []
    for idx in indices:
        s = sample_to_batch(dataset[idx])
        tensors.append({k: v.float() for k, v in s.items()})
    mean = {}
    for key in ["down1280", "down2048", "up", "d_in"]:
        mean[key] = torch.stack([t[key] for t in tensors]).mean(dim=0)
    return {k: v.to(device) for k, v in mean.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    meta = {k: v.to(device) for k, v in make_metadata_tensors(dataset.specs).items()}

    target_full = sample_to_batch(dataset[args.idx])
    target_full = {k: v.to(device) for k, v in target_full.items()}

    mean_loras = compute_mean(dataset, [args.idx, args.other_idx], device)

    residual = {k: target_full[k] - mean_loras[k] for k in ["down1280", "down2048", "up"]}
    residual["d_in"] = target_full["d_in"]

    model = StyleLoRAAutoencoder(
        num_pairs=dataset.num_pairs, rank=dataset.rank,
        latent_dim=args.latent_dim, d_model=512, num_layers=4, num_heads=8,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    print(f"[residual] target={dataset.style_names[args.idx]} other={dataset.style_names[args.other_idx]} "
          f"num_pairs={dataset.num_pairs} lr={args.lr}", flush=True)

    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(residual, meta)
        metrics = reconstruction_losses(pred, residual, tensor_weight=1.0, delta_weight=0.0,
                                        cos_weight=0.0, rel_weight=0.0, norm_weight=0.0)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                pred_full = {k: pred[k] + mean_loras[k] for k in ["down1280", "down2048", "up"]}
                pred_full["d_in"] = pred.get("d_in", target_full["d_in"])
                full_metrics = reconstruction_losses(pred_full, target_full)
            line = (
                f"step={step:05d}/{args.max_steps} "
                f"mse={metrics['tensor_mse'].item():.8f} "
                f"full_cos={full_metrics['cos'].item():.4f} "
                f"full_rel={full_metrics['rel'].item():.4f} "
                f"full_norm={full_metrics['norm_ratio'].item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

    with torch.no_grad():
        pred = model(residual, meta)
        pred_full = {k: pred[k] + mean_loras[k] for k in ["down1280", "down2048", "up"]}
        pred_full["d_in"] = pred.get("d_in", target_full["d_in"])
        final = reconstruction_losses(pred_full, target_full)
        final_metrics = {k: float(v.item()) for k, v in final.items()}
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict()}, out / "checkpoint.pt")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
