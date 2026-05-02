#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import StyleLoRAAutoencoder, reconstruction_losses


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check LoRA AE reconstruction metrics.")
    p.add_argument("--checkpoint", default="/scratch/eyavuz21/lora_autoencoder/overfit1_debug/latest.pt")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--other_idx", type=int, default=1)
    p.add_argument("--output", default="/scratch/eyavuz21/lora_autoencoder/overfit1_debug/sanity_latest.json")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_to_batch(sample: dict) -> dict[str, torch.Tensor]:
    return {
        "down1280": sample["down1280"].unsqueeze(0),
        "down2048": sample["down2048"].unsqueeze(0),
        "up": sample["up"].unsqueeze(0),
        "d_in": sample["d_in"].unsqueeze(0),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu().item()) for k, v in metrics.items()}


def zero_pred_like(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "down1280": torch.zeros_like(batch["down1280"]),
        "down2048": torch.zeros_like(batch["down2048"]),
        "up": torch.zeros_like(batch["up"]),
    }


def pred_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "down1280": batch["down1280"],
        "down2048": batch["down2048"],
        "up": batch["up"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    target = move_batch(sample_to_batch(dataset[args.idx]), device)
    other = move_batch(sample_to_batch(dataset[args.other_idx]), device)
    meta = {k: v.to(device) for k, v in make_metadata_tensors(dataset.specs).items()}

    result = {
        "checkpoint": args.checkpoint,
        "target_style": dataset.style_names[args.idx],
        "other_style": dataset.style_names[args.other_idx],
        "identity": detach_metrics(reconstruction_losses(pred_from_batch(target), target)),
        "zero": detach_metrics(reconstruction_losses(zero_pred_like(target), target)),
        "shuffled": detach_metrics(reconstruction_losses(pred_from_batch(other), target)),
    }

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        model = StyleLoRAAutoencoder(
            num_pairs=dataset.num_pairs,
            rank=dataset.rank,
            latent_dim=cfg["latent_dim"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(target, meta)
            result["model"] = detach_metrics(reconstruction_losses(pred, target))
            result["checkpoint_step"] = int(payload.get("step", -1))
    else:
        result["model_error"] = f"checkpoint not found: {ckpt}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
