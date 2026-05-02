#!/usr/bin/env python3
"""Lookup-table baseline: each (pair, rank) gets a unique embedding → MLP → output slice.
No transformer, no shared metadata. Pure per-token memorization.
If this can't reach cos>0.95, the output head parameterization is the bottleneck."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import reconstruction_losses


class LookupBaseline(nn.Module):
    """One embedding per (pair, rank) token, no shared structure."""

    def __init__(self, num_pairs: int, rank: int, emb_dim: int = 256):
        super().__init__()
        self.num_pairs = num_pairs
        self.rank = rank
        self.num_tokens = num_pairs * rank

        self.token_emb = nn.Embedding(self.num_tokens, emb_dim)
        nn.init.normal_(self.token_emb.weight, std=0.02)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.down1280_head = nn.Linear(emb_dim, 1280)
        self.down2048_head = nn.Linear(emb_dim, 2048)
        self.up_head = nn.Linear(emb_dim, 1280)

        for head in [self.down1280_head, self.down2048_head, self.up_head]:
            nn.init.normal_(head.weight, std=1e-5)
            nn.init.zeros_(head.bias)

    def forward(self) -> dict[str, torch.Tensor]:
        device = self.token_emb.weight.device
        ids = torch.arange(self.num_tokens, device=device)
        h = self.token_emb(ids)                                  # [5120, 256]
        h = self.mlp(h)                                           # [5120, 256]
        h = h.reshape(1, self.num_pairs, self.rank, 256)         # [1, 80, 64, 256]
        return {
            "down1280": self.down1280_head(h),
            "down2048": self.down2048_head(h),
            "up": self.up_head(h),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lookup-table baseline.")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/lookup_baseline")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_to_batch(sample: dict) -> dict[str, torch.Tensor]:
    return {k: (sample[k].unsqueeze(0)) for k in ["down1280", "down2048", "up", "d_in"]}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    sample = dataset[args.idx]
    target = sample_to_batch(sample)
    target = {k: v.to(device) for k, v in target.items()}

    model = LookupBaseline(num_pairs=dataset.num_pairs, rank=dataset.rank, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[lookup] style={sample['style_name']} num_pairs={dataset.num_pairs} rank={dataset.rank} "
          f"params={total_params} lr={args.lr}", flush=True)

    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model()
        metrics = reconstruction_losses(pred, target, tensor_weight=1.0, delta_weight=0.0,
                                        cos_weight=0.0, rel_weight=0.0, norm_weight=0.0)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            line = (
                f"step={step:05d}/{args.max_steps} "
                f"mse={metrics['tensor_mse'].item():.8f} "
                f"cos={metrics['cos'].item():.4f} "
                f"rel={metrics['rel'].item():.4f} "
                f"norm_ratio={metrics['norm_ratio'].item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

    with torch.no_grad():
        pred = model()
        final = reconstruction_losses(pred, target)
        final_metrics = {k: float(v.item()) for k, v in final.items()}
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict()}, out / "checkpoint.pt")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
