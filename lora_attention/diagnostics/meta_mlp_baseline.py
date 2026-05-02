#!/usr/bin/env python3
"""Metadata-MLP baseline: no transformer, no attention.
Each rank token gets metadata embeddings → MLP → LoRA slice directly.
Tests whether a simple MLP per token can memorize 1 LoRA."""
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


class MetaMLPBaseline(nn.Module):
    """Simple per-token MLP: metadata embeddings → output slice. No transformer."""

    def __init__(self, num_pairs: int, rank: int, d_model: int = 512):
        super().__init__()
        self.num_pairs = num_pairs
        self.rank = rank

        self.block_emb = nn.Embedding(16, d_model)
        self.attn_emb = nn.Embedding(2, d_model)
        self.matrix_emb = nn.Embedding(4, d_model)
        self.rank_emb = nn.Embedding(rank, d_model)
        self.pair_emb = nn.Embedding(num_pairs, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.down1280_head = nn.Linear(d_model, 1280)
        self.down2048_head = nn.Linear(d_model, 2048)
        self.up_head = nn.Linear(d_model, 1280)

        for head in [self.down1280_head, self.down2048_head, self.up_head]:
            nn.init.normal_(head.weight, std=1e-5)
            nn.init.zeros_(head.bias)

    def forward(self, meta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.pair_emb.weight.device
        pair_ids = torch.arange(self.num_pairs, device=device)
        rank_ids = torch.arange(self.rank, device=device)

        emb = (
            self.pair_emb(pair_ids)[:, None, :]                    # [80, 1, 512]
            + self.rank_emb(rank_ids)[None, :, :]                  # [1, 64, 512]
            + self.block_emb(meta["block_idx"].to(device))[:, None, :]  # [80, 1, 512]
            + self.attn_emb(meta["attn_idx"].to(device))[:, None, :]
            + self.matrix_emb(meta["matrix_idx"].to(device))[:, None, :]
        )  # [80, 64, 512]

        h = self.mlp(emb)                                          # [80, 64, 512]
        h = h.unsqueeze(0)                                         # [1, 80, 64, 512]
        return {
            "down1280": self.down1280_head(h),
            "down2048": self.down2048_head(h),
            "up": self.up_head(h),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metadata-MLP baseline (no transformer).")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/meta_mlp_baseline")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_to_batch(sample: dict) -> dict[str, torch.Tensor]:
    return {k: (sample[k].unsqueeze(0) if k != "d_in" else sample[k].unsqueeze(0))
            for k in ["down1280", "down2048", "up", "d_in"]}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    sample = dataset[args.idx]
    target = sample_to_batch(sample)
    target = {k: v.to(device) for k, v in target.items()}
    meta = {k: v.to(device) for k, v in make_metadata_tensors(dataset.specs).items()}

    model = MetaMLPBaseline(num_pairs=dataset.num_pairs, rank=dataset.rank, d_model=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    print(f"[meta_mlp] style={sample['style_name']} num_pairs={dataset.num_pairs} rank={dataset.rank} lr={args.lr}", flush=True)

    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(meta)
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
        pred = model(meta)
        final = reconstruction_losses(pred, target)
        final_metrics = {k: float(v.item()) for k, v in final.items()}
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict()}, out / "checkpoint.pt")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
