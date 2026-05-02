#!/usr/bin/env python3
"""Metadata-only decoder baseline: no encoder, no latent bottleneck.
Decoder maps (pair, rank) identity tokens → LoRA slices directly.
Tests whether the decoder + heads can memorize 1 LoRA from metadata alone."""
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


class MetaDecoderBaseline(nn.Module):
    """Decoder-only model: maps metadata tokens to LoRA tensor slices."""

    def __init__(self, num_pairs: int, rank: int, d_model: int = 512, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_pairs = num_pairs
        self.rank = rank
        self.d_model = d_model

        self.block_emb = nn.Embedding(16, d_model)
        self.attn_emb = nn.Embedding(2, d_model)
        self.matrix_emb = nn.Embedding(4, d_model)
        self.rank_emb = nn.Embedding(rank, d_model)
        self.pair_emb = nn.Embedding(num_pairs, d_model)

        self.mem_tokens = nn.Parameter(torch.zeros(1, 16, d_model))
        nn.init.normal_(self.mem_tokens, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.down1280_head = nn.Linear(d_model, 1280)
        self.down2048_head = nn.Linear(d_model, 2048)
        self.up_head = nn.Linear(d_model, 1280)

        for head in [self.down1280_head, self.down2048_head, self.up_head]:
            nn.init.normal_(head.weight, std=1e-5)
            nn.init.zeros_(head.bias)

    def _metadata_queries(self, meta: dict[str, torch.Tensor], device: torch.device, bsz: int) -> torch.Tensor:
        pair_ids = torch.arange(self.num_pairs, device=device)
        rank_ids = torch.arange(self.rank, device=device)
        base = (
            self.pair_emb(pair_ids)[:, None, :]
            + self.rank_emb(rank_ids)[None, :, :]
            + self.block_emb(meta["block_idx"].to(device))[:, None, :]
            + self.attn_emb(meta["attn_idx"].to(device))[:, None, :]
            + self.matrix_emb(meta["matrix_idx"].to(device))[:, None, :]
        )
        return base.reshape(1, self.num_pairs * self.rank, self.d_model).expand(bsz, -1, -1)

    def forward(self, meta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bsz = 1
        device = self.mem_tokens.device
        queries = self._metadata_queries(meta, device, bsz)
        memory = self.mem_tokens.expand(bsz, -1, -1)
        h = self.decoder(queries, memory)
        h = self.out_norm(h).reshape(bsz, self.num_pairs, self.rank, self.d_model)
        return {
            "down1280": self.down1280_head(h),
            "down2048": self.down2048_head(h),
            "up": self.up_head(h),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metadata-only decoder baseline.")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/meta_decoder_baseline")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
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

    model = MetaDecoderBaseline(
        num_pairs=dataset.num_pairs,
        rank=dataset.rank,
        d_model=512,
        num_layers=4,
        num_heads=8,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    print(f"[meta_decoder] style={sample['style_name']} num_pairs={dataset.num_pairs} rank={dataset.rank} lr={args.lr}", flush=True)

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
