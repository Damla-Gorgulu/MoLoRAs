#!/usr/bin/env python3
"""Direct-parameter baseline: learn raw LoRA tensors without any model.
Sanity check: can Adam + tensor MSE memorize 1 fixed LoRA target?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset

LOSS_KEYS = ["down1280", "down2048", "up"]
D_IN_NAME = "d_in"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct parameter baseline for LoRA AE.")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/direct_param_baseline")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_to_batch(sample: dict) -> dict[str, torch.Tensor]:
    return {k: sample[k].unsqueeze(0) for k in ["down1280", "down2048", "up", "d_in"]}


def metrics(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float]:
    d_in = target[D_IN_NAME]
    mask2048 = (d_in == 2048).float().view(d_in.shape[0], d_in.shape[1], 1, 1)
    mask1280 = 1.0 - mask2048

    tensor_mse = (
        F.mse_loss(pred["up"], target["up"])
        + (F.mse_loss(pred["down1280"], target["down1280"], reduction="none") * mask1280).sum()
          / (mask1280.sum() * pred["down1280"].shape[-2] * pred["down1280"].shape[-1]).clamp_min(1.0)
        + (F.mse_loss(pred["down2048"], target["down2048"], reduction="none") * mask2048).sum()
          / (mask2048.sum() * pred["down2048"].shape[-2] * pred["down2048"].shape[-1]).clamp_min(1.0)
    )

    flat_pred = torch.cat([
        pred["up"].flatten(1),
        (pred["down1280"] * mask1280).flatten(1),
        (pred["down2048"] * mask2048).flatten(1),
    ], dim=1)
    flat_tgt = torch.cat([
        target["up"].flatten(1),
        (target["down1280"] * mask1280).flatten(1),
        (target["down2048"] * mask2048).flatten(1),
    ], dim=1)
    cos = F.cosine_similarity(flat_pred, flat_tgt, dim=1).mean()
    rel = (flat_pred - flat_tgt).norm(dim=1).mean() / (flat_tgt.norm(dim=1).mean() + 1e-8)
    nr = flat_pred.norm(dim=1) / (flat_tgt.norm(dim=1) + 1e-8)
    return {
        "tensor_mse": float(tensor_mse.item()),
        "cos": float(cos.item()),
        "rel": float(rel.item()),
        "norm_ratio": float(nr.mean().item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    sample = dataset[args.idx]
    target = sample_to_batch(sample)
    target = {k: v.to(device) for k, v in target.items()}

    params = {
        "down1280": torch.nn.Parameter(torch.zeros_like(target["down1280"])),
        "down2048": torch.nn.Parameter(torch.zeros_like(target["down2048"])),
        "up": torch.nn.Parameter(torch.zeros_like(target["up"])),
    }

    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0)
    log_path = out / "train_log.txt"

    print(f"[direct_param] style={sample['style_name']} num_pairs={dataset.num_pairs} rank={dataset.rank} lr={args.lr}", flush=True)

    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = {k: p for k, p in params.items()}
        m = metrics(pred, target)
        m["loss"] = m["tensor_mse"]
        loss = torch.tensor(m["loss"], device=device, requires_grad=True) * 0 + m["tensor_mse"]
        # Actually compute proper loss
        d_in = target["d_in"]
        mask2048 = (d_in == 2048).float().view(d_in.shape[0], d_in.shape[1], 1, 1)
        mask1280 = 1.0 - mask2048
        loss = (
            F.mse_loss(params["up"], target["up"])
            + (F.mse_loss(params["down1280"], target["down1280"], reduction="none") * mask1280).sum()
              / (mask1280.sum() * params["down1280"].shape[-2] * params["down1280"].shape[-1]).clamp_min(1.0)
            + (F.mse_loss(params["down2048"], target["down2048"], reduction="none") * mask2048).sum()
              / (mask2048.sum() * params["down2048"].shape[-2] * params["down2048"].shape[-1]).clamp_min(1.0)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            m_rec = metrics({k: v.detach() for k, v in params.items()}, target)
            line = (
                f"step={step:05d}/{args.max_steps} "
                f"mse={m_rec['tensor_mse']:.8f} "
                f"cos={m_rec['cos']:.4f} "
                f"rel={m_rec['rel']:.4f} "
                f"norm_ratio={m_rec['norm_ratio']:.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

    final_metrics = metrics({k: v.detach() for k, v in params.items()}, target)
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
