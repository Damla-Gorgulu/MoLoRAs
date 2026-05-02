#!/usr/bin/env python3
"""Mini v3 tensor-space synthesis training."""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="MoELoRA v3 mini synthesis training")
    p.add_argument("--zoo_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/zoo/bloras")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/cache_v3")
    p.add_argument("--manifest_path", default="/scratch/eyavuz21/lora_attention/mini_exact_v1/manifest.json")
    p.add_argument("--routing_checkpoint", default="/scratch/eyavuz21/lora_attention/mini_v3_clip_strong/latest.pt")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_attention/mini_v3_synth_clip")
    p.add_argument("--image_encoder", choices=["clip", "vae"], default="clip")
    p.add_argument("--rank_tokens", type=int, default=32)
    p.add_argument("--max_tensor_groups", type=int, default=16)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--query_layers", type=int, default=3)
    p.add_argument("--key_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--views_per_style", type=int, default=32)
    p.add_argument("--pool_size", type=int, default=4)
    p.add_argument("--exclude_gt", action="store_true")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cos_weight", type=float, default=1.0)
    p.add_argument("--mse_weight", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_routing_checkpoint(model: MoELoRAv3, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    q_state = {
        k: v for k, v in ckpt["query_encoder_state_dict"].items()
        if not k.startswith("_clip_model.") and not k.startswith("_vae.")
    }
    model.query_encoder.load_state_dict(q_state, strict=False)
    model.key_encoder.load_state_dict(ckpt["key_encoder_state_dict"], strict=False)


def product_loss_for_sample(model: MoELoRAv3, A, meta, gt_idx: int, device, args):
    losses = []
    cosines = []
    rel_errors = []
    pool_indices = meta["pool_indices"]

    for t_idx, (down_key, up_key) in enumerate(zip(meta["down_keys"], meta["up_keys"])):
        W_down = model.pool.get_stacked_tensors(pool_indices, down_key).to(device)[:, : args.rank_tokens, :]
        W_up = model.pool.get_stacked_tensors(pool_indices, up_key).to(device)[:, :, : args.rank_tokens]
        A_t = A[:, t_idx, : args.rank_tokens]

        synth_down = (A_t.unsqueeze(-1) * W_down).sum(dim=0)
        synth_up = (A_t.unsqueeze(1) * W_up).sum(dim=0)
        delta_synth = torch.matmul(synth_up.float(), synth_down.float())

        target = model.pool.get_style_tensors(gt_idx)
        target_down = target[down_key].to(device)[: args.rank_tokens, :]
        target_up = target[up_key].to(device)[:, : args.rank_tokens]
        delta_target = torch.matmul(target_up.float(), target_down.float())

        synth_flat = delta_synth.reshape(-1)
        target_flat = delta_target.reshape(-1)
        cos = F.cosine_similarity(synth_flat.unsqueeze(0), target_flat.unsqueeze(0)).mean()
        rel = (synth_flat - target_flat).norm() / (target_flat.norm() + 1e-8)
        mse = F.mse_loss(delta_synth, delta_target)
        losses.append(args.cos_weight * (1.0 - cos) + args.mse_weight * mse)
        cosines.append(cos.detach())
        rel_errors.append(rel.detach())

    return torch.stack(losses).mean(), torch.stack(cosines).mean(), torch.stack(rel_errors).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    pool = LoRAPool(args.zoo_dir, args.cache_dir, force_rebuild=False, device=str(device))
    model = MoELoRAv3(
        pool=pool,
        image_encoder=args.image_encoder,
        rank_tokens=args.rank_tokens,
        max_tensor_groups=args.max_tensor_groups,
        d_model=args.d_model,
        query_layers=args.query_layers,
        key_layers=args.key_layers,
        num_heads=args.num_heads,
    ).to(device)
    if args.routing_checkpoint:
        load_routing_checkpoint(model, args.routing_checkpoint)

    dataset = ExactExemplarStage1Dataset(
        pool=pool,
        manifest_path=args.manifest_path,
        min_pool_size=args.pool_size,
        max_pool_size=args.pool_size,
        views_per_style=args.views_per_style,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=exact_stage1_collate_fn)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=args.lr, weight_decay=1e-4)

    log_path = out / "train_log.txt"
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        losses = []
        cosines = []
        rels = []
        entropies = []
        for image, pool_indices, gt_idx in zip(batch["images"], batch["pool_indices"], batch["gt_indices"]):
            if args.exclude_gt:
                pool_indices = [i for i in pool_indices if i != gt_idx]
            A, meta = model(image, pool_indices, temperature=args.temperature)
            loss, cos, rel = product_loss_for_sample(model, A, meta, gt_idx, device, args)
            losses.append(loss)
            cosines.append(cos)
            rels.append(rel)
            entropies.append(model.attention_entropy(A).detach())

        loss = torch.stack(losses).mean()
        loss.backward()
        for p in model.trainable_parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            line = (
                f"step={step:5d}/{args.max_steps} loss={loss.item():.6f} "
                f"cos={torch.stack(cosines).mean().item():.4f} "
                f"rel={torch.stack(rels).mean().item():.4f} "
                f"entropy={torch.stack(entropies).mean().item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

        if step % args.save_every == 0 or step == args.max_steps:
            payload = {
                "step": step,
                "version": "v3-synthesis-mini",
                "query_encoder_state_dict": {k: v for k, v in model.query_encoder.state_dict().items() if not k.startswith("_clip_model.") and not k.startswith("_vae.")},
                "key_encoder_state_dict": model.key_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            }
            ckpt_dir = out / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, ckpt_dir / "checkpoint.pt")
            torch.save(payload, out / "latest.pt")
            print(f"[save] {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
