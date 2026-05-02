#!/usr/bin/env python3
"""Diagnostic: 1-LoRA residual training with 16-LoRA mean template.
Isolates whether the 16-style failure is due to tiny residual signal or multi-style capacity."""
from __future__ import annotations

import argparse, json, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import StyleLoRAAutoencoder, reconstruction_losses


def parse_args():
    p = argparse.ArgumentParser(description="1-LoRA residual with N-LoRA mean template.")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--mean_limit", type=int, default=16)
    p.add_argument("--train_idx", type=int, default=0)
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder/residual1_on16mean")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.mean_limit)
    meta = {k: v.to(device) for k, v in make_metadata_tensors(dataset.specs).items()}

    # Mean template from all loaded LoRAs
    s0 = dataset[0]
    sum_d1280 = torch.zeros(1, s0["down1280"].shape[0], s0["down1280"].shape[1], 1280)
    sum_d2048 = torch.zeros(1, s0["down2048"].shape[0], s0["down2048"].shape[1], 2048)
    sum_up = torch.zeros(1, s0["up"].shape[0], s0["up"].shape[1], 1280)
    for i in range(len(dataset)):
        s = dataset[i]
        sum_d1280 += s["down1280"].unsqueeze(0)
        sum_d2048 += s["down2048"].unsqueeze(0)
        sum_up += s["up"].unsqueeze(0)
    mean = {
        "down1280": (sum_d1280 / len(dataset)).to(device),
        "down2048": (sum_d2048 / len(dataset)).to(device),
        "up": (sum_up / len(dataset)).to(device),
    }

    target = dataset[args.train_idx]
    target_dev = {k: target[k].unsqueeze(0).to(device) for k in ["down1280", "down2048", "up"]}
    d_in = target["d_in"].unsqueeze(0).to(device)
    target_dev["d_in"] = d_in

    residual = {
        "down1280": target_dev["down1280"] - mean["down1280"],
        "down2048": target_dev["down2048"] - mean["down2048"],
        "up": target_dev["up"] - mean["up"],
        "d_in": d_in,
    }

    # Mean baseline
    with torch.no_grad():
        mp = {k: mean[k].clone() for k in ["down1280", "down2048", "up"]}
        mp["d_in"] = d_in
        base = reconstruction_losses(mp, target_dev)
        print(f"[mean_baseline] cos={base['cos'].item():.4f} rel={base['rel'].item():.4f} mse={base['tensor_mse'].item():.8f}", flush=True)

    model = StyleLoRAAutoencoder(
        num_pairs=s0["down1280"].shape[0], rank=s0["down1280"].shape[1],
        latent_dim=41472, d_model=512, num_layers=4, num_heads=8,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = out / "train_log.txt"

    print(f"[residual1_on{args.mean_limit}] target={dataset.style_names[args.train_idx]} pairs={s0['down1280'].shape[0]} lr={args.lr}", flush=True)

    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        pred_res = model(residual, meta)
        metrics = reconstruction_losses(pred_res, residual, tensor_weight=1.0, delta_weight=0.0,
                                        cos_weight=0.0, rel_weight=0.0, norm_weight=0.0)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step == 1 or step % 10 == 0:
            with torch.no_grad():
                pf = {k: pred_res[k].detach() + mean[k] for k in ["down1280", "down2048", "up"]}
                pf["d_in"] = d_in
                fm = reconstruction_losses(pf, target_dev)
            line = (
                f"s={step:05d}/{args.max_steps} "
                f"r_mse={metrics['tensor_mse'].item():.8f} "
                f"r_cos={metrics['cos'].item():.4f} "
                f"r_rel={metrics['rel'].item():.4f} "
                f"r_nr={metrics['norm_ratio'].item():.4f} "
                f"fcos={fm['cos'].item():.4f} "
                f"frel={fm['rel'].item():.4f} "
                f"fnr={fm['norm_ratio'].item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

        if step % 1000 == 0:
            ckpt = out / f"checkpoint-{step}"
            ckpt.mkdir(exist_ok=True)
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt / "checkpoint.pt")

    with torch.no_grad():
        pr = model(residual, meta)
        pf = {k: pr[k].detach() + mean[k] for k in ["down1280", "down2048", "up"]}
        pf["d_in"] = d_in
        final = reconstruction_losses(pf, target_dev)
        final_metrics = {k: float(v.item()) for k, v in final.items()}
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict()}, out / "latest.pt")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
