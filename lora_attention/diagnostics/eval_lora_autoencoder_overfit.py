#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.diagnostics.oracle_copy_ablation import build_pipeline, generate, pixel_mae
from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import StyleLoRAAutoencoder, reconstruction_losses
from lora_attention.utils.lora_inject import unload_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate overfit LoRA autoencoder reconstructions.")
    p.add_argument("--checkpoint", default="/scratch/eyavuz21/lora_autoencoder/overfit16_v1/latest.pt")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_autoencoder_eval/overfit16_v1")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--prompt", default="A [v] dog")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--style_alpha", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    return p.parse_args()


def resolve_device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if dtype_str == "fp16" else torch.float32


def denormalize_sample(sample: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor]:
    return {
        "down1280": sample["down1280"].unsqueeze(0),
        "down2048": sample["down2048"].unsqueeze(0),
        "up": sample["up"].unsqueeze(0),
        "d_in": sample["d_in"].unsqueeze(0),
    }


def build_reconstructed_state_dict(
    original_sd: dict[str, torch.Tensor],
    specs,
    recon: dict[str, torch.Tensor],
    d_in: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in original_sd.items()}
    down1280 = recon["down1280"][0].cpu()
    down2048 = recon["down2048"][0].cpu()
    up = recon["up"][0].cpu()
    d_in = d_in.cpu()

    for i, spec in enumerate(specs):
        down = down2048[i] if int(d_in[i].item()) == 2048 else down1280[i]
        out[spec.down_key] = down.to(dtype=original_sd[spec.down_key].dtype).contiguous()
        out[spec.up_key] = up[i].transpose(0, 1).to(dtype=original_sd[spec.up_key].dtype).contiguous()
    return out


def state_metrics(real_sd: dict[str, torch.Tensor], recon_sd: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = sorted(real_sd.keys())
    flat_real = []
    flat_recon = []
    mean_abs = []
    max_abs = []
    for key in keys:
        r = real_sd[key].float().reshape(-1)
        q = recon_sd[key].float().reshape(-1)
        flat_real.append(r)
        flat_recon.append(q)
        diff = (r - q).abs()
        mean_abs.append(float(diff.mean().item()))
        max_abs.append(float(diff.max().item()))
    real = torch.cat(flat_real)
    recon = torch.cat(flat_recon)
    rel = float((recon - real).norm().item() / (real.norm().item() + 1e-8))
    cos = float(torch.nn.functional.cosine_similarity(real.unsqueeze(0), recon.unsqueeze(0)).item())
    return {
        "state_rel_l2": rel,
        "state_cosine": cos,
        "mean_tensor_abs_diff": float(np.mean(mean_abs)),
        "max_tensor_abs_diff": float(np.max(max_abs)),
    }


def save_grid(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    tile_w, tile_h = 320, 320
    label_h = 34
    canvas = Image.new("RGB", (tile_w * len(items), tile_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        x = i * tile_w
        canvas.paste(image.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS), (x, label_h))
        draw.text((x + 8, 9), label, fill=(0, 0, 0))
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = StyleLoRAAutoencoderDataset(args.zoo_dir, limit=args.limit)
    meta = make_metadata_tensors(dataset.specs)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]

    model = StyleLoRAAutoencoder(
        num_pairs=dataset.num_pairs,
        rank=dataset.rank,
        latent_dim=cfg["latent_dim"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    model = model.to(device)
    meta_dev = {k: v.to(device) for k, v in meta.items()}

    pipe = build_pipeline(args, device=device, dtype=dtype)
    summary = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "seed": args.seed,
        "indices": args.indices,
        "results": [],
    }

    for idx in args.indices:
        sample = dataset[idx]
        batch = denormalize_sample(sample)
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            pred = model(batch_dev, meta_dev)
            losses = reconstruction_losses(pred, batch_dev)

        weights_path = dataset.paths[idx]
        real_sd = load_file(str(weights_path), device="cpu")
        recon_sd = build_reconstructed_state_dict(real_sd, dataset.specs, pred, batch["d_in"][0])

        style_name = sample["style_name"]
        style_dir = out_dir / style_name
        style_dir.mkdir(parents=True, exist_ok=True)
        recon_path = style_dir / "reconstructed_style_lora.safetensors"
        save_file(recon_sd, str(recon_path))

        unload_lora(pipe)
        direct_style = {k: v * args.style_alpha for k, v in real_sd.items() if "unet.up_blocks.0.attentions.1" in k}
        pipe.load_lora_into_unet(direct_style, None, pipe.unet)
        img_real = generate(pipe, args.prompt, args.seed, args.num_inference_steps, args.guidance_scale)
        img_real.save(style_dir / "original.png")
        unload_lora(pipe)

        recon_style = {k: v * args.style_alpha for k, v in recon_sd.items() if "unet.up_blocks.0.attentions.1" in k}
        pipe.load_lora_into_unet(recon_style, None, pipe.unet)
        img_recon = generate(pipe, args.prompt, args.seed, args.num_inference_steps, args.guidance_scale)
        img_recon.save(style_dir / "reconstructed.png")
        unload_lora(pipe)

        save_grid([
            ("original_lora", img_real),
            ("reconstructed_lora", img_recon),
        ], style_dir / "comparison_grid.png")

        result = {
            "index": idx,
            "style_name": style_name,
            "weights_path": str(weights_path),
            "recon_path": str(recon_path),
            "loss": float(losses["loss"].item()),
            "tensor_mse": float(losses["tensor_mse"].item()),
            "delta_mse": float(losses["delta_mse"].item()),
            "cos": float(losses["cos"].item()),
            "rel": float(losses["rel"].item()),
            "pixel_mae": float(pixel_mae(img_real, img_recon)),
        }
        result.update(state_metrics(real_sd, recon_sd))
        summary["results"].append(result)

    unload_lora(pipe)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
