#!/usr/bin/env python3
"""Generate direct-vs-v3-synth images for mini synthesis checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v3 import MoELoRAv3
from lora_attention.utils.lora_inject import inject_lora, unload_lora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt", default="A dog")
    p.add_argument("--style_alpha", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    return p.parse_args()


def load_model(ckpt, pool, device):
    cfg = ckpt["config"]
    model = MoELoRAv3(
        pool=pool,
        image_encoder=cfg["image_encoder"],
        rank_tokens=cfg["rank_tokens"],
        max_tensor_groups=cfg["max_tensor_groups"],
        d_model=cfg["d_model"],
        query_layers=cfg["query_layers"],
        key_layers=cfg["key_layers"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.query_encoder.load_state_dict(ckpt["query_encoder_state_dict"], strict=False)
    model.key_encoder.load_state_dict(ckpt["key_encoder_state_dict"], strict=False)
    model.eval()
    return model


def synth_state(model, A, meta, device):
    synth = {}
    pool_indices = meta["pool_indices"]
    for t_idx, (down_key, up_key) in enumerate(zip(meta["down_keys"], meta["up_keys"])):
        W_down = model.pool.get_stacked_tensors(pool_indices, down_key).to(device)[:, : model.rank_tokens, :]
        W_up = model.pool.get_stacked_tensors(pool_indices, up_key).to(device)[:, :, : model.rank_tokens]
        A_t = A[:, t_idx, : model.rank_tokens]
        synth[down_key] = (A_t.unsqueeze(-1) * W_down).sum(dim=0)
        synth[up_key] = (A_t.unsqueeze(1) * W_up).sum(dim=0)
    return synth


def generate(pipe, prompt, seed, steps, guidance):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]


def grid(items, path):
    tile = 320
    label_h = 34
    canvas = Image.new("RGB", (tile * len(items), tile + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(items):
        x = i * tile
        canvas.paste(img.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, label_h))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
    canvas.save(path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = LoRAPool(cfg["zoo_dir"], cfg["cache_dir"], force_rebuild=False, device=str(device))
    model = load_model(ckpt, pool, device)

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model, vae=vae, torch_dtype=dtype).to(device)

    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest["styles"]:
        expert = entry["expert_name"]
        gt_idx = pool.index_of(expert)
        image = Image.open(entry["source_image"]).convert("RGB")
        pool_indices = list(range(pool.num_experts))

        vanilla = generate(pipe, args.prompt, args.seed, args.steps, args.guidance)
        direct_sd = pool.get_style_tensors(gt_idx)
        inject_lora(pipe, direct_sd, style_alpha=args.style_alpha)
        direct = generate(pipe, args.prompt, args.seed, args.steps, args.guidance)
        unload_lora(pipe)

        with torch.no_grad():
            A, meta = model(image, pool_indices, temperature=1.0)
            synth_sd = synth_state(model, A, meta, device)
        inject_lora(pipe, synth_sd, style_alpha=args.style_alpha)
        synth = generate(pipe, args.prompt, args.seed, args.steps, args.guidance)
        unload_lora(pipe)

        grid_path = out / f"{expert}_grid.png"
        grid([("query", image), ("vanilla", vanilla), ("direct", direct), ("v3_synth", synth)], grid_path)
        rows.append({"expert": expert, "grid": str(grid_path), "top1": pool.style_names[pool_indices[int(A.mean(dim=(1,2)).argmax().item())]]})

    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
