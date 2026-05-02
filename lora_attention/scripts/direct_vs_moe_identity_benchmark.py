#!/usr/bin/env python3
"""Tiny direct-vs-MoE benchmark for query-style identity checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.utils.lora_inject import inject_lora, unload_lora


CASES = [
    {
        "label": "Baroque",
        "gt_expert": "style_0000_Baroque",
        "query": "/home/eyavuz21/datasets/wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg",
    },
    {
        "label": "Cubism",
        "gt_expert": "style_0003_Cubism",
        "query": "/home/eyavuz21/datasets/wikiart/Cubism/adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg",
    },
    {
        "label": "Fauvism",
        "gt_expert": "style_0148_Fauvism",
        "query": "/home/eyavuz21/datasets/wikiart/Fauvism/abraham-manievich_artist-s-wife-1937.jpg",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/scratch/eyavuz21/lora_attention/mini_generalization_v2/stage1_train/latest.pt")
    p.add_argument("--output_dir", default="/scratch/eyavuz21/lora_attention/diagnostics/direct_vs_moe_identity_benchmark")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--style_alpha", type=float, default=1.0)
    p.add_argument("--norm_target", type=float, default=32.0)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    return p.parse_args()


def image_grid(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    thumb_w, thumb_h = 320, 320
    label_h = 34
    grid = Image.new("RGB", (thumb_w * len(items), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate(items):
        tile = img.convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        x = i * thumb_w
        grid.paste(tile, (x, label_h))
        draw.text((x + 8, 9), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def generate(pipe, prompt: str, seed: int, steps: int, guidance: float) -> Image.Image:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
    ).images[0]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = [
        ("dog_neutral", lambda label: "A dog"),
        ("dog_style_word", lambda label: f"A dog in {label} style"),
    ]

    print(f"[bench] device={device}")
    print(f"[bench] checkpoint={args.checkpoint}")
    print(f"[bench] output_dir={out_root}")

    pool = LoRAPool(zoo_dir=args.zoo_dir, cache_dir=args.cache_dir)
    model = MoELoRAv2(pool=pool).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.encoder.eval()
    print(f"[bench] loaded checkpoint step={ckpt.get('step', '?')}")

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)

    summary = {
        "checkpoint": args.checkpoint,
        "temperature": args.temperature,
        "style_alpha": args.style_alpha,
        "norm_target": args.norm_target,
        "prompts": [name for name, _ in prompts],
        "cases": [],
    }

    for case in CASES:
        label = case["label"]
        gt_expert = case["gt_expert"]
        query_path = Path(case["query"])
        case_dir = out_root / label
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[bench] case={label} gt={gt_expert}")
        style_image = Image.open(query_path).convert("RGB")
        q = model.encode_image(style_image, device)
        with torch.no_grad():
            attn, synth_top1 = model.forward(
                q,
                list(range(pool.num_experts)),
                temperature=args.temperature,
                top_k=1,
                product_space=True,
            )

        avg_attn = attn.mean(dim=(1, 2))
        top_vals, top_idxs = avg_attn.topk(5)
        top_names = [pool.style_names[i] for i in top_idxs.tolist()]
        gt_idx = next(i for i, name in enumerate(pool.style_names) if gt_expert in name)
        gt_rank = int((avg_attn > avg_attn[gt_idx]).sum().item() + 1)
        entropy = model.attention_entropy(attn).item()
        synth_cpu = {k: v.detach().cpu() for k, v in synth_top1.items()}
        synth_norm = sum(v.norm().item() ** 2 for v in synth_cpu.values()) ** 0.5
        scale = args.norm_target / synth_norm if synth_norm > 1e-6 else 1.0

        print(f"[bench] gt_rank={gt_rank} entropy={entropy:.4f}/{math.log(pool.num_experts):.4f}")
        print("[bench] top5=" + ", ".join(f"{n}:{v.item():.4f}" for n, v in zip(top_names, top_vals)))
        print(f"[bench] synth_norm={synth_norm:.4f} norm_scale={scale:.4f}")

        ref_lora = Path(args.zoo_dir) / gt_expert / "pytorch_lora_weights.safetensors"
        real_sd = {
            k: v * args.style_alpha
            for k, v in load_file(str(ref_lora)).items()
            if "up_blocks.0.attentions.1" in k
        }

        case_summary = {
            "label": label,
            "gt_expert": gt_expert,
            "query": str(query_path),
            "gt_rank": gt_rank,
            "entropy": entropy,
            "top5": [{"name": n, "attention": float(v.item())} for n, v in zip(top_names, top_vals)],
            "synth_norm": synth_norm,
            "norm_match_scale": scale,
            "prompts": [],
        }

        for prompt_name, prompt_fn in prompts:
            prompt = prompt_fn(label)
            prompt_dir = case_dir / prompt_name
            prompt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[bench] prompt[{prompt_name}]={prompt}")

            base = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            base.save(prompt_dir / "base.png")

            pipe.load_lora_into_unet(real_sd, None, pipe.unet)
            ref = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            ref.save(prompt_dir / "reference_blora.png")
            unload_lora(pipe)

            inject_lora(pipe, synth_cpu, style_alpha=args.style_alpha)
            synth = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            synth.save(prompt_dir / "synth_top1_alpha1.png")
            unload_lora(pipe)

            inject_lora(pipe, synth_cpu, style_alpha=args.style_alpha * scale)
            synth_nm = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            synth_nm.save(prompt_dir / "synth_top1_norm_match.png")
            unload_lora(pipe)

            image_grid(
                [
                    ("query", style_image),
                    ("base", base),
                    ("reference", ref),
                    ("synth top1", synth),
                    ("synth norm", synth_nm),
                ],
                prompt_dir / "comparison_grid.png",
            )

            case_summary["prompts"].append(
                {
                    "name": prompt_name,
                    "prompt": prompt,
                    "grid": str(prompt_dir / "comparison_grid.png"),
                }
            )

        summary["cases"].append(case_summary)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[bench] complete: {out_root}")
    print(f"[bench] summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
