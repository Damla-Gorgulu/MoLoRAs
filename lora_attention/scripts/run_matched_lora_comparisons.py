#!/usr/bin/env python3
"""Generate side-by-side comparisons for matched numeric LoRA IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/scratch/eyavuz21/lora_zoo/_trained_loras")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--id", dest="one_id", default=None)
    p.add_argument("--ids_file", default=None)
    p.add_argument("--prompt", default="A detailed painting of a vase on a table")
    p.add_argument("--manifest", default=None)
    p.add_argument("--content", default=None)
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    return p.parse_args()


def discover_ids(root: Path) -> list[str]:
    blora = {p.name for p in (root / "blora").iterdir() if (p / "pytorch_lora_weights.safetensors").exists()}
    unz = {
        p.name
        for p in (root / "unziplora").iterdir()
        if (p / "model_style" / "pytorch_lora_weights.safetensors").exists()
    }
    return sorted(blora & unz)


def grid(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    w, h, label_h = 320, 320, 34
    canvas = Image.new("RGB", (w * len(items), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(items):
        x = idx * w
        canvas.paste(img.convert("RGB").resize((w, h), Image.LANCZOS), (x, label_h))
        draw.text((x + 8, 9), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate(pipe, prompt: str, negative_prompt: str | None, seed: int, steps: int, guidance: float) -> Image.Image:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        cross_attention_kwargs={"scale": 1.0},
    ).images[0]


def unload(pipe) -> None:
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
    if hasattr(pipe, "unload_lora"):
        pipe.unload_lora()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = {}
    if args.manifest:
        manifest_rows = {row["image_id"]: row for row in json.loads(Path(args.manifest).read_text())}

    if args.one_id:
        ids = [args.one_id]
    elif args.ids_file:
        ids = [line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()]
    else:
        ids = discover_ids(root)

    print(f"[matched] root={root}")
    print(f"[matched] output_dir={out_root}")
    print(f"[matched] ids={','.join(ids)}")

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda")

    rows = []
    for id_ in ids:
        id_dir = out_root / id_
        id_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[matched] id={id_}")

        source_path = root.parent / "_dataset_" / f"{id_}.jpg"
        paths = {
            "blora": root / "blora" / id_ / "pytorch_lora_weights.safetensors",
            "unz_style": root / "unziplora" / id_ / "model_style" / "pytorch_lora_weights.safetensors",
            "unz_content": root / "unziplora" / id_ / "model_content" / "pytorch_lora_weights.safetensors",
        }
        prompt = args.prompt
        if args.content is not None:
            style_token = manifest_rows.get(id_, {}).get("image_id", id_)
            prompt = f"A {args.content} in {style_token} style"
        print(f"[matched] prompt={prompt}")

        images: list[tuple[str, Image.Image]] = []
        if source_path.exists():
            source = Image.open(source_path).convert("RGB")
            images.append(("source", source))

        base = generate(pipe, prompt, args.negative_prompt, args.seed, args.num_inference_steps, args.guidance_scale)
        base.save(id_dir / "base.png")
        images.append(("base", base))

        status = {
            "id": id_,
            "grid": str(id_dir / "comparison_grid.png"),
            "source": str(source_path),
            "paths": {k: str(v) for k, v in paths.items()},
            "ok": [],
        }
        for label, path in paths.items():
            if not path.exists():
                status.setdefault("missing", []).append(label)
                continue
            try:
                pipe.load_lora_weights(str(path.parent))
                img = generate(pipe, prompt, args.negative_prompt, args.seed, args.num_inference_steps, args.guidance_scale)
                img.save(id_dir / f"{label}.png")
                images.append((label, img))
                status["ok"].append(label)
            except Exception as exc:
                print(f"[matched] failed {id_}/{label}: {exc}")
                status.setdefault("failed", {})[label] = str(exc)
            finally:
                unload(pipe)

        grid(images, id_dir / "comparison_grid.png")
        rows.append(status)

    summary = {
        "root": str(root),
        "output_dir": str(out_root),
        "prompt": args.prompt,
        "manifest": args.manifest,
        "content": args.content,
        "seed": args.seed,
        "ids": ids,
        "results": rows,
    }
    summary_name = f"summary_{ids[0]}.json" if len(ids) == 1 else "summary.json"
    (out_root / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[matched] complete: {out_root}")
    print(f"[matched] summary: {out_root / summary_name}")


if __name__ == "__main__":
    main()
