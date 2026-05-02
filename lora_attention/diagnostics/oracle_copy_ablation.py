#!/usr/bin/env python3
"""Strict oracle-copy ablation for MoELoRA style-block injection path."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_attention.models.lora_pool import LoRAPool, STYLE_BLOCK_PREFIX
from lora_attention.models.moe_lora_v2 import MoELoRAv2
from lora_attention.utils.lora_inject import inject_lora, unload_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle-copy ablation for MoELoRA path isolation.")
    p.add_argument("--style_name", default="Baroque")
    p.add_argument("--expert_name", default=None, help="Exact expert override (e.g. style_0000_Baroque).")
    p.add_argument("--prompt", default="A dog")
    p.add_argument("--style_prompt", action="store_true", help="Use prompt: A dog in <style_name> style")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--style_alpha", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--checkpoint", default="/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt")
    p.add_argument("--zoo_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras")
    p.add_argument("--style_images_dir", default="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images")
    p.add_argument("--cache_dir", default="/scratch/eyavuz21/lora_attention")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default=None, help="cuda or cpu. Default: cuda if available.")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_model", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--product_synth", action="store_true", default=True)
    p.add_argument("--legacy_synth", action="store_false", dest="product_synth")
    return p.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(f"/scratch/eyavuz21/lora_attention/diagnostics/oracle_copy_{now_tag()}")


def resolve_device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if dtype_str == "fp16" else torch.float32


def select_expert(pool: LoRAPool, style_name: str, expert_name: str | None) -> Tuple[int, str, list[str]]:
    style_names = pool.style_names
    if expert_name:
        if expert_name not in style_names:
            raise ValueError(f"--expert_name '{expert_name}' not found in pool.")
        return style_names.index(expert_name), expert_name, [expert_name]

    matches = [n for n in style_names if style_name.lower() in n.lower()]
    if not matches:
        raise ValueError(f"No expert names matched style_name='{style_name}'.")
    chosen = matches[0]
    return style_names.index(chosen), chosen, matches


def build_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        torch_dtype=dtype,
    ).to(device)
    return pipe


def generate(pipe, prompt: str, seed: int, steps: int, guidance: float) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
    ).images[0]


def pixel_mae(a: Image.Image, b: Image.Image) -> float:
    a_np = np.asarray(a).astype(np.float32)
    b_np = np.asarray(b).astype(np.float32)
    return float(np.abs(a_np - b_np).mean())


def make_grid(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    tile_w, tile_h = 320, 320
    label_h = 34
    canvas = Image.new("RGB", (tile_w * len(items), tile_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        x = i * tile_w
        canvas.paste(image.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS), (x, label_h))
        draw.text((x + 8, 9), label, fill=(0, 0, 0))
    canvas.save(out_path)


def synth_one_hot_gt(
    model: MoELoRAv2,
    pool_indices: list[int],
    gt_local_index: int,
    device: torch.device,
    use_product_space: bool,
) -> Dict[str, torch.Tensor]:
    n = len(pool_indices)
    t = model.num_tensor_groups
    r = model.rank

    w_down_parts = []
    w_up_parts = []
    for d_in in model._dim_order:  # pylint: disable=protected-access
        dk = model._down_keys_by_dim[d_in]  # pylint: disable=protected-access
        uk = model._up_keys_by_dim[d_in]  # pylint: disable=protected-access
        w_down = model.pool.get_stacked_tensors_multi(pool_indices, dk, device=str(device))
        w_up = model.pool.get_stacked_tensors_multi(pool_indices, uk, device=str(device))
        w_down_parts.append(w_down)
        w_up_parts.append(w_up)

    a = torch.zeros((n, t, r), device=device, dtype=torch.float32)
    a[gt_local_index, :, :] = 1.0

    if use_product_space:
        return model._synthesise_product_space(a, w_down_parts, w_up_parts)  # pylint: disable=protected-access
    return model._synthesise_batched(a, w_down_parts, w_up_parts)  # pylint: disable=protected-access


def tensor_diff_rows(
    direct_sd: Dict[str, torch.Tensor],
    oracle_sd: Dict[str, torch.Tensor],
) -> tuple[list[dict], dict]:
    rows = []
    all_keys = sorted(set(direct_sd.keys()) | set(oracle_sd.keys()))

    max_abs_vals = []
    mean_abs_vals = []
    cos_vals = []
    ratio_vals = []
    missing_in_direct = 0
    missing_in_oracle = 0

    for key in all_keys:
        d = direct_sd.get(key)
        o = oracle_sd.get(key)
        miss_d = d is None
        miss_o = o is None
        missing_in_direct += int(miss_d)
        missing_in_oracle += int(miss_o)

        row = {
            "key": key,
            "shape_direct": "" if d is None else str(tuple(d.shape)),
            "shape_oracle": "" if o is None else str(tuple(o.shape)),
            "shape_match": False if (d is None or o is None) else (tuple(d.shape) == tuple(o.shape)),
            "missing_in_direct": miss_d,
            "missing_in_oracle": miss_o,
            "max_abs_diff": "",
            "mean_abs_diff": "",
            "norm_direct": "",
            "norm_oracle": "",
            "norm_ratio_oracle_over_direct": "",
            "cosine_similarity": "",
        }

        if d is not None and o is not None and tuple(d.shape) == tuple(o.shape):
            d32 = d.detach().float().reshape(-1)
            o32 = o.detach().float().reshape(-1)
            diff = (d32 - o32).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            norm_d = float(d32.norm().item())
            norm_o = float(o32.norm().item())

            if norm_d <= 1e-12 and norm_o <= 1e-12:
                cos = 1.0
            elif norm_d <= 1e-12 or norm_o <= 1e-12:
                cos = 0.0
            else:
                cos = float(torch.nn.functional.cosine_similarity(d32.unsqueeze(0), o32.unsqueeze(0)).item())

            ratio = float(norm_o / norm_d) if norm_d > 1e-12 else (1.0 if norm_o <= 1e-12 else float("inf"))

            row["max_abs_diff"] = max_abs
            row["mean_abs_diff"] = mean_abs
            row["norm_direct"] = norm_d
            row["norm_oracle"] = norm_o
            row["norm_ratio_oracle_over_direct"] = ratio
            row["cosine_similarity"] = cos

            max_abs_vals.append(max_abs)
            mean_abs_vals.append(mean_abs)
            cos_vals.append(cos)
            if np.isfinite(ratio):
                ratio_vals.append(ratio)

        rows.append(row)

    agg = {
        "num_keys_direct": len(direct_sd),
        "num_keys_oracle": len(oracle_sd),
        "num_missing_in_direct": missing_in_direct,
        "num_missing_in_oracle": missing_in_oracle,
        "max_of_max_abs_diff": float(max(max_abs_vals)) if max_abs_vals else None,
        "mean_of_mean_abs_diff": float(np.mean(mean_abs_vals)) if mean_abs_vals else None,
        "mean_cosine_similarity": float(np.mean(cos_vals)) if cos_vals else None,
        "min_cosine_similarity": float(min(cos_vals)) if cos_vals else None,
        "mean_norm_ratio": float(np.mean(ratio_vals)) if ratio_vals else None,
    }
    return rows, agg


def save_tensor_diff_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "key",
        "shape_direct",
        "shape_oracle",
        "shape_match",
        "missing_in_direct",
        "missing_in_oracle",
        "max_abs_diff",
        "mean_abs_diff",
        "norm_direct",
        "norm_oracle",
        "norm_ratio_oracle_over_direct",
        "cosine_similarity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    prompt = f"A dog in {args.style_name} style" if args.style_prompt else args.prompt

    report = {
        "style_name": args.style_name,
        "expert_name": None,
        "expert_index": None,
        "prompt": prompt,
        "seed": args.seed,
        "style_alpha": args.style_alpha,
        "temperature": args.temperature,
        "checkpoint": args.checkpoint,
        "output_dir": str(out_dir),
        "product_synth": args.product_synth,
        "variant_errors": {},
    }

    print(f"[oracle] output_dir={out_dir}")
    print(f"[oracle] device={device}, dtype={dtype}")
    print(f"[oracle] prompt={prompt}")

    pool = LoRAPool(zoo_dir=args.zoo_dir, cache_dir=args.cache_dir)
    expert_idx, expert_name, matches = select_expert(pool, args.style_name, args.expert_name)
    report["expert_name"] = expert_name
    report["expert_index"] = int(expert_idx)
    report["matched_experts"] = matches
    print(f"[oracle] expert matches={matches}")
    print(f"[oracle] selected expert={expert_name} (idx={expert_idx})")

    expert_path = Path(args.zoo_dir) / expert_name / "pytorch_lora_weights.safetensors"
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert safetensors not found: {expert_path}")

    style_ref_path = Path(args.style_images_dir) / expert_name / f"{expert_name}.jpg"
    style_ref_img = Image.open(style_ref_path).convert("RGB") if style_ref_path.exists() else None
    report["style_reference_image"] = str(style_ref_path) if style_ref_path.exists() else None

    print(f"[oracle] loading direct reference: {expert_path}")
    direct_full = load_file(str(expert_path))
    direct_styleblock = {k: v.detach().cpu() for k, v in direct_full.items() if STYLE_BLOCK_PREFIX in k}
    if not direct_styleblock:
        raise RuntimeError("Direct reference style-block tensors are empty.")
    torch.save(direct_styleblock, out_dir / "used_direct_reference_subset.pt")
    report["used_direct_reference_subset"] = str(out_dir / "used_direct_reference_subset.pt")
    report["num_keys_direct_styleblock"] = len(direct_styleblock)
    report["num_keys_direct_full"] = len(direct_full)

    oracle_copy = {k: v.detach().cpu() for k, v in pool.get_style_tensors(expert_idx).items()}
    torch.save(oracle_copy, out_dir / "used_synth_oracle.pt")
    report["used_synth_oracle"] = str(out_dir / "used_synth_oracle.pt")
    report["num_keys_oracle_copy"] = len(oracle_copy)

    rows, tensor_agg = tensor_diff_rows(direct_styleblock, oracle_copy)
    save_tensor_diff_csv(rows, out_dir / "tensor_diff.csv")
    report.update(tensor_agg)

    images: dict[str, Image.Image] = {}
    grid_items: list[tuple[str, Image.Image]] = []
    if style_ref_img is not None:
        grid_items.append(("query/ref", style_ref_img))

    pipe = None
    try:
        pipe = build_pipeline(args, device=device, dtype=dtype)

        # Variant 1: vanilla
        try:
            unload_lora(pipe)
            img = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            images["vanilla"] = img
            img.save(out_dir / "vanilla.png")
            grid_items.append(("vanilla", img))
        except Exception as e:  # pylint: disable=broad-except
            report["variant_errors"]["vanilla"] = str(e)

        # Variant 2: direct_reference (style-block only; fairness with MoE style path)
        try:
            unload_lora(pipe)
            direct_scaled = {k: v * args.style_alpha for k, v in direct_styleblock.items()}
            pipe.load_lora_into_unet(direct_scaled, None, pipe.unet)
            img = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            images["direct_reference"] = img
            img.save(out_dir / "direct_reference.png")
            grid_items.append(("direct_ref_style", img))
            unload_lora(pipe)
        except Exception as e:  # pylint: disable=broad-except
            report["variant_errors"]["direct_reference_styleblock"] = str(e)

        # Optional baseline: full direct B-LoRA
        if len(direct_full) != len(direct_styleblock):
            try:
                unload_lora(pipe)
                full_scaled = {k: v * args.style_alpha for k, v in direct_full.items()}
                pipe.load_lora_into_unet(full_scaled, None, pipe.unet)
                img = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
                images["direct_reference_full"] = img
                img.save(out_dir / "direct_reference_full.png")
                grid_items.append(("direct_ref_full", img))
                unload_lora(pipe)
                report["direct_reference_full_generated"] = True
            except Exception as e:  # pylint: disable=broad-except
                report["variant_errors"]["direct_reference_full"] = str(e)

        # Variant 3: oracle_copy through synth injection path
        try:
            unload_lora(pipe)
            inject_lora(pipe, oracle_copy, style_alpha=args.style_alpha)
            img = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            images["oracle_copy"] = img
            img.save(out_dir / "oracle_copy.png")
            grid_items.append(("oracle_copy", img))
            unload_lora(pipe)
        except Exception as e:  # pylint: disable=broad-except
            report["variant_errors"]["oracle_copy"] = str(e)

        # Variant 4: moe_forced_gt one-hot synthesis
        try:
            model = MoELoRAv2(pool=pool).to(device)
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            model.encoder.eval()
            pool_indices = list(range(pool.num_experts))
            gt_local_idx = pool_indices.index(expert_idx)
            moe_forced = synth_one_hot_gt(
                model=model,
                pool_indices=pool_indices,
                gt_local_index=gt_local_idx,
                device=device,
                use_product_space=args.product_synth,
            )
            moe_forced_cpu = {k: v.detach().cpu() for k, v in moe_forced.items()}
            torch.save(moe_forced_cpu, out_dir / "used_moe_forced_gt.pt")
            report["used_moe_forced_gt"] = str(out_dir / "used_moe_forced_gt.pt")
            report["num_keys_moe_forced_gt"] = len(moe_forced_cpu)

            unload_lora(pipe)
            inject_lora(pipe, moe_forced_cpu, style_alpha=args.style_alpha)
            img = generate(pipe, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            images["moe_forced_gt"] = img
            img.save(out_dir / "moe_forced_gt.png")
            grid_items.append(("moe_forced_gt", img))
            unload_lora(pipe)
        except Exception as e:  # pylint: disable=broad-except
            report["variant_errors"]["moe_forced_gt"] = str(e)
    except Exception as e:  # pylint: disable=broad-except
        report["variant_errors"]["pipeline_init_or_global"] = str(e)
    finally:
        if pipe is not None:
            try:
                unload_lora(pipe)
            except Exception:
                pass

    # Pixel MAE metrics
    def mae_pair(a: str, b: str):
        if a not in images or b not in images:
            return None
        return pixel_mae(images[a], images[b])

    report["pixel_mae_direct_vs_vanilla"] = mae_pair("direct_reference", "vanilla")
    report["pixel_mae_oracle_vs_vanilla"] = mae_pair("oracle_copy", "vanilla")
    report["pixel_mae_oracle_vs_direct"] = mae_pair("oracle_copy", "direct_reference")
    report["pixel_mae_moe_forced_vs_oracle"] = mae_pair("moe_forced_gt", "oracle_copy")
    report["pixel_mae_moe_forced_vs_direct"] = mae_pair("moe_forced_gt", "direct_reference")

    # Grid
    if grid_items:
        make_grid(grid_items, out_dir / "grid.png")
        report["grid"] = str(out_dir / "grid.png")

    # Verdict logic
    max_abs = report.get("max_of_max_abs_diff")
    mean_abs = report.get("mean_of_mean_abs_diff")
    mean_cos = report.get("mean_cosine_similarity")
    oracle_vs_direct = report.get("pixel_mae_oracle_vs_direct")
    moe_vs_oracle = report.get("pixel_mae_moe_forced_vs_oracle")
    moe_vs_direct = report.get("pixel_mae_moe_forced_vs_direct")

    tensor_near_zero = (
        max_abs is not None
        and mean_abs is not None
        and mean_cos is not None
        and max_abs <= 1e-5
        and mean_abs <= 1e-6
        and mean_cos >= 0.99999
    )
    image_match = oracle_vs_direct is not None and oracle_vs_direct <= 1.0
    moe_oracle_match = moe_vs_oracle is not None and moe_vs_oracle <= 1.0
    moe_direct_match = moe_vs_direct is not None and moe_vs_direct <= 1.0

    if not tensor_near_zero:
        verdict = "Case C: Oracle state dict is not equivalent to direct reference. Investigate key filtering, transpose, missing keys, dtype, or sign/scale."
    elif not image_match:
        verdict = "Case B: Tensor copy is correct, but injection path differs. Investigate injection wrapper, scale, active processors, and layer coverage."
    elif moe_vs_oracle is None:
        verdict = "Case A: Oracle copy path is correct. Direct expert tensors survive the synth/inject wrapper."
    elif not moe_oracle_match:
        verdict = "Case D: Oracle copy matches direct reference, but moe_forced_gt differs. One-hot MoE synthesis is not identity-preserving."
    elif moe_oracle_match and moe_direct_match:
        verdict = "Case E: Oracle copy and moe_forced_gt both match direct reference. Tensor and injection paths are correct; remaining issue is routing or soft mixing."
    else:
        verdict = "Case A: Oracle copy path is correct. Direct expert tensors survive the synth/inject wrapper."
    report["verdict"] = verdict

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[oracle] saved: {out_dir / 'vanilla.png'}")
    print(f"[oracle] saved: {out_dir / 'direct_reference.png'}")
    print(f"[oracle] saved: {out_dir / 'oracle_copy.png'}")
    print(f"[oracle] saved: {out_dir / 'moe_forced_gt.png'}")
    print(f"[oracle] saved: {out_dir / 'tensor_diff.csv'}")
    print(f"[oracle] saved: {report_path}")
    print(f"[oracle] verdict: {verdict}")


if __name__ == "__main__":
    main()
