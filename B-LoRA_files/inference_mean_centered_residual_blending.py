"""
Mean-Centered Residual Blending for B-LoRA inference.

Algorithm A: Extracts a shared base from anchor LoRAs, isolates style residuals,
blends residuals using DINOv2-based proximity to a new ref image, then reconstructs
the final LoRA.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from blora_utils import BLOCKS, filter_lora, scale_lora


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dinov2():
    """Load DINOv2 model and processor from Hugging Face (uses local cache)."""
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    return model, processor


def get_dino_embedding(model, processor, image_path, device="cuda"):
    """Extract DINOv2 CLS embedding for an image."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token is first in last_hidden_state
    return outputs.last_hidden_state[:, 0].float()


def compute_proximity_weights(new_embedding, anchor_embeddings, temperature=1.0):
    """Softmax-normalized cosine similarity between new image and anchors."""
    new_norm = new_embedding / (new_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    anchor_norm = anchor_embeddings / (anchor_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (new_norm @ anchor_norm.mT).squeeze(0)
    weights = torch.softmax(cos_sim / temperature, dim=0)
    return weights.cpu().numpy()


def state_dict_mean(state_dicts):
    """Element-wise mean of state dicts (must share keys)."""
    keys = list(state_dicts[0].keys())
    mean_dict = {}
    for k in keys:
        mean_dict[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(dim=0)
    return mean_dict


def state_dict_subtract(a, b):
    """Element-wise a - b for state dicts."""
    return {k: a[k].float() - b[k].float() for k in a.keys()}


def state_dict_add(a, b):
    """Element-wise a + b for state dicts."""
    return {k: a[k].float() + b[k].float() for k in a.keys()}


def state_dict_scale(sd, scalar):
    """Scale state dict by scalar."""
    return {k: v * scalar for k, v in sd.items()}


def state_dict_weighted_sum(state_dicts, weights):
    """Weighted sum of state dicts."""
    result = {}
    for k in state_dicts[0].keys():
        result[k] = sum(sd[k].float() * w for sd, w in zip(state_dicts, weights))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Mean-Centered Residual Blending inference")
    parser.add_argument("--prompt", type=str, required=True, help="B-LoRA prompt")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save images")
    parser.add_argument(
        "--content_B_LoRA", type=str, default=None, help="Path for content B-LoRA"
    )
    parser.add_argument(
        "--anchor_loras", type=str, nargs="+", required=True,
        help="Paths to K anchor style LoRA files (e.g. 5 anchors)",
    )
    parser.add_argument(
        "--anchor_images", type=str, nargs="+", required=True,
        help="Paths to K anchor ref images (one per anchor LoRA)",
    )
    parser.add_argument(
        "--new_ref_image", type=str, required=True,
        help="Path to new reference image I_new for proximity-based blending",
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1.0,
        help="Alpha for content B-LoRA scaling",
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1.0,
        help="Alpha for synthesized style LoRA scaling",
    )
    parser.add_argument(
        "--proximity_temperature", type=float, default=1.0,
        help="Temperature for softmax over cosine similarities (higher = flatter weights)",
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if len(args.anchor_loras) != len(args.anchor_images):
        parser.error(
            f"Number of anchor_loras ({len(args.anchor_loras)}) must match "
            f"number of anchor_images ({len(args.anchor_images)})"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pipeline
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)

    # Load DINOv2 for proximity (uses local cache facebook/dinov2-base)
    dinov2, processor = load_dinov2()
    dinov2 = dinov2.to(device).eval()

    # 1. Load anchor LoRAs and extract style blocks
    anchor_style_loras = []
    for path in args.anchor_loras:
        sd, _ = pipeline.lora_state_dict(path)
        style_sd = filter_lora(sd, BLOCKS["style"])
        anchor_style_loras.append(style_sd)

    # 2. LoRA_base = mean of anchor LoRAs
    lora_base = state_dict_mean(anchor_style_loras)

    # 3. LoRA_im_k = LoRA_k - LoRA_base (style residuals)
    lora_residuals = [
        state_dict_subtract(lorak, lora_base)
        for lorak in anchor_style_loras
    ]

    # 4. Proximity: DINOv2 embeddings, softmax-normalized cosine similarity
    new_embedding = get_dino_embedding(dinov2, processor, args.new_ref_image, device)  # (1, 768)
    anchor_embeddings = torch.stack([
        get_dino_embedding(dinov2, processor, path, device).squeeze(0)
        for path in args.anchor_images
    ])  # (K, 768)
    weights = compute_proximity_weights(
        new_embedding, anchor_embeddings,
        temperature=args.proximity_temperature,
    )
    print("Proximity weights (DINOv2 softmax cos-sim):", [round(w, 4) for w in weights])

    # 5. LoRA_new_im = sum(w_k * LoRA_im_k)
    lora_new_im = state_dict_weighted_sum(lora_residuals, weights)

    # 6. Delta W_new = LoRA_base + LoRA_new_im
    style_B_LoRA = state_dict_add(lora_base, lora_new_im)
    style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)

    # Content B-LoRA
    if args.content_B_LoRA is not None:
        content_B_LoRA_sd, _ = pipeline.lora_state_dict(args.content_B_LoRA)
        content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS["content"])
        content_B_LoRA = scale_lora(content_B_LoRA, args.content_alpha)
    else:
        content_B_LoRA = {}

    # Merge and load
    res_lora = {**content_B_LoRA, **style_B_LoRA}
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    # Generate
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipeline(
        args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
    ).images

    # Save
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    prompt_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.prompt)[:80]
    for i, img in enumerate(images):
        img.save(Path(args.output_path) / f"{prompt_safe}_{i}.jpg")

    print(f"Saved {len(images)} images to {args.output_path}")
