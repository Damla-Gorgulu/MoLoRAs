import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from blora_utils import BLOCKS, filter_lora, scale_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="B-LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--content_B_LoRA", type=str, default=None, help="path for the content B-LoRA"
    )
    parser.add_argument(
        "--style_B_LoRAs", type=str, nargs='+', default=None, help="paths for the style B-LoRAs (space-separated)"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content B-LoRA weights"
    )
    parser.add_argument(
        "--style_alphas", type=float, nargs='+', default=None, help="alpha parameters to scale each style B-LoRA (space-separated, must match number of style_B_LoRAs)"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


def merge_loras(lora_dicts):
    """Merge multiple LoRA state dictionaries by adding overlapping weights."""
    merged = {}
    for lora_dict in lora_dicts:
        for key, value in lora_dict.items():
            if key in merged:
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


if __name__ == '__main__':
    args = parse_args()
    
    # Validate style_B_LoRAs and style_alphas match
    if args.style_B_LoRAs is not None and args.style_alphas is not None:
        if len(args.style_B_LoRAs) != len(args.style_alphas):
            raise ValueError(
                f"Number of style_B_LoRAs ({len(args.style_B_LoRAs)}) must match "
                f"number of style_alphas ({len(args.style_alphas)})"
            )
    elif args.style_B_LoRAs is not None and args.style_alphas is None:
        # Default all alphas to 1.0
        args.style_alphas = [1.0] * len(args.style_B_LoRAs)
    
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    # Get Content B-LoRA SD
    if args.content_B_LoRA is not None:
        content_B_LoRA_sd, _ = pipeline.lora_state_dict(args.content_B_LoRA)
        content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
        content_B_LoRA = scale_lora(content_B_LoRA, args.content_alpha)
    else:
        content_B_LoRA = {}

    # Get Style B-LoRAs SD and merge them
    style_B_LoRAs_list = []
    if args.style_B_LoRAs is not None:
        for style_path, style_alpha in zip(args.style_B_LoRAs, args.style_alphas):
            style_B_LoRA_sd, _ = pipeline.lora_state_dict(style_path)
            style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
            style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
            style_B_LoRAs_list.append(style_B_LoRA)
            print(f"Loaded style LoRA from {style_path} with alpha={style_alpha}")
    
    # Merge all style LoRAs
    if style_B_LoRAs_list:
        merged_style_B_LoRA = merge_loras(style_B_LoRAs_list)
    else:
        merged_style_B_LoRA = {}

    # Merge Content and Style B-LoRAs
    res_lora = {**content_B_LoRA, **merged_style_B_LoRA}

    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    # Generate
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
