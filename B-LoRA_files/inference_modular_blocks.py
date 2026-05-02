import argparse
import random

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from blora_utils import BLOCKS, filter_lora, scale_lora, close_blocks


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
        "--style_B_LoRA", type=str, default=None, help="path for the style B-LoRA"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content B-LoRA weights"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style B-LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    parser.add_argument(
        "--blocks_to_close", type=int, nargs="*", default=None,
        help="transformer block indices (0-9) to close/disable in style LoRA. "
             "When set, the specified blocks' LoRA weights are excluded (e.g. --blocks_to_close 0 1 2)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    if args.blocks_to_close is not None:
        for b in args.blocks_to_close:
            if b < 0 or b > 9:
                parser.error(f"--blocks_to_close indices must be between 0 and 9 (inclusive), got {b}")
    return args


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

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

    # Get Style B-LoRA SD
    if args.style_B_LoRA is not None:
        style_B_LoRA_sd, _ = pipeline.lora_state_dict(args.style_B_LoRA)
        style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
        if args.blocks_to_close:
            style_prefix = BLOCKS['style'][0]  # 'unet.up_blocks.0.attentions.1'
            style_B_LoRA = close_blocks(style_B_LoRA, args.blocks_to_close, style_prefix)
        style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)
    else:
        style_B_LoRA = {}

    # Merge B-LoRAs SD
    res_lora = {**content_B_LoRA, **style_B_LoRA}

    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    # Generate
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    images = pipeline(
        args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator
    ).images

    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
