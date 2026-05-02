import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B-LoRA_files"))
from blora_utils import filter_lora, BLOCKS


STYLE_BLOCK = BLOCKS["style"]  # ['unet.up_blocks.0.attentions.1']


class StyleLoRADataset(Dataset):
    """
    Each sample is one style: a reference image + its B-LoRA style-block weights.

    Directory layout expected:
        checkpoint_dir/<style>/pytorch_lora_weights.safetensors
        image_dir/<style>/<any image file>   (first image found is used)

    Returns:
        pixel_values  : Tensor [3, H, W] processed by CLIPProcessor (float32)
        vae_pixel_values : optional Tensor [3, H_vae, W_vae] in [-1, 1] for SDXL VAE encode
        lora_dict     : dict[str, Tensor]  — style block LoRA weights
        style_name    : str
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        checkpoint_dir: str,
        image_dir: str,
        clip_model_id: str = "openai/clip-vit-large-patch14",
        cache_loras: bool = True,
        vae_image_size: Optional[Tuple[int, int]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.image_dir = Path(image_dir)
        self.cache_loras = cache_loras
        self.vae_image_size = vae_image_size

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        if vae_image_size is not None:
            h, w = vae_image_size
            self._vae_tf = transforms.Compose(
                [
                    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        self.styles = self._discover_styles()
        if len(self.styles) == 0:
            raise ValueError(
                f"No valid styles found. "
                f"checkpoint_dir={checkpoint_dir}, image_dir={image_dir}"
            )

        self._lora_cache: dict[str, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def _discover_styles(self) -> list[str]:
        styles = []
        for ckpt_path in sorted(self.checkpoint_dir.iterdir()):
            if not ckpt_path.is_dir():
                continue
            weights = ckpt_path / "pytorch_lora_weights.safetensors"
            if not weights.exists():
                continue
            img_dir = self.image_dir / ckpt_path.name
            if not img_dir.is_dir():
                continue
            if self._find_image(img_dir) is None:
                continue
            styles.append(ckpt_path.name)
        return styles

    def _find_image(self, directory: Path) -> Optional[Path]:
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() in self.IMAGE_EXTENSIONS:
                return f
        return None

    # ------------------------------------------------------------------
    def _load_lora(self, style: str) -> dict[str, torch.Tensor]:
        if self.cache_loras and style in self._lora_cache:
            return self._lora_cache[style]

        path = self.checkpoint_dir / style / "pytorch_lora_weights.safetensors"
        state_dict: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

        style_dict = filter_lora(state_dict, STYLE_BLOCK)

        if self.cache_loras:
            self._lora_cache[style] = style_dict
        return style_dict

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.styles)

    def __getitem__(self, idx: int) -> dict:
        style = self.styles[idx]

        img_path = self._find_image(self.image_dir / style)
        image = Image.open(img_path).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].squeeze(0)  # [3, H, W]

        lora_dict = self._load_lora(style)

        out = {
            "pixel_values": pixel_values,
            "lora_dict": lora_dict,
            "style_name": style,
        }
        if self.vae_image_size is not None:
            out["vae_pixel_values"] = self._vae_tf(image)
        return out


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: stack pixel_values as a batch tensor; keep lora_dicts as
    a list (keys/shapes are identical across styles so stacking is also fine).
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Stack each LoRA weight tensor across the batch dimension
    keys = list(batch[0]["lora_dict"].keys())
    lora_batch = {k: torch.stack([b["lora_dict"][k] for b in batch]) for k in keys}

    style_names = [b["style_name"] for b in batch]

    out = {
        "pixel_values": pixel_values,
        "lora_dict": lora_batch,
        "style_names": style_names,
    }
    if "vae_pixel_values" in batch[0]:
        out["vae_pixel_values"] = torch.stack([b["vae_pixel_values"] for b in batch])
    return out
