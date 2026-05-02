"""
LoRAGenerator: style image → B-LoRA style-block weight tensors.

Architecture
------------
  Style encoding (one of):

  A) CLIP path (default ``style_encoder="clip"``)
      pixel_values [B,3,H,W] (CLIP preprocessing)
          → CLIPVisionModel (frozen) → patch tokens [B, N, 768]
          → Linear(768, d_model)     → memory [B, N, d_model]

  B) SDXL VAE latents (``style_encoder="vae_latents"``), same preprocessing as
     ``train_dreambooth_b-lora_sdxl.py``: RGB in [-1,1], ``vae.encode``,
     ``latent_dist.sample()``, then ``* vae.config.scaling_factor``.
      scaled_latents [B, 4, h, w]
          → Conv2d patch embed (4 → d_model) → memory [B, N, d_model]

  Then (both paths):
      nn.TransformerDecoder (Q = learnable queries, K/V = memory)
          → per-shape Linear heads → dict of LoRA weights
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------------
# SDXL VAE loading (matches MoLoRAs/B-LoRA_files/train_dreambooth_b-lora_sdxl.py)
# ---------------------------------------------------------------------------

def load_sdxl_vae(
    pretrained_model_name_or_path: str,
    pretrained_vae_model_name_or_path: Optional[str] = None,
    revision: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float32,
) -> AutoencoderKL:
    """
    Load the same VAE as the B-LoRA SDXL script: either the ``vae`` subfolder
    of the base model, or a standalone VAE checkpoint.
    """
    vae_path = (
        pretrained_model_name_or_path
        if pretrained_vae_model_name_or_path is None
        else pretrained_vae_model_name_or_path
    )
    kwargs = {"torch_dtype": torch_dtype}
    if revision is not None:
        kwargs["revision"] = revision
    return AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if pretrained_vae_model_name_or_path is None else None,
        **kwargs,
    )


def encode_rgb_to_scaled_latents(
    vae: AutoencoderKL,
    pixel_values: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """
    B-LoRA / SDXL training convention: ``pixel_values`` in VAE dtype,
    shape [B, 3, H, W], normalized to [-1, 1].

    Returns scaled latents [B, 4, h, w] (``sample * scaling_factor``).
    """
    if sample:
        dist = vae.encode(pixel_values).latent_dist
        latents = dist.sample()
    else:
        latents = vae.encode(pixel_values).latent_dist.mode()
    return latents * vae.config.scaling_factor


# ---------------------------------------------------------------------------
# Shape group registry
# ---------------------------------------------------------------------------

def _shape_group_id(down_shape: tuple[int, int], up_shape: tuple[int, int]) -> str:
    return f"{down_shape[0]}x{down_shape[1]}_{up_shape[0]}x{up_shape[1]}"


# ---------------------------------------------------------------------------
# LoRAGenerator
# ---------------------------------------------------------------------------

class LoRAGenerator(nn.Module):
    """
    Parameters
    ----------
    key_shapes : dict[str, tuple[int,int]]
        Maps each LoRA weight key to its (rows, cols) shape.
        Built once from a sample safetensors file (style block keys only).
    d_model    : int   Hidden dimension of the Transformer decoder.
    n_heads    : int   Number of attention heads.
    n_layers   : int   Number of Transformer decoder layers.
    clip_model : str   HuggingFace model ID for the frozen CLIP vision encoder.
    style_encoder : ``"clip"`` | ``"vae_latents"`` — what feeds the Transformer memory.
    latent_patch_size : Patch stride for Conv2d embedding of VAE latents (4 → 64×64 lat → 16×16 tokens).
    dropout    : float Dropout applied inside the Transformer.
    """

    def __init__(
        self,
        key_shapes: dict[str, tuple[int, int]],
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        clip_model: str = "openai/clip-vit-base-patch32",
        style_encoder: Literal["clip", "vae_latents"] = "clip",
        latent_patch_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.style_encoder = style_encoder
        self.latent_patch_size = latent_patch_size

        if style_encoder == "clip":
            self.clip = CLIPVisionModel.from_pretrained(clip_model)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            clip_hidden = self.clip.config.hidden_size
            self.img_proj = nn.Linear(clip_hidden, d_model)
            self.latent_patch = None
        elif style_encoder == "vae_latents":
            self.clip = None
            self.img_proj = None
            self.latent_patch = nn.Conv2d(
                4, d_model,
                kernel_size=latent_patch_size,
                stride=latent_patch_size,
            )
        else:
            raise ValueError(f"Unknown style_encoder: {style_encoder!r}")

        # ---- sort keys → deterministic query ordering ----
        self.keys: list[str] = sorted(key_shapes.keys())
        self.key_shapes: dict[str, tuple[int, int]] = {k: key_shapes[k] for k in self.keys}
        n_queries = len(self.keys)

        # ---- learnable queries (one per LoRA weight tensor) ----
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # ---- Transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ---- per-shape-group decoder heads ----
        # Discover unique (down_shape, up_shape) groups.
        # A key ending with ".down.weight" and its partner ".up.weight" form one pair.
        # Each down/up tensor is decoded independently from its own query.
        self._build_decoder_heads(d_model)

        # ---- index: query_i → (group_id, flat_size, shape) ----
        self._query_meta: list[tuple[str, int, tuple[int, int]]] = []
        for k in self.keys:
            shape = self.key_shapes[k]
            gid = self._tensor_group(k, shape)
            self._query_meta.append((gid, math.prod(shape), shape))

    # ------------------------------------------------------------------
    def _tensor_group(self, key: str, shape: tuple[int, int]) -> str:
        """Return the group id for a given key/shape."""
        r, c = shape
        return f"{r}x{c}"

    def _build_decoder_heads(self, d_model: int) -> None:
        """Build one Linear head per unique tensor shape."""
        shape_set: set[tuple[int, int]] = set()
        for k in self.keys:
            shape_set.add(self.key_shapes[k])

        self.heads = nn.ModuleDict()
        for shape in shape_set:
            gid = f"{shape[0]}x{shape[1]}"
            flat = math.prod(shape)
            self.heads[gid] = nn.Linear(d_model, flat)

    # ------------------------------------------------------------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values : [B, 3, H, W] (CLIP preprocessing)
        returns      : [B, N_patches+1, d_model]
        """
        if self.style_encoder != "clip" or self.clip is None:
            raise RuntimeError("encode_image requires style_encoder='clip'.")
        with torch.no_grad():
            outputs = self.clip(pixel_values=pixel_values)
        img_tokens = outputs.last_hidden_state
        return self.img_proj(img_tokens)

    def encode_scaled_latents(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        """
        scaled_latents : [B, 4, h, w] after VAE encode × ``scaling_factor`` (trainable path).

        returns : [B, N, d_model] with N = (h / patch) * (w / patch).
        """
        if self.style_encoder != "vae_latents" or self.latent_patch is None:
            raise RuntimeError("encode_scaled_latents requires style_encoder='vae_latents'.")
        x = self.latent_patch(scaled_latents)
        return x.flatten(2).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        scaled_latents: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Provide exactly one of:

        - ``pixel_values`` [B, 3, H, W] when ``style_encoder == "clip"``.
        - ``scaled_latents`` [B, 4, h, w] when ``style_encoder == "vae_latents"``.

        Returns a dict mapping each LoRA key → Tensor of shape [B, *weight_shape].
        """
        if (pixel_values is None) == (scaled_latents is None):
            raise ValueError("Pass exactly one of pixel_values or scaled_latents.")

        if pixel_values is not None:
            if self.style_encoder != "clip":
                raise ValueError("pixel_values only valid for style_encoder='clip'.")
            B = pixel_values.shape[0]
            memory = self.encode_image(pixel_values)
        else:
            if self.style_encoder != "vae_latents":
                raise ValueError("scaled_latents only valid for style_encoder='vae_latents'.")
            B = scaled_latents.shape[0]
            memory = self.encode_scaled_latents(scaled_latents)

        # Expand learnable queries to batch
        tgt = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_queries, d_model]

        # Cross-attend queries to image tokens
        out = self.transformer(tgt=tgt, memory=memory)   # [B, n_queries, d_model]

        # Decode each query to its LoRA weight
        lora_out: dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.keys):
            gid, flat, shape = self._query_meta[i]
            token = out[:, i, :]                         # [B, d_model]
            weight_flat = self.heads[gid](token)         # [B, flat]
            lora_out[key] = weight_flat.view(B, *shape)  # [B, rows, cols]

        return lora_out

    # ------------------------------------------------------------------
    @classmethod
    def from_sample_lora(
        cls,
        sample_lora_dict: dict[str, torch.Tensor],
        **kwargs,
    ) -> "LoRAGenerator":
        """Convenience constructor: build key_shapes from a loaded lora dict."""
        key_shapes = {k: tuple(v.shape) for k, v in sample_lora_dict.items()}
        return cls(key_shapes=key_shapes, **kwargs)
