"""
LoRAGenerator: style image → B-LoRA style-block weight tensors.

Architecture
------------
  pixel_values
      │
  CLIPVisionModel (frozen)   →  patch tokens  [B, N_patches+1, 768]
      │
  img_proj  Linear(768, d_model)              [B, N_patches+1, d_model]
      │
  nn.TransformerDecoder   (Q = learnable queries, K/V = image tokens)
      queries  [n_queries, d_model]  →  [B, n_queries, d_model]
      │
  Per-shape-group Linear decoders
      group_A  Linear(d_model, flat_A)  →  down [rank, 1280]  + up [1280, rank]
      group_B  Linear(d_model, flat_B)  →  down [rank, 2048]  + up [1280, rank]
      │
  Reconstruct dict[key → Tensor]  (same keys/shapes as B-LoRA style block)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


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
    dropout    : float Dropout applied inside the Transformer.
    """

    def __init__(
        self,
        key_shapes: dict[str, tuple[int, int]],
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        clip_model: str = "openai/clip-vit-base-patch32",
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---- frozen CLIP vision encoder ----
        self.clip = CLIPVisionModel.from_pretrained(clip_model)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        clip_hidden = self.clip.config.hidden_size  # 768 for ViT-L/14

        # ---- project CLIP tokens to d_model ----
        self.img_proj = nn.Linear(clip_hidden, d_model)

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
        pixel_values : [B, 3, H, W]
        returns      : [B, N_patches+1, d_model]
        """
        with torch.no_grad():
            outputs = self.clip(pixel_values=pixel_values)
        # last_hidden_state includes CLS + patch tokens
        img_tokens = outputs.last_hidden_state          # [B, N, 768]
        return self.img_proj(img_tokens)                # [B, N, d_model]

    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        pixel_values : [B, 3, H, W]
        Returns a dict mapping each LoRA key → Tensor of shape [B, *weight_shape].
        """
        B = pixel_values.shape[0]

        # Encode style image
        memory = self.encode_image(pixel_values)         # [B, N, d_model]

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
