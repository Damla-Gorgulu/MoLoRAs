from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoRAAutoencoder(nn.Module):
    """Tokenizes style B-LoRA rank slices, compresses to z, and reconstructs tensors."""

    def __init__(
        self,
        num_pairs: int,
        rank: int = 64,
        latent_dim: int = 65536,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_pairs = num_pairs
        self.rank = rank
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.down1280_proj = nn.Linear(1280, d_model)
        self.down2048_proj = nn.Linear(2048, d_model)
        self.up_proj = nn.Linear(1280, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.block_emb = nn.Embedding(16, d_model)
        self.attn_emb = nn.Embedding(2, d_model)
        self.matrix_emb = nn.Embedding(4, d_model)
        self.rank_emb = nn.Embedding(rank, d_model)
        self.pair_emb = nn.Embedding(num_pairs, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.from_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.down1280_head = nn.Linear(d_model, 1280)
        self.down2048_head = nn.Linear(d_model, 2048)
        self.up_head = nn.Linear(d_model, 1280)

        nn.init.normal_(self.cls, std=0.02)

        for head in [self.down1280_head, self.down2048_head, self.up_head]:
            nn.init.normal_(head.weight, std=1e-5)
            nn.init.zeros_(head.bias)

    def _metadata_tokens(self, meta: Dict[str, torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
        pair_ids = torch.arange(self.num_pairs, device=device)
        rank_ids = torch.arange(self.rank, device=device)
        base = (
            self.pair_emb(pair_ids)[:, None, :]
            + self.rank_emb(rank_ids)[None, :, :]
            + self.block_emb(meta["block_idx"].to(device))[:, None, :]
            + self.attn_emb(meta["attn_idx"].to(device))[:, None, :]
            + self.matrix_emb(meta["matrix_idx"].to(device))[:, None, :]
        )
        base = base.reshape(1, self.num_pairs * self.rank, self.d_model)
        return base.expand(batch_size, -1, -1)

    def encode(self, batch: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        down1280 = batch["down1280"]
        down2048 = batch["down2048"]
        up = batch["up"]
        d_in = batch["d_in"]
        device = up.device
        bsz = up.shape[0]

        h1280 = self.down1280_proj(down1280)
        h2048 = self.down2048_proj(down2048)
        mask2048 = (d_in == 2048).to(up.dtype).view(bsz, self.num_pairs, 1, 1)
        h_down = h1280 * (1.0 - mask2048) + h2048 * mask2048
        h = self.fuse(torch.cat([h_down, self.up_proj(up)], dim=-1))
        h = h.reshape(bsz, self.num_pairs * self.rank, self.d_model)
        h = h + self._metadata_tokens(meta, device, bsz)

        cls = self.cls.expand(bsz, -1, -1)
        encoded = self.encoder(torch.cat([cls, h], dim=1))
        global_token = encoded[:, :1]
        rank_tokens = encoded[:, 1:]
        pair_tokens = rank_tokens.reshape(bsz, self.num_pairs, self.rank, self.d_model).mean(dim=2)
        return self.to_latent(torch.cat([global_token, pair_tokens], dim=1))

    def decode(self, z: torch.Tensor, meta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsz = z.shape[0]
        device = z.device
        queries = self._metadata_tokens(meta, device, bsz)
        memory = self.from_latent(z)
        h = self.decoder(queries, memory)
        h = self.out_norm(h).reshape(bsz, self.num_pairs, self.rank, self.d_model)
        return {
            "down1280": self.down1280_head(h),
            "down2048": self.down2048_head(h),
            "up": self.up_head(h),
        }

    def forward(self, batch: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self.encode(batch, meta)
        recon = self.decode(z, meta)
        recon["z"] = z
        return recon


def reconstruction_losses(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    tensor_weight: float = 1.0,
    delta_weight: float = 0.1,
    cos_weight: float = 0.01,
    rel_weight: float = 1.0,
    norm_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    d_in = target["d_in"]
    mask2048 = (d_in == 2048).float().view(d_in.shape[0], d_in.shape[1], 1, 1)
    mask1280 = 1.0 - mask2048

    down1280_denom = (mask1280.sum() * pred["down1280"].shape[-2] * pred["down1280"].shape[-1]).clamp_min(1.0)
    down2048_denom = (mask2048.sum() * pred["down2048"].shape[-2] * pred["down2048"].shape[-1]).clamp_min(1.0)
    tensor_loss = (
        F.mse_loss(pred["up"], target["up"])
        + (F.mse_loss(pred["down1280"], target["down1280"], reduction="none") * mask1280).sum() / down1280_denom
        + (F.mse_loss(pred["down2048"], target["down2048"], reduction="none") * mask2048).sum() / down2048_denom
    )

    up_pred = pred["up"]
    up_tgt = target["up"]
    delta1280_pred = torch.matmul(up_pred.transpose(-1, -2).float(), pred["down1280"].float())
    delta1280_tgt = torch.matmul(up_tgt.transpose(-1, -2).float(), target["down1280"].float())
    delta2048_pred = torch.matmul(up_pred.transpose(-1, -2).float(), pred["down2048"].float())
    delta2048_tgt = torch.matmul(up_tgt.transpose(-1, -2).float(), target["down2048"].float())
    delta1280_loss = (F.mse_loss(delta1280_pred, delta1280_tgt, reduction="none") * mask1280).sum() / (mask1280.sum() * delta1280_pred.shape[-2] * delta1280_pred.shape[-1]).clamp_min(1.0)
    delta2048_loss = (F.mse_loss(delta2048_pred, delta2048_tgt, reduction="none") * mask2048).sum() / (mask2048.sum() * delta2048_pred.shape[-2] * delta2048_pred.shape[-1]).clamp_min(1.0)
    delta_loss = delta1280_loss + delta2048_loss

    delta1280_rel = ((delta1280_pred - delta1280_tgt).pow(2) * mask1280).sum() / ((delta1280_tgt.pow(2) * mask1280).sum() + 1e-8)
    delta2048_rel = ((delta2048_pred - delta2048_tgt).pow(2) * mask2048).sum() / ((delta2048_tgt.pow(2) * mask2048).sum() + 1e-8)
    delta_rel = delta1280_rel + delta2048_rel
    delta_rel_loss = torch.log1p(delta_rel)

    flat_pred = torch.cat([
        pred["up"].flatten(1),
        (pred["down1280"] * mask1280).flatten(1),
        (pred["down2048"] * mask2048).flatten(1),
    ], dim=1)
    flat_tgt = torch.cat([
        target["up"].flatten(1),
        (target["down1280"] * mask1280).flatten(1),
        (target["down2048"] * mask2048).flatten(1),
    ], dim=1)
    cos = F.cosine_similarity(flat_pred, flat_tgt, dim=1).mean()
    cos_loss = 1.0 - cos
    rel = (flat_pred - flat_tgt).norm(dim=1).mean() / (flat_tgt.norm(dim=1).mean() + 1e-8)
    norm_ratio = flat_pred.norm(dim=1) / (flat_tgt.norm(dim=1) + 1e-8)
    norm_loss = (norm_ratio.clamp_min(1e-8).log().pow(2)).mean()
    loss = (
        tensor_weight * tensor_loss
        + delta_weight * delta_rel_loss
        + cos_weight * cos_loss
        + rel_weight * rel
        + norm_weight * norm_loss
    )
    return {
        "loss": loss,
        "tensor_mse": tensor_loss.detach(),
        "delta_mse": delta_loss.detach(),
        "delta_rel": delta_rel.detach(),
        "delta_rel_loss": delta_rel_loss.detach(),
        "cos": cos.detach(),
        "rel": rel.detach(),
        "norm_ratio": norm_ratio.mean().detach(),
        "norm_loss": norm_loss.detach(),
    }
