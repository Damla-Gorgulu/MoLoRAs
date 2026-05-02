from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


STYLE_PREFIX = "unet.up_blocks.0.attentions.1"
PAIR_RE = re.compile(
    r"unet\.up_blocks\.0\.attentions\.1\.transformer_blocks\.(\d+)\."
    r"(attn[12])\.(to_q|to_k|to_v|to_out\.0)\.lora\.(down|up)\.weight"
)


@dataclass(frozen=True)
class PairSpec:
    base_key: str
    down_key: str
    up_key: str
    block_idx: int
    attn_idx: int
    matrix_idx: int
    d_in: int
    d_out: int
    rank: int


def discover_lora_paths(zoo_dir: str, limit: int | None = None) -> List[Path]:
    paths = sorted(Path(zoo_dir).glob("*/pytorch_lora_weights.safetensors"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No B-LoRA weights found under {zoo_dir}")
    return paths


def _parse_pair_base(key: str) -> Tuple[str, int, int, int] | None:
    m = PAIR_RE.fullmatch(key)
    if not m:
        return None
    block = int(m.group(1))
    attn = 0 if m.group(2) == "attn1" else 1
    matrix_map = {"to_q": 0, "to_k": 1, "to_v": 2, "to_out.0": 3}
    matrix = matrix_map[m.group(3)]
    base_key = key.replace(".down.weight", "").replace(".up.weight", "")
    return base_key, block, attn, matrix


def build_pair_specs(weights_path: Path) -> List[PairSpec]:
    sd = load_file(str(weights_path), device="cpu")
    style_keys = sorted(k for k in sd if STYLE_PREFIX in k)
    by_base: Dict[str, Dict[str, str]] = {}
    meta: Dict[str, Tuple[int, int, int]] = {}

    for key in style_keys:
        parsed = _parse_pair_base(key)
        if parsed is None:
            continue
        base_key, block, attn, matrix = parsed
        by_base.setdefault(base_key, {})
        if key.endswith(".down.weight"):
            by_base[base_key]["down"] = key
        elif key.endswith(".up.weight"):
            by_base[base_key]["up"] = key
        meta[base_key] = (block, attn, matrix)

    specs: List[PairSpec] = []
    for base_key in sorted(by_base):
        keys = by_base[base_key]
        if "down" not in keys or "up" not in keys:
            continue
        down = sd[keys["down"]]
        up = sd[keys["up"]]
        block, attn, matrix = meta[base_key]
        specs.append(
            PairSpec(
                base_key=base_key,
                down_key=keys["down"],
                up_key=keys["up"],
                block_idx=block,
                attn_idx=attn,
                matrix_idx=matrix,
                d_in=int(down.shape[1]),
                d_out=int(up.shape[0]),
                rank=int(down.shape[0]),
            )
        )
    if not specs:
        raise ValueError(f"No style LoRA up/down pairs found in {weights_path}")
    return specs


class StyleLoRAAutoencoderDataset(Dataset):
    """Loads style-only B-LoRA rank-pair tensors in a fixed schema."""

    def __init__(self, zoo_dir: str, limit: int | None = None):
        self.paths = discover_lora_paths(zoo_dir, limit=limit)
        self.style_names = [p.parent.name for p in self.paths]
        self.specs = build_pair_specs(self.paths[0])
        self.num_pairs = len(self.specs)
        self.rank = self.specs[0].rank
        if any(s.rank != self.rank for s in self.specs):
            raise ValueError("Mixed LoRA ranks are not supported in v1")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sd = load_file(str(self.paths[idx]), device="cpu")
        down1280 = torch.zeros(self.num_pairs, self.rank, 1280, dtype=torch.float32)
        down2048 = torch.zeros(self.num_pairs, self.rank, 2048, dtype=torch.float32)
        up = torch.zeros(self.num_pairs, self.rank, 1280, dtype=torch.float32)
        d_in = torch.empty(self.num_pairs, dtype=torch.long)

        for i, spec in enumerate(self.specs):
            down = sd[spec.down_key].float()
            up_i = sd[spec.up_key].float().transpose(0, 1)  # (rank, d_out)
            if spec.d_in == 1280:
                down1280[i] = down
            elif spec.d_in == 2048:
                down2048[i] = down
            else:
                raise ValueError(f"Unsupported d_in={spec.d_in}")
            up[i] = up_i
            d_in[i] = spec.d_in

        return {
            "down1280": down1280,
            "down2048": down2048,
            "up": up,
            "d_in": d_in,
            "idx": torch.tensor(idx, dtype=torch.long),
            "style_name": self.style_names[idx],
        }


def make_metadata_tensors(specs: List[PairSpec]) -> Dict[str, torch.Tensor]:
    return {
        "block_idx": torch.tensor([s.block_idx for s in specs], dtype=torch.long),
        "attn_idx": torch.tensor([s.attn_idx for s in specs], dtype=torch.long),
        "matrix_idx": torch.tensor([s.matrix_idx for s in specs], dtype=torch.long),
        "d_in": torch.tensor([s.d_in for s in specs], dtype=torch.long),
    }
