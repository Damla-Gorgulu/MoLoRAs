"""
utils.py — Shared utilities for LoRA Linear Composition Experiment.

Handles:
  - Config loading
  - LoRA ΔW extraction (B @ A)
  - Flatten / unflatten for regression
  - Direct weight-merge injection
  - Logging
  - Metrics computation

Key format convention (B-LoRA zoo):
  *.lora.up.weight   → B matrix (output_dim, rank) = (1280, 64)
  *.lora.down.weight → A matrix (rank, input_dim)  = (64, 1280) or (64, 2048)
  ΔW = up @ down = B @ A → (output_dim, input_dim)

IMPORTANT: Always work with ΔW = B @ A. Never operate on A, B separately
for linear combination — only ΔW is unique.
"""

import os
import json
import time
import datetime
import resource
from pathlib import Path
from collections import OrderedDict

import yaml
import torch
import numpy as np
from safetensors.torch import load_file


# ================================================================
# Config
# ================================================================

def load_config(config_path=None):
    """Load experiment config from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ================================================================
# LoRA pool discovery
# ================================================================

def discover_lora_pool(pool_dir):
    """
    Scan the B-LoRA zoo directory and return an ordered list of
    (style_index, style_name, safetensors_path) sorted alphabetically.

    Args:
        pool_dir: Path to bloras/ directory containing style_NNNN_Name/ subdirs.

    Returns:
        List of dicts: [{"index": int, "name": str, "dir_name": str, "path": str}, ...]
    """
    pool_dir = Path(pool_dir)
    styles = []
    for d in sorted(pool_dir.iterdir()):
        if not d.is_dir():
            continue
        sf_path = d / "pytorch_lora_weights.safetensors"
        if sf_path.exists():
            styles.append({
                "index": len(styles),
                "name": d.name,
                "dir_name": d.name,
                "path": str(sf_path),
            })
    return styles


# ================================================================
# ΔW extraction
# ================================================================

def _parse_adapter_key(key):
    """
    Parse a B-LoRA zoo tensor key into (adapter_name, component).

    Example:
      "unet.up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q.lora.up.weight"
      → adapter_name = "unet.up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q"
        component = "up"

    Returns:
        (adapter_name, "up"|"down") or (None, None) if not a LoRA key.
    """
    if key.endswith(".lora.up.weight"):
        adapter_name = key.replace(".lora.up.weight", "")
        return adapter_name, "up"
    elif key.endswith(".lora.down.weight"):
        adapter_name = key.replace(".lora.down.weight", "")
        return adapter_name, "down"
    return None, None


def parse_tensor_key_structure(adapter_name):
    """
    Parse an adapter name into structured components.

    Example:
      "unet.up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q"
      → {"block_idx": 3, "attn_type": "attn1", "projection": "to_q"}
    """
    parts = adapter_name.split(".")
    result = {}
    for i, part in enumerate(parts):
        if part == "transformer_blocks" and i + 1 < len(parts):
            result["block_idx"] = int(parts[i + 1])
        if part in ("attn1", "attn2"):
            result["attn_type"] = part
        if part.startswith("to_"):
            # Handle "to_out" -> join with next part if it's "0"
            if part == "to_out" and i + 1 < len(parts) and parts[i + 1] == "0":
                result["projection"] = "to_out.0"
            elif part != "to_out":
                result["projection"] = part
    return result


def load_lora_deltaw(safetensors_path, dtype=torch.float32):
    """
    Load a LoRA safetensors file and compute ΔW = up @ down for every adapter.

    Args:
        safetensors_path: Path to pytorch_lora_weights.safetensors
        dtype: Computation dtype (default: float32 for numerical stability)

    Returns:
        OrderedDict mapping adapter_name → ΔW tensor (output_dim × input_dim)
        Keys are sorted alphabetically for reproducible ordering.
    """
    raw = load_file(safetensors_path, device="cpu")

    # Group by adapter name
    adapters = {}
    for key, tensor in raw.items():
        adapter_name, component = _parse_adapter_key(key)
        if adapter_name is None:
            continue
        if adapter_name not in adapters:
            adapters[adapter_name] = {}
        adapters[adapter_name][component] = tensor.to(dtype)

    # Compute ΔW = up @ down (= B @ A) for each adapter
    deltaw = OrderedDict()
    for name in sorted(adapters.keys()):
        parts = adapters[name]
        if "up" not in parts or "down" not in parts:
            print(f"  WARNING: adapter '{name}' missing up or down, skipping.")
            continue
        up = parts["up"]     # B: (output_dim, rank)
        down = parts["down"]  # A: (rank, input_dim)
        deltaw[name] = torch.matmul(up, down)  # ΔW: (output_dim, input_dim)

    return deltaw


def get_adapter_key_order(deltaw_dict):
    """Return sorted list of adapter keys (canonical ordering for flatten/unflatten)."""
    return sorted(deltaw_dict.keys())


def get_adapter_shapes(deltaw_dict):
    """Return OrderedDict of adapter_name → shape."""
    return OrderedDict(
        (k, deltaw_dict[k].shape) for k in get_adapter_key_order(deltaw_dict)
    )


# ================================================================
# Flatten / Unflatten
# ================================================================

def flatten_deltaw(deltaw_dict, key_order=None):
    """
    Flatten all ΔW tensors into a single 1-D vector using canonical key order.

    Args:
        deltaw_dict: OrderedDict of adapter_name → ΔW tensor
        key_order: Optional explicit key ordering. If None, uses sorted keys.

    Returns:
        1-D torch.Tensor of length D (total number of parameters)
    """
    if key_order is None:
        key_order = get_adapter_key_order(deltaw_dict)
    parts = [deltaw_dict[k].flatten() for k in key_order]
    return torch.cat(parts)


def unflatten_deltaw(flat_vector, key_order, shapes):
    """
    Unflatten a 1-D vector back into per-adapter ΔW tensors.

    Args:
        flat_vector: 1-D tensor of length D
        key_order: List of adapter names (same order used for flattening)
        shapes: Dict or OrderedDict of adapter_name → (rows, cols)

    Returns:
        OrderedDict of adapter_name → ΔW tensor
    """
    deltaw = OrderedDict()
    offset = 0
    for name in key_order:
        shape = shapes[name] if isinstance(shapes, dict) else shapes
        numel = shape[0] * shape[1]
        deltaw[name] = flat_vector[offset:offset + numel].reshape(shape)
        offset += numel
    return deltaw


# ================================================================
# Tensor grouping
# ================================================================

def assign_tensor_groups(key_order, grouping_scheme):
    """
    Assign each adapter key to a group based on the grouping scheme.

    Args:
        key_order: List of adapter key names
        grouping_scheme: Dict describing groups. Each group has filter criteria.

    Returns:
        Dict mapping group_name → list of adapter key names
    """
    groups = {}
    for group_name, group_def in grouping_scheme.items():
        members = []
        for key in key_order:
            info = parse_tensor_key_structure(key)
            match = True

            if "pattern" in group_def:
                if group_def["pattern"] not in key:
                    match = False

            if "blocks" in group_def:
                if info.get("block_idx") not in group_def["blocks"]:
                    match = False

            if "attn" in group_def:
                if info.get("attn_type") != group_def["attn"]:
                    match = False

            if "projection" in group_def:
                proj = info.get("projection", "")
                if not proj.startswith(group_def["projection"]):
                    match = False

            if match:
                members.append(key)

        groups[group_name] = members
    return groups


# ================================================================
# Injection
# ================================================================

def inject_deltaw(pipe, deltaw_dict, alpha=2.0):
    """
    Inject reconstructed ΔW into a pipeline's UNet via direct weight merge.

    W_new = W_orig + alpha * ΔW

    Args:
        pipe: StableDiffusionXLPipeline
        deltaw_dict: OrderedDict of adapter_name → ΔW tensor
        alpha: LoRA scaling factor (default 2.0, validated in MoLoRAs roadmap §26)

    Returns:
        Dict of original weights (for reverting): adapter_name → original W tensor
    """
    unet = pipe.unet
    originals = {}

    for adapter_name, dw in deltaw_dict.items():
        # Navigate the UNet module tree to find the target parameter
        parts = adapter_name.split(".")
        # Strip leading "unet." if present
        if parts[0] == "unet":
            parts = parts[1:]

        module = unet
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)

        # module should now be e.g. a Linear layer
        param = module.weight
        originals[adapter_name] = param.data.clone()
        param.data += alpha * dw.to(param.device, param.dtype)

    return originals


def revert_injection(pipe, originals):
    """
    Revert a previous injection by restoring original weights.

    Args:
        pipe: StableDiffusionXLPipeline
        originals: Dict from inject_deltaw() return value
    """
    unet = pipe.unet
    for adapter_name, orig_weight in originals.items():
        parts = adapter_name.split(".")
        if parts[0] == "unet":
            parts = parts[1:]

        module = unet
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)

        module.weight.data.copy_(orig_weight)


# ================================================================
# Metrics
# ================================================================

def relative_reconstruction_error(x_target, x_reconstructed):
    """ε = ‖x_target − x_reconstructed‖ / ‖x_target‖"""
    diff_norm = torch.norm(x_target - x_reconstructed).item()
    target_norm = torch.norm(x_target).item()
    if target_norm < 1e-12:
        return float("inf")
    return diff_norm / target_norm


def cosine_similarity(x_target, x_reconstructed):
    """cos(x_target, x_reconstructed)"""
    dot = torch.dot(x_target.flatten(), x_reconstructed.flatten()).item()
    norm_a = torch.norm(x_target).item()
    norm_b = torch.norm(x_reconstructed).item()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_metrics(x_target, x_reconstructed, coefficients=None, threshold=1e-4):
    """
    Compute all standard metrics for a reconstruction.

    Returns:
        Dict with: relative_error, cosine_similarity, target_norm,
                   reconstructed_norm, sparsity, num_nonzero
    """
    metrics = {
        "relative_error": relative_reconstruction_error(x_target, x_reconstructed),
        "cosine_similarity": cosine_similarity(x_target, x_reconstructed),
        "target_norm": torch.norm(x_target).item(),
        "reconstructed_norm": torch.norm(x_reconstructed).item(),
    }
    if coefficients is not None:
        w = np.asarray(coefficients).flatten()
        nonzero = np.sum(np.abs(w) > threshold)
        metrics["num_nonzero"] = int(nonzero)
        metrics["sparsity"] = 1.0 - nonzero / len(w)
        # Top-k energy
        sorted_abs = np.sort(np.abs(w))[::-1]
        total = np.sum(sorted_abs)
        if total > 0:
            cumulative = np.cumsum(sorted_abs) / total
            metrics["top5_energy"] = float(cumulative[min(4, len(cumulative) - 1)])
            metrics["top10_energy"] = float(cumulative[min(9, len(cumulative) - 1)])
            metrics["top20_energy"] = float(cumulative[min(19, len(cumulative) - 1)])
    return metrics


# ================================================================
# Logging
# ================================================================

def get_log_path(config):
    """Return path to the master experiment log file."""
    return Path(config["experiment_dir"]) / config["log_file"]


def log_entry(config, entry):
    """
    Append a timestamped entry to the experiment log (JSON lines format).

    Args:
        config: Experiment config dict
        entry: Dict to log
    """
    log_path = get_log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry["timestamp"] = datetime.datetime.now().isoformat()
    entry["peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_json(data, path):
    """Save a dict/list as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ================================================================
# Image generation helpers
# ================================================================

def load_pipeline(config, device="cuda"):
    """Load SDXL pipeline with fp16 and VAE fix."""
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        config["vae_model_id"], torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["base_model_id"],
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)
    return pipe


def generate_image(pipe, config, generator=None):
    """Generate one image using the standard prompt/seed from config."""
    if generator is None:
        generator = torch.Generator(device=pipe.device).manual_seed(config["seed"])

    image = pipe(
        prompt=config["prompt"],
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        generator=generator,
    ).images[0]
    return image
