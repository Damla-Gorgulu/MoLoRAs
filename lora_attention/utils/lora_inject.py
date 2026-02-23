"""
LoRA Injection Utilities for SDXL Pipeline.

Provides functions to inject a synthesised (or real) style LoRA state dict
into a frozen StableDiffusionXLPipeline and cleanly unload it afterwards.

The synthesised state dict keys follow the B-LoRA naming convention:
  "unet.up_blocks.0.attentions.1.<...>.lora.down.weight"
  "unet.up_blocks.0.attentions.1.<...>.lora.up.weight"

Compatible with the diffusers version used in B-LoRA-fresh
(old API: pipeline.load_lora_into_unet / pipeline.lora_state_dict).
"""

import contextlib
from typing import Dict, List, Optional

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

from ..models.lora_pool import STYLE_BLOCK_PREFIX

# Keys for the content block (used when optionally combining with content LoRA)
CONTENT_BLOCK_PREFIX = "unet.up_blocks.0.attentions.0"


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def inject_lora(
    pipeline: StableDiffusionXLPipeline,
    style_state_dict: Dict[str, torch.Tensor],
    style_alpha: float = 1.0,
    content_lora_path: Optional[str] = None,
    content_alpha: float = 1.0,
) -> None:
    """
    Inject style (and optionally content) LoRA weights into the pipeline UNet.

    Args:
        pipeline:          Frozen SDXL pipeline.
        style_state_dict:  Synthesised style-block state dict
                           (keys: unet.up_blocks.0.attentions.1.*).
        style_alpha:       Scale factor applied to style LoRA weights.
        content_lora_path: Optional path to a content B-LoRA .safetensors file.
                           If supplied, its content-block tensors are merged in.
        content_alpha:     Scale factor for content LoRA.
    """
    # Scale style weights
    merged: Dict[str, torch.Tensor] = {
        k: v * style_alpha for k, v in style_state_dict.items()
    }

    # Optionally merge content LoRA
    if content_lora_path is not None:
        content_sd = load_file(content_lora_path)
        content_sd = {
            k: v * content_alpha
            for k, v in content_sd.items()
            if CONTENT_BLOCK_PREFIX in k
        }
        merged.update(content_sd)

    if len(merged) == 0:
        return

    pipeline.load_lora_into_unet(merged, None, pipeline.unet)


def unload_lora(pipeline: StableDiffusionXLPipeline) -> None:
    """
    Remove any injected LoRA adapters from the pipeline UNet.

    Attempts the diffusers 0.25.x compatible API first (unet.set_default_attn_processor),
    then falls back to manual weight restoration via attention processor reset.
    """
    try:
        # Old diffusers approach used in B-LoRA-fresh
        from diffusers.loaders import LoraLoaderMixin
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
            return
    except Exception:
        pass

    # Fallback: reset attention processors (removes LoRA hooks)
    try:
        pipeline.unet.set_default_attn_processor()
    except Exception:
        pass


@contextlib.contextmanager
def LoRAInjectionContext(
    pipeline: StableDiffusionXLPipeline,
    style_state_dict: Dict[str, torch.Tensor],
    style_alpha: float = 1.0,
    content_lora_path: Optional[str] = None,
    content_alpha: float = 1.0,
):
    """
    Context manager: injects a LoRA on enter and unloads on exit.

    Usage:
        with LoRAInjectionContext(pipeline, synth_lora) as pipe:
            out = pipe("a painting in Impressionism style", num_inference_steps=20)
    """
    inject_lora(
        pipeline=pipeline,
        style_state_dict=style_state_dict,
        style_alpha=style_alpha,
        content_lora_path=content_lora_path,
        content_alpha=content_alpha,
    )
    try:
        yield pipeline
    finally:
        unload_lora(pipeline)


# ──────────────────────────────────────────────────────────────
# Diffusion loss helper (used in Stage 2 training)
# ──────────────────────────────────────────────────────────────
def compute_ldm_loss(
    pipeline: StableDiffusionXLPipeline,
    style_state_dict: Dict[str, torch.Tensor],
    image: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    noise_scheduler,
    style_alpha: float = 1.0,
    weight_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Compute the standard LDM noise prediction loss with a synthesised LoRA.

    The LoRA is injected, a forward pass is run, then unloaded.
    This function differentiates through inject_lora (the synthesis step),
    so gradients flow back to the RoutingMLP via style_state_dict.

    Args:
        pipeline:              Frozen SDXL pipeline (unet must be in train mode
                               for gradient tracking of the injected weights).
        style_state_dict:      Synthesised LoRA state dict (w/ gradients).
        image:                 Batch of latents or pixel images. Shape: (B, C, H, W).
        prompt_embeds:         Text embeddings. Shape: (B, seq, d).
        pooled_prompt_embeds:  Pooled text embeddings. Shape: (B, d).
        noise_scheduler:       DDPMScheduler instance.
        style_alpha:           LoRA scale.
        weight_dtype:          Compute dtype.

    Returns:
        loss: Scalar tensor (mean MSE between predicted and true noise).
    """
    vae = pipeline.vae
    unet = pipeline.unet
    device = next(unet.parameters()).device

    # Encode image to latent space
    with torch.no_grad():
        image = image.to(device=device, dtype=weight_dtype)
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample noise and timestep
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (bsz,), device=device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Build time_ids for SDXL
    add_time_ids = _build_time_ids(
        original_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        target_size=(1024, 1024),
        dtype=weight_dtype,
        device=device,
        batch_size=bsz,
    )
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds.to(device=device, dtype=weight_dtype),
        "time_ids": add_time_ids,
    }

    # Inject LoRA (weights with gradients) into UNet
    scaled_sd = {k: v * style_alpha for k, v in style_state_dict.items()}
    pipeline.load_lora_into_unet(scaled_sd, None, unet)

    # Forward pass through frozen UNet (LoRA weights carry grads)
    noise_pred = unet(
        noisy_latents.to(dtype=weight_dtype),
        timesteps,
        encoder_hidden_states=prompt_embeds.to(device=device, dtype=weight_dtype),
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # Unload LoRA after forward
    unload_lora(pipeline)

    # MSE on noise
    loss = torch.nn.functional.mse_loss(
        noise_pred.float(), noise.float(), reduction="mean"
    )
    return loss


def _build_time_ids(
    original_size,
    crops_coords_top_left,
    target_size,
    dtype,
    device,
    batch_size: int,
) -> torch.Tensor:
    """Build SDXL additional time conditioning ids."""
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids.repeat(batch_size, 1)


# ──────────────────────────────────────────────────────────────
# Gradient-compatible LoRA hook injection (used in Stage 2)
# ──────────────────────────────────────────────────────────────
def apply_lora_hooks_with_grad(
    unet: torch.nn.Module,
    style_state_dict: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on UNet linear layers that add the LoRA delta
    INSIDE the forward pass while maintaining the computation graph.

    This allows gradients to flow back through the synthesised LoRA weights
    to the RoutingMLP during Stage 2 training.

    Hook formula (LoRA):
        output = W_frozen(x) + alpha * (x @ W_down.T) @ W_up.T
        where W_down: (rank, in_dim), W_up: (out_dim, rank)

    Args:
        unet:              UNet2DConditionModel (frozen base weights).
        style_state_dict:  Dict of synthesised LoRA tensors (with .grad_fn).
        alpha:             LoRA scale factor.

    Returns:
        List of hook handles — call handle.remove() to unload.
    """
    # Group keys by layer path  →  {layer_path: {"down": Tensor, "up": Tensor}}
    layer_loras: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in style_state_dict.items():
        # Key format: "unet.up_blocks.0....to_k.lora.down.weight"
        # Strip leading "unet." and trailing ".lora.{down|up}.weight"
        if "lora.down.weight" in key:
            layer_path = key.replace("unet.", "", 1).replace(".lora.down.weight", "")
            layer_loras.setdefault(layer_path, {})["down"] = tensor
        elif "lora.up.weight" in key:
            layer_path = key.replace("unet.", "", 1).replace(".lora.up.weight", "")
            layer_loras.setdefault(layer_path, {})["up"] = tensor

    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for layer_path, lora_weights in layer_loras.items():
        if "down" not in lora_weights or "up" not in lora_weights:
            continue
        W_down = lora_weights["down"]  # (rank, in_dim)
        W_up = lora_weights["up"]      # (out_dim, rank)

        try:
            layer = unet.get_submodule(layer_path)
        except AttributeError:
            continue  # Layer not found (e.g. missing in this UNet version)

        # Closure captures W_down, W_up, alpha for this layer
        def make_hook(wd, wu, a):
            def hook_fn(module, inp, output):
                # inp[0]: (batch, seq, in_dim) or (batch, in_dim)
                x = inp[0]
                # LoRA delta: (x @ wd.T) @ wu.T
                delta = (x @ wd.T.to(x.dtype)) @ wu.T.to(x.dtype)
                return output + a * delta
            return hook_fn

        h = layer.register_forward_hook(make_hook(W_down, W_up, alpha))
        hooks.append(h)

    return hooks


def remove_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove all hooks from the list returned by apply_lora_hooks_with_grad."""
    for h in hooks:
        h.remove()
