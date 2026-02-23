#!/usr/bin/env python3
"""
Diagnostic script to debug LoRA injection.
Tests:
  1. Key format: pipeline.lora_state_dict() vs raw safetensors
  2. Whether load_lora_into_unet actually modifies UNet outputs
  3. Synthesized LoRA magnitudes vs real LoRA
  4. S1 vs S2 attention comparison
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from safetensors.torch import load_file

# ── 1. Key format comparison ─────────────────────────────────
print("=" * 60)
print("TEST 1: Key format comparison")
print("=" * 60)

zoo_dir = "/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
first_style = sorted(os.listdir(zoo_dir))[0]
real_path = os.path.join(zoo_dir, first_style, "pytorch_lora_weights.safetensors")

# Raw safetensors keys
raw_sd = load_file(real_path)
raw_style_keys = sorted([k for k in raw_sd if "up_blocks.0.attentions.1" in k])
print(f"\nRaw safetensors keys (first 5 of {len(raw_style_keys)}):")
for k in raw_style_keys[:5]:
    print(f"  RAW: {k}")

# Pipeline.lora_state_dict keys
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
print("\nLoading pipeline for key comparison...")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

pipe_sd, _ = pipeline.lora_state_dict(real_path)
pipe_style_keys = sorted([k for k in pipe_sd if "up_blocks.0.attentions.1" in k])
print(f"\npipeline.lora_state_dict keys (first 5 of {len(pipe_style_keys)}):")
for k in pipe_style_keys[:5]:
    print(f"  PIPE: {k}")

# Check overlap
raw_set = set(raw_style_keys)
pipe_set = set(pipe_style_keys)
print(f"\nKeys identical? {raw_set == pipe_set}")
print(f"Raw-only keys: {raw_set - pipe_set}")
print(f"Pipe-only keys: {pipe_set - raw_set}")

# ── 2. Test actual injection via load_lora_into_unet ─────────
print("\n" + "=" * 60)
print("TEST 2: Does load_lora_into_unet actually modify UNet?")
print("=" * 60)

# Get UNet output BEFORE LoRA
test_latent = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
test_timestep = torch.tensor([500], device="cuda").long()
test_encoder_hidden = torch.randn(1, 77, 2048, device="cuda", dtype=torch.float16)
test_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float16)
test_text_embeds = torch.randn(1, 1280, device="cuda", dtype=torch.float16)

with torch.no_grad():
    out_before = pipeline.unet(
        test_latent, test_timestep,
        encoder_hidden_states=test_encoder_hidden,
        added_cond_kwargs={"text_embeds": test_text_embeds, "time_ids": test_time_ids},
    ).sample.clone()
print(f"UNet output BEFORE LoRA: mean={out_before.mean():.6f}, std={out_before.std():.6f}")

# Inject real LoRA (the B-LoRA-fresh way, via lora_state_dict)
from lora_attention.utils.lora_inject import inject_lora, unload_lora
from blora_utils import BLOCKS, filter_lora, scale_lora

# Method A: B-LoRA-fresh way
style_sd = filter_lora(pipe_sd, BLOCKS['style'])
style_sd = scale_lora(style_sd, 1.0)
pipeline.load_lora_into_unet(style_sd, None, pipeline.unet)

with torch.no_grad():
    out_blora_way = pipeline.unet(
        test_latent, test_timestep,
        encoder_hidden_states=test_encoder_hidden,
        added_cond_kwargs={"text_embeds": test_text_embeds, "time_ids": test_time_ids},
    ).sample.clone()
print(f"UNet output AFTER  LoRA (B-LoRA-fresh way): mean={out_blora_way.mean():.6f}, std={out_blora_way.std():.6f}")
diff_blora = (out_blora_way - out_before).abs().mean().item()
print(f"Diff (B-LoRA-fresh inject): {diff_blora:.6f}")

unload_lora(pipeline)

# Method B: Our inject_lora way (raw safetensors keys)
raw_style = {k: v for k, v in raw_sd.items() if "up_blocks.0.attentions.1" in k}
inject_lora(pipeline, raw_style, style_alpha=1.0)

with torch.no_grad():
    out_our_way = pipeline.unet(
        test_latent, test_timestep,
        encoder_hidden_states=test_encoder_hidden,
        added_cond_kwargs={"text_embeds": test_text_embeds, "time_ids": test_time_ids},
    ).sample.clone()
print(f"UNet output AFTER  LoRA (our inject_lora way): mean={out_our_way.mean():.6f}, std={out_our_way.std():.6f}")
diff_ours = (out_our_way - out_before).abs().mean().item()
print(f"Diff (our inject): {diff_ours:.6f}")

unload_lora(pipeline)

# ── 3. Load MoELoRA and test synth weights ────────────────────
print("\n" + "=" * 60)
print("TEST 3: Synthesized LoRA magnitude check")
print("=" * 60)

from lora_attention.models.lora_pool import LoRAPool
from lora_attention.models.moe_lora import MoELoRA

pool = LoRAPool(zoo_dir=zoo_dir, cache_dir="/scratch/eyavuz21/lora_attention")

model = MoELoRA(pool=pool).to("cuda")

# S1 checkpoint
ckpt_s1 = torch.load("/scratch/eyavuz21/lora_attention/stage1/latest.pt", map_location="cpu", weights_only=False)
model.routing_mlp.load_state_dict(ckpt_s1["routing_mlp_state_dict"])
model.routing_mlp.eval()

# Encode baroque style image
img = Image.open("/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images/style_0000_Baroque/style_0000_Baroque.jpg").convert("RGB")
q = model.encode_image(img, torch.device("cuda"))
print(f"CLIP embedding: shape={q.shape}, norm={q.norm().item():.4f}")

# Route with full pool (S1)
with torch.no_grad():
    A_s1, synth_s1 = model.forward(q, list(range(pool.num_experts)))

print(f"\nS1 Attention stats:")
print(f"  A shape: {A_s1.shape}")
print(f"  A max per expert (top-5): {A_s1.mean(1).topk(5)}")
print(f"  A[0] (Baroque expert, first 10 ranks): {A_s1[0, :10]}")

# Check synth weights magnitude
k0 = list(synth_s1.keys())[0]
print(f"\nSynth weight [{k0}]:")
print(f"  shape={synth_s1[k0].shape}, |mean|={synth_s1[k0].mean().abs():.6f}, std={synth_s1[k0].std():.6f}, max={synth_s1[k0].abs().max():.6f}")

# Compare with real Baroque LoRA weight
real_baroque = load_file(os.path.join(zoo_dir, first_style, "pytorch_lora_weights.safetensors"))
if k0 in real_baroque:
    real_w = real_baroque[k0]
    print(f"Real weight [{k0}]:")
    print(f"  shape={real_w.shape}, |mean|={real_w.mean().abs():.6f}, std={real_w.std():.6f}, max={real_w.abs().max():.6f}")
    cos_sim = torch.nn.functional.cosine_similarity(synth_s1[k0].cpu().flatten(), real_w.flatten(), dim=0)
    print(f"  Cosine similarity synth vs real: {cos_sim:.4f}")

# Inject synth and check UNet diff
synth_cpu = {k: v.detach().cpu() for k, v in synth_s1.items()}
inject_lora(pipeline, synth_cpu, style_alpha=1.0)

with torch.no_grad():
    out_synth = pipeline.unet(
        test_latent, test_timestep,
        encoder_hidden_states=test_encoder_hidden,
        added_cond_kwargs={"text_embeds": test_text_embeds, "time_ids": test_time_ids},
    ).sample.clone()
diff_synth = (out_synth - out_before).abs().mean().item()
print(f"\nDiff (synth inject vs baseline): {diff_synth:.6f}")
print(f"Diff (synth inject vs real inject): {(out_synth - out_our_way).abs().mean().item():.6f}")

unload_lora(pipeline)

# ── 4. S2 checkpoint comparison ───────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: S2 attention comparison")
print("=" * 60)

ckpt_s2 = torch.load("/scratch/eyavuz21/lora_attention/stage2/latest.pt", map_location="cpu", weights_only=False)
model.routing_mlp.load_state_dict(ckpt_s2["routing_mlp_state_dict"])
model.routing_mlp.eval()

with torch.no_grad():
    A_s2, synth_s2 = model.forward(q, list(range(pool.num_experts)))

print(f"S2 Attention stats:")
print(f"  A max per expert (top-5): {A_s2.mean(1).topk(5)}")
print(f"  A[0] (Baroque expert, first 10 ranks): {A_s2[0, :10]}")

# Entropy comparison
import torch.nn.functional as Fn
s1_entropy = -(A_s1 * A_s1.clamp(min=1e-8).log()).sum(0).mean().item()
s2_entropy = -(A_s2 * A_s2.clamp(min=1e-8).log()).sum(0).mean().item()
print(f"\nS1 avg entropy per rank: {s1_entropy:.4f}")
print(f"S2 avg entropy per rank: {s2_entropy:.4f}")
print(f"Max entropy (uniform): {torch.log(torch.tensor(float(pool.num_experts))).item():.4f}")

# Key finding: inject S2 synth and check diff
synth_s2_cpu = {k: v.detach().cpu() for k, v in synth_s2.items()}
inject_lora(pipeline, synth_s2_cpu, style_alpha=1.0)
with torch.no_grad():
    out_synth_s2 = pipeline.unet(
        test_latent, test_timestep,
        encoder_hidden_states=test_encoder_hidden,
        added_cond_kwargs={"text_embeds": test_text_embeds, "time_ids": test_time_ids},
    ).sample.clone()
diff_synth_s2 = (out_synth_s2 - out_before).abs().mean().item()
print(f"Diff (S2 synth inject vs baseline): {diff_synth_s2:.6f}")

unload_lora(pipeline)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
