#!/usr/bin/env python3
"""
GPU diagnostic: is load_lora_into_unet actually applying LoRA weights?

Tests:
  1. Key format: raw safetensors vs pipeline.lora_state_dict()
  2. USE_PEFT_BACKEND flag
  3. Base B-LoRA (real weights) → does pixel output differ from vanilla SDXL?
  4. Synthesised LoRA (uniform-attention average) → does it differ from vanilla?
  5. Hook-based injection (apply_lora_hooks_with_grad) as fallback path
  6. Pixel-level diff statistics saved as image grid

Results saved to:  $OUT_DIR/diagnose_lora_inject/
  - vanilla_vs_real_blora.jpg  (2-column: vanilla | B-LoRA Baroque)
  - vanilla_vs_synth_lora.jpg  (2-column: vanilla | averaged synth)
  - vanilla_vs_hook_inject.jpg (2-column: vanilla | hook-based inject)
  - report.txt                 (numbers: diff norms, processor counts, etc.)
"""

import sys, math, shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "B-LoRA-fresh" / "B-LoRA"))

from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.utils import USE_PEFT_BACKEND

from lora_attention.utils.lora_inject import inject_lora, unload_lora, apply_lora_hooks_with_grad, remove_hooks
from lora_attention.models.lora_pool import LoRAPool

# ── Config ────────────────────────────────────────────────────────────────────
ZOO_DIR    = "/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR  = "/scratch/eyavuz21/lora_attention"
CKPT_S1    = "/scratch/eyavuz21/lora_attention/stage1_v2/latest.pt"
STYLE_IMG  = "/home/eyavuz21/datasets/wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
BLORA_PATH = f"{ZOO_DIR}/style_0000_Baroque/pytorch_lora_weights.safetensors"
PROMPT     = "A Baroque painting"
OUT_DIR    = Path("/scratch/eyavuz21/lora_attention/diagnose_lora_inject")
SEED       = 42
STEPS      = 20
GUIDANCE   = 7.5
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)
report_lines = []

def log(msg):
    print(msg, flush=True)
    report_lines.append(msg)

def generate(pipe, prompt, seed=SEED, steps=STEPS, guidance=GUIDANCE, lora_scale=1.0):
    gen = torch.Generator(device=device).manual_seed(seed)
    img = pipe(
        prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]
    return img

def img_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute pixel difference (0-255 range)."""
    a_np = np.array(a).astype(float)
    b_np = np.array(b).astype(float)
    return float(np.abs(a_np - b_np).mean())

def save_pair(tag: str, img_a: Image.Image, img_b: Image.Image,
              label_a: str, label_b: str, diff: float):
    """Save a side-by-side comparison with labels and diff score."""
    W, H = img_a.size
    canvas = Image.new("RGB", (W * 2 + 20, H + 40), (40, 40, 40))
    canvas.paste(img_a, (0, 40))
    canvas.paste(img_b, (W + 20, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4),      f"{label_a}", fill=(200, 200, 200))
    draw.text((W + 24, 4), f"{label_b}  |  MAE diff={diff:.2f}", fill=(200, 200, 200))
    path = OUT_DIR / f"{tag}.jpg"
    canvas.save(path)
    log(f"  [saved] {path.name}")
    return path

# ── Load pipeline ─────────────────────────────────────────────────────────────
log("="*60)
log(f"USE_PEFT_BACKEND = {USE_PEFT_BACKEND}")
log(f"Device: {device}")
log("="*60)

log("\n[1] Loading SDXL pipeline...")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16,
).to(device)

# ── Key format audit ─────────────────────────────────────────────────────────
log("\n[2] Key format audit...")
raw_sd = load_file(BLORA_PATH)
via_api_sd, _ = pipe.lora_state_dict(BLORA_PATH)
style_raw  = {k: v for k, v in raw_sd.items() if "attentions.1" in k}
style_api  = {k: v for k, v in via_api_sd.items() if "attentions.1" in k}
log(f"  Raw safetensors keys (attentions.1): {len(style_raw)}")
log(f"  Via lora_state_dict  (attentions.1): {len(style_api)}")
if style_raw and style_api:
    r0 = sorted(style_raw.keys())[0]
    a0 = sorted(style_api.keys())[0]
    log(f"  Raw key[0]:     {r0}")
    log(f"  API key[0]:     {a0}")
    log(f"  Keys match:     {r0 == a0}")

# ── Test 1: Vanilla SDXL ─────────────────────────────────────────────────────
log("\n[3] Generating vanilla SDXL image (no LoRA)...")
img_vanilla = generate(pipe, PROMPT)
img_vanilla.save(OUT_DIR / "vanilla.jpg")
log(f"  [saved] vanilla.jpg")

# ── Test 2: Real B-LoRA via load_lora_into_unet (our current path) ───────────
log("\n[4] Test: load_lora_into_unet with raw keys (current inject_lora path)...")
pipe.load_lora_into_unet(style_raw, None, pipe.unet)
procs_after = {k: v for k, v in pipe.unet.attn_processors.items() if "LoRA" in type(v).__name__}
log(f"  LoRA processors installed: {len(procs_after)} / {len(pipe.unet.attn_processors)}")
img_inject_raw = generate(pipe, PROMPT)
diff_raw = img_diff(img_vanilla, img_inject_raw)
log(f"  MAE diff vs vanilla: {diff_raw:.4f}  ({'ACTIVE' if diff_raw > 2.0 else 'DEAD — no effect'})")
save_pair("vanilla_vs_inject_raw", img_vanilla, img_inject_raw, "Vanilla SDXL", "inject_lora(raw keys)", diff_raw)
unload_lora(pipe)

# ── Test 3: Real B-LoRA via pipeline.lora_state_dict (B-LoRA's original path) ─
log("\n[5] Test: load_lora_into_unet with lora_state_dict keys (B-LoRA original path)...")
pipe.load_lora_into_unet(style_api, None, pipe.unet)
procs_api = {k: v for k, v in pipe.unet.attn_processors.items() if "LoRA" in type(v).__name__}
log(f"  LoRA processors installed: {len(procs_api)} / {len(pipe.unet.attn_processors)}")
img_inject_api = generate(pipe, PROMPT)
diff_api = img_diff(img_vanilla, img_inject_api)
log(f"  MAE diff vs vanilla: {diff_api:.4f}  ({'ACTIVE' if diff_api > 2.0 else 'DEAD — no effect'})")
save_pair("vanilla_vs_inject_api", img_vanilla, img_inject_api, "Vanilla SDXL", "inject via lora_state_dict keys", diff_api)
unload_lora(pipe)

# ── Test 4: Hook-based injection (apply_lora_hooks_with_grad) ─────────────────
log("\n[6] Test: hook-based injection (apply_lora_hooks_with_grad)...")
hook_sd = {k: v.float().to(device) for k, v in style_raw.items()}
hooks = apply_lora_hooks_with_grad(pipe.unet, hook_sd, alpha=1.0)
log(f"  Hooks registered: {len(hooks)}")
img_hooks = generate(pipe, PROMPT)
diff_hooks = img_diff(img_vanilla, img_hooks)
log(f"  MAE diff vs vanilla: {diff_hooks:.4f}  ({'ACTIVE' if diff_hooks > 2.0 else 'DEAD — no effect'})")
save_pair("vanilla_vs_hooks", img_vanilla, img_hooks, "Vanilla SDXL", "hook-based injection", diff_hooks)
remove_hooks(hooks)

# ── Test 5: Synthesised LoRA from MoELoRAv2 (S1 checkpoint) ───────────────────
log("\n[7] Test: MoELoRAv2 synthesised LoRA (S1 checkpoint, Baroque query)...")
try:
    from lora_attention.models.lora_pool import LoRAPool
    from lora_attention.models.moe_lora_v2 import MoELoRAv2

    pool = LoRAPool(zoo_dir=ZOO_DIR, cache_dir=CACHE_DIR)
    model = MoELoRAv2(pool=pool, clip_model_id="openai/clip-vit-base-patch32",
                      rank=64, clip_dim=512, hidden_dim=512, normalize_keys=True).to(device)
    ckpt = torch.load(CKPT_S1, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.encoder.eval()

    style_image = Image.open(STYLE_IMG).convert("RGB")
    with torch.no_grad():
        q = model.encode_image(style_image, device)
        A, synth_lora = model.forward(q, list(range(pool.num_experts)), temperature=0.1)

    avg = A.mean(dim=(1, 2))
    top5_vals, top5_idx = avg.topk(5)
    log("  Top-5 experts (τ=0.1):")
    for rank_i, ei in enumerate(top5_idx.tolist()):
        log(f"    #{rank_i+1}: {pool.style_names[ei]}  (avg_A={top5_vals[rank_i]:.4f})")
    entropy = -((avg * (avg + 1e-10).log()).sum()).item()
    log(f"  Entropy: {entropy:.4f} / {math.log(pool.num_experts):.4f}")

    # Inject via raw keys (current path) 
    synth_cpu = {k: v.detach().cpu() for k, v in synth_lora.items()}
    inject_lora(pipe, synth_cpu, style_alpha=1.0)
    img_synth_raw = generate(pipe, PROMPT)
    diff_synth_raw = img_diff(img_vanilla, img_synth_raw)
    log(f"  Synth inject_lora MAE diff vs vanilla: {diff_synth_raw:.4f}  ({'ACTIVE' if diff_synth_raw > 2.0 else 'DEAD'})")
    save_pair("vanilla_vs_synth_raw", img_vanilla, img_synth_raw, "Vanilla SDXL", "synth LoRA (inject_lora)", diff_synth_raw)
    unload_lora(pipe)

    # Inject via hooks
    synth_gpu = {k: v.detach().to(device) for k, v in synth_lora.items()}
    hooks = apply_lora_hooks_with_grad(pipe.unet, synth_gpu, alpha=1.0)
    img_synth_hooks = generate(pipe, PROMPT)
    diff_synth_hooks = img_diff(img_vanilla, img_synth_hooks)
    log(f"  Synth hooks MAE diff vs vanilla:       {diff_synth_hooks:.4f}  ({'ACTIVE' if diff_synth_hooks > 2.0 else 'DEAD'})")
    save_pair("vanilla_vs_synth_hooks", img_vanilla, img_synth_hooks, "Vanilla SDXL", "synth LoRA (hooks)", diff_synth_hooks)
    remove_hooks(hooks)

    # Sharp routing: τ=0.01
    with torch.no_grad():
        A_sharp, synth_sharp = model.forward(q, list(range(pool.num_experts)), temperature=0.01)
    avg_sharp = A_sharp.mean(dim=(1, 2))
    top1_name = pool.style_names[avg_sharp.argmax().item()]
    ent_sharp = -((avg_sharp * (avg_sharp + 1e-10).log()).sum()).item()
    log(f"\n  At τ=0.01: top-1={top1_name}, entropy={ent_sharp:.4f}")

    inject_lora(pipe, {k: v.detach().cpu() for k, v in synth_sharp.items()}, style_alpha=1.0)
    img_sharp = generate(pipe, PROMPT)
    diff_sharp = img_diff(img_vanilla, img_sharp)
    log(f"  Sharp (τ=0.01) inject_lora MAE diff:  {diff_sharp:.4f}  ({'ACTIVE' if diff_sharp > 2.0 else 'DEAD'})")
    save_pair("vanilla_vs_synth_sharp", img_vanilla, img_sharp, "Vanilla SDXL", f"synth τ=0.01 (inject_lora)", diff_sharp)
    unload_lora(pipe)

    hooks_sharp = apply_lora_hooks_with_grad(pipe.unet, {k: v.detach().to(device) for k, v in synth_sharp.items()}, alpha=1.0)
    img_sharp_hooks = generate(pipe, PROMPT)
    diff_sharp_hooks = img_diff(img_vanilla, img_sharp_hooks)
    log(f"  Sharp (τ=0.01) hooks  MAE diff:        {diff_sharp_hooks:.4f}  ({'ACTIVE' if diff_sharp_hooks > 2.0 else 'DEAD'})")
    save_pair("vanilla_vs_synth_sharp_hooks", img_vanilla, img_sharp_hooks, "Vanilla SDXL", f"synth τ=0.01 (hooks)", diff_sharp_hooks)
    remove_hooks(hooks_sharp)

    # ── Alpha amplification sweep ─────────────────────────────────────────────
    # The synth LoRA at α=1 has MAE≈8.6 vs real B-LoRA MAE≈32.
    # The average of 109 experts partially cancels (different styles push in
    # opposing directions), making the net delta 3-4x weaker than a single B-LoRA.
    # Here we scale up α to reach real B-LoRA strength and confirm visible changes.
    log("\n[9] Alpha amplification: synth LoRA (τ=0.01) at α=2, 4, 8...")
    synth_sharp_cpu = {k: v.detach().cpu() for k, v in synth_sharp.items()}
    for alpha_val in [2.0, 4.0, 8.0]:
        inject_lora(pipe, synth_sharp_cpu, style_alpha=alpha_val)
        img_a = generate(pipe, PROMPT)
        diff_a = img_diff(img_vanilla, img_a)
        log(f"  τ=0.01 α={alpha_val:.1f}  MAE diff: {diff_a:.4f}  ({'ACTIVE' if diff_a > 2.0 else 'DEAD'})")
        save_pair(f"vanilla_vs_synth_sharp_a{int(alpha_val)}", img_vanilla, img_a,
                  "Vanilla SDXL", f"synth τ=0.01 α={alpha_val}", diff_a)
        unload_lora(pipe)

    # ── Single expert injection: what would perfect top-1 routing look like? ──
    # Inject ONLY the top-1 expert (style_0001_Realism) at α=1 to show:
    # 1. Injection at α=1 gives MAE≈32 (same as any real B-LoRA)
    # 2. The style is Realism, NOT Baroque — confirms routing is wrong, not injection
    log("\n[10] Single-expert injection: top-1 expert (style_0001_Realism) directly...")
    top1_idx = avg_sharp.argmax().item()
    top1_name = pool.style_names[top1_idx]
    top1_sd_all = pool.get_style_tensors(top1_idx)  # full style dict for this expert
    top1_sd = {k: v for k, v in top1_sd_all.items() if "attentions.1" in k}
    log(f"  Expert: {top1_name}  ({len(top1_sd)} tensors)")
    inject_lora(pipe, top1_sd, style_alpha=1.0)
    img_top1 = generate(pipe, PROMPT)
    diff_top1 = img_diff(img_vanilla, img_top1)
    log(f"  MAE diff vs vanilla: {diff_top1:.4f}")
    save_pair("vanilla_vs_top1_expert", img_vanilla, img_top1,
              "Vanilla SDXL", f"top-1 expert ({top1_name}) α=1", diff_top1)
    unload_lora(pipe)

    # Also inject the actual Baroque expert directly
    baroque_idx = next((i for i, n in enumerate(pool.style_names) if "Baroque" in n), None)
    if baroque_idx is not None:
        baroque_sd_all = pool.get_style_tensors(baroque_idx)
        baroque_sd = {k: v for k, v in baroque_sd_all.items() if "attentions.1" in k}
        log(f"\n[11] Baroque expert direct injection ({pool.style_names[baroque_idx]})...")
        inject_lora(pipe, baroque_sd, style_alpha=1.0)
        img_baroque = generate(pipe, PROMPT)
        diff_baroque = img_diff(img_vanilla, img_baroque)
        log(f"  MAE diff vs vanilla: {diff_baroque:.4f}")
        save_pair("vanilla_vs_baroque_expert", img_vanilla, img_baroque,
                  "Vanilla SDXL", f"Baroque expert α=1 (oracle)", diff_baroque)
        unload_lora(pipe)

except Exception as e:
    log(f"  [ERROR] MoELoRAv2 test failed: {e}")
    import traceback; traceback.print_exc()

# ── Test 6: Direct weight merging (merge LoRA into base weights, bypass API) ──
log("\n[8] Test: direct weight merging into UNet base weights...")
try:
    # Manually apply LoRA delta: W_new = W_base + alpha * W_up @ W_down
    unet = pipe.unet
    applied = 0
    for down_key in [k for k in style_raw if "lora.down.weight" in k]:
        up_key = down_key.replace("lora.down.weight", "lora.up.weight")
        if up_key not in style_raw:
            continue
        # Layer path: strip "unet." prefix and ".lora.down.weight" suffix
        layer_path = down_key.replace("unet.", "", 1).replace(".lora.down.weight", "")
        try:
            layer = unet.get_submodule(layer_path)
        except AttributeError:
            continue
        W_down = style_raw[down_key].float().to(device)  # (r, d_in)
        W_up   = style_raw[up_key].float().to(device)    # (d_out, r)
        delta  = (W_up @ W_down).to(layer.weight.dtype)  # (d_out, d_in)
        layer.weight.data += 1.0 * delta
        applied += 1
    log(f"  Merged {applied} LoRA deltas directly into UNet weights")
    img_merged = generate(pipe, PROMPT, lora_scale=0.0)  # scale=0 since already merged
    diff_merged = img_diff(img_vanilla, img_merged)
    log(f"  Direct merge MAE diff vs vanilla: {diff_merged:.4f}  ({'ACTIVE' if diff_merged > 2.0 else 'DEAD'})")
    save_pair("vanilla_vs_direct_merge", img_vanilla, img_merged, "Vanilla SDXL", "direct weight merge (B-LoRA Baroque)", diff_merged)
    # Undo the merge
    for down_key in [k for k in style_raw if "lora.down.weight" in k]:
        up_key = down_key.replace("lora.down.weight", "lora.up.weight")
        if up_key not in style_raw:
            continue
        layer_path = down_key.replace("unet.", "", 1).replace(".lora.down.weight", "")
        try:
            layer = unet.get_submodule(layer_path)
        except AttributeError:
            continue
        W_down = style_raw[down_key].float().to(device)
        W_up   = style_raw[up_key].float().to(device)
        delta  = (W_up @ W_down).to(layer.weight.dtype)
        layer.weight.data -= 1.0 * delta
except Exception as e:
    log(f"  [ERROR] Direct merge failed: {e}")
    import traceback; traceback.print_exc()

# ── Final summary ─────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("SUMMARY")
log("="*60)
(OUT_DIR / "report.txt").write_text("\n".join(report_lines))
log(f"\nAll outputs → {OUT_DIR}")
