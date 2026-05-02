#!/bin/bash
#SBATCH --job-name=LoRA-AE-Resid1on16
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-resid1on16-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-resid1on16-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}
WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python -c '
import argparse, json, os, sys, torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd().parents[0]))
from lora_attention.lora_autoencoder.dataset import StyleLoRAAutoencoderDataset, make_metadata_tensors
from lora_attention.lora_autoencoder.model import StyleLoRAAutoencoder, reconstruction_losses

zoo_dir = "/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
output_dir = Path("/scratch/eyavuz21/lora_autoencoder/residual1_on16mean")
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

# Load 16 LoRAs for mean template, train on index 0 only
dataset_16 = StyleLoRAAutoencoderDataset(zoo_dir, limit=16)
train_idx = 0

# Compute mean template from all 16
sum_d1280 = torch.zeros(1, dataset_16.num_pairs, dataset_16.rank, 1280)
sum_d2048 = torch.zeros(1, dataset_16.num_pairs, dataset_16.rank, 2048)
sum_up = torch.zeros(1, dataset_16.num_pairs, dataset_16.rank, 1280)
for i in range(len(dataset_16)):
    s = dataset_16[i]
    sum_d1280 += s["down1280"].unsqueeze(0)
    sum_d2048 += s["down2048"].unsqueeze(0)
    sum_up += s["up"].unsqueeze(0)
mean = {
    "down1280": (sum_d1280 / len(dataset_16)).to(device),
    "down2048": (sum_d2048 / len(dataset_16)).to(device),
    "up": (sum_up / len(dataset_16)).to(device),
}

# Train on style_0000 only
target = dataset_16[train_idx]
target = {k: target[k].unsqueeze(0).to(device) for k in ["down1280", "down2048", "up"]}
d_in = dataset_16[train_idx]["d_in"].unsqueeze(0).to(device)
target["d_in"] = d_in

residual_target = {
    "down1280": target["down1280"] - mean["down1280"],
    "down2048": target["down2048"] - mean["down2048"],
    "up": target["up"] - mean["up"],
    "d_in": d_in,
}

# Mean template baseline
with torch.no_grad():
    mean_pred = {k: mean[k].clone() for k in ["down1280", "down2048", "up"]}
    mean_pred["d_in"] = d_in
    base = reconstruction_losses(mean_pred, target)
    print(f"[mean_baseline] cos={base[\"cos\"].item():.4f} rel={base[\"rel\"].item():.4f} mse={base[\"tensor_mse\"].item():.8f}", flush=True)

meta = {k: v.to(device) for k, v in make_metadata_tensors(dataset_16.specs).items()}

model = StyleLoRAAutoencoder(
    num_pairs=dataset_16.num_pairs, rank=dataset_16.rank,
    latent_dim=41472, d_model=512, num_layers=4, num_heads=8,
).to(device)

lr, clip, max_steps = 1e-4, 0.1, 5000
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
log_path = output_dir / "train_log.txt"

print(f"[residual1_on16] target={dataset_16.style_names[train_idx]} pairs={dataset_16.num_pairs} lr={lr}", flush=True)

for step in range(1, max_steps + 1):
    opt.zero_grad(set_to_none=True)
    pred_res = model(residual_target, meta)
    metrics = reconstruction_losses(pred_res, residual_target,
        tensor_weight=1.0, delta_weight=0.0, cos_weight=0.0, rel_weight=0.0, norm_weight=0.0)
    metrics["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()

    if step == 1 or step % 10 == 0:
        with torch.no_grad():
            pred_full = {
                "down1280": pred_res["down1280"].detach() + mean["down1280"],
                "down2048": pred_res["down2048"].detach() + mean["down2048"],
                "up": pred_res["up"].detach() + mean["up"],
                "d_in": d_in,
            }
            fm = reconstruction_losses(pred_full, target)
        line = (
            f"s={step:05d}/{max_steps} "
            f"r_mse={metrics[\"tensor_mse\"].item():.8f} "
            f"r_cos={metrics[\"cos\"].item():.4f} "
            f"r_rel={metrics[\"rel\"].item():.4f} "
            f"r_nr={metrics[\"norm_ratio\"].item():.4f} "
            f"fcos={fm[\"cos\"].item():.4f} "
            f"frel={fm[\"rel\"].item():.4f} "
            f"fnr={fm[\"norm_ratio\"].item():.4f}"
        )
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    if step % 1000 == 0:
        ckpt = output_dir / f"checkpoint-{step}"
        ckpt.mkdir(exist_ok=True)
        torch.save({"step": step, "model_state_dict": model.state_dict(), "mean_template": {k: v.cpu() for k, v in mean.items()}}, ckpt / "checkpoint.pt")

with torch.no_grad():
    pred_res = model(residual_target, meta)
    pred_full = {k: pred_res[k].detach() + mean[k] for k in ["down1280", "down2048", "up"]}
    pred_full["d_in"] = d_in
    final = reconstruction_losses(pred_full, target)
    print(json.dumps({k: float(v.item()) for k, v in final.items()}, indent=2))
'
