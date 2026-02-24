# Project Guidelines — MoLoRAs

## Code Style
- Language: **Python 3.10+** predominately (training scripts, utilities, and experiments).
- Follow existing project patterns: use argparse for CLIs, modular functions in `unziplora_unet/`, and docstrings consistent with Hugging Face style.
- Keep files small and focused (see `train_unziplora.py` and `train_dreambooth_b-lora_sdxl.py` for examples).

## Architecture
- Major components:
  - `unziplora_unet/`: core UnZipLoRA implementation (UNet, pipeline, linear layers).
  - `B-LoRA_files/`: B-LoRA training/inference utilities.
  - `experiments/meta-learning/`: orchestration for meta-learning experiments (style discovery, job submission, manifests, meta-learner training code).
  - `valar_scripts/`: SLURM job scripts used for cluster runs.
- Data flows: image folders in `data/b_lora_data/` -> per-style LoRA training -> saved weights in `experiments/meta-learning/trained_loras/` -> meta-learner training/evaluation.

## Build and Test
- Install deps (GPU recommended):
  - pip: `pip install -r requirements.txt` (see top-level `requirements.txt` for extras).
- Quick smoke checks:
  - Local inference: `bash infer.sh` (set `MODEL_NAME`, `OUTPUT_DIR` as needed).
  - Local training (single style): adapt `train.sh` and run `accelerate launch train_unziplora.py`.
- Note: SLURM-specific scripts require `sbatch`; use `experiments/meta-learning/train_base_loras.py --dry-run` to preview jobs without cluster access.

## Project Conventions
- Large artifacts and credentials are excluded in `.gitignore`: do NOT commit `models/`, `.safetensors`, `wandb/*`, or large outputs.
- Naming conventions:
  - Trained LoRAs: `experiments/meta-learning/trained_loras/<style>_rank<N>/pytorch_lora_weights.safetensors`.
  - Model outputs often use suffixes `_content` and `_style` for LoRA weights.
- When adding a new style:
  1. Add image files to `data/b_lora_data/<style>/` (include a `prompt.txt` when helpful).
  2. Run `python experiments/meta-learning/train_base_loras.py` (use `--dry-run` then submit).

## Integration Points
- External services: Hugging Face Hub (uploading artifacts), Weights & Biases (logging), and SLURM cluster for distributed training.
- Key files demonstrating integrations:
  - HF / uploads: `train_unziplora.py` (model card creation and upload logic).
  - SLURM submission: `experiments/meta-learning/train_base_loras.py` and `valar_scripts/train_b_lora.sh`.

## Security
- Do not store secrets in repo. Use environment variables for HF and W&B tokens (e.g., `HF_TOKEN`, `WANDB_API_KEY`).
- `.gitignore` already excludes `wandb/*` and model artifacts—preserve this.

## Working with Agents (Practical Tips)
- Use `train_base_loras.py --dry-run` to validate job submission behavior before cluster submission.
- Prefer small, reproducible experiments when adding code; heavy training should be run on cluster with `sbatch`.
- When modifying model-loading or pipeline logic, add a small reproducible script (or notebook cell in `playground.ipynb`) that runs a forward pass locally.

---

If you'd like, I can add a CI check template (basic linting and a lightweight smoke inference test) and a short CONTRIBUTING note describing how to add new styles and job manifest expectations. Would you like that added?