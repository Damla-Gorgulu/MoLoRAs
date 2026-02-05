# Meta-Learning Experiment: Gradient Transformer for Continual Learning

## Quick Start

### Step 1: Train Base B-LoRA Weights

First, train B-LoRA weights for all styles in your dataset:

```bash
cd /home/eyavuz21/repos/MoLoRAs

# Preview what will be submitted (dry run)
python experiments/meta-learning/train_base_loras.py --dry-run

# Submit all training jobs
python experiments/meta-learning/train_base_loras.py

# Resume training (skip already completed styles)
python experiments/meta-learning/train_base_loras.py --resume
```

**Options:**
- `--data-dir`: Path to style images (default: `data/b_lora_data`)
- `--output-dir`: Where to save trained LoRAs (default: `experiments/meta-learning/trained_loras`)
- `--rank`: LoRA rank (default: 8)
- `--resume`: Skip styles that already have trained weights
- `--dry-run`: Preview without submitting jobs

**Output Structure:**
```
experiments/meta-learning/
├── trained_loras/
│   ├── cartoon_rank8/
│   │   └── pytorch_lora_weights.safetensors
│   ├── watercolor_rank8/
│   │   └── pytorch_lora_weights.safetensors
│   └── ...
└── training_manifest.json
```

**Monitor Jobs:**
```bash
# Check job status
squeue -u $USER

# View logs
tail -f /logs/b_lora_training/b-lora-training-<JOB_ID>.log

# Check manifest
cat experiments/meta-learning/trained_loras/training_manifest.json
```

### Step 2: Train Meta-Learner (TODO)

Coming soon: Script to train the gradient transformer using trained LoRA weights.

### Step 3: Evaluate (TODO)

Coming soon: Validation script to test forgetting prevention.

## Implementation Details

### train_base_loras.py

- Discovers all style directories in `data/b_lora_data`
- Validates each directory contains images
- Submits SLURM job for each style using `valar_scripts/train_b_lora.sh`
- Tracks submissions in `training_manifest.json`
- Supports resume mode to skip already-trained styles

### valar_scripts/train_b_lora.sh

Parameterized SLURM script that accepts:
1. `INSTANCE_DIR`: Path to style images
2. `OUTPUT_DIR`: Where to save trained LoRA
3. `PROMPT`: Style name for training

**Training Configuration:**
- Rank: 8
- Steps: 1000
- Learning rate: 5e-5
- Batch size: 1
- Mixed precision: fp16
- GPU: 1x Tesla V100
- Time limit: 24 hours

## Troubleshooting

**No images found:**
- Ensure `data/b_lora_data` contains subdirectories with image files
- Supported formats: .jpg, .jpeg, .png

**sbatch command not found:**
- Script must be run on SLURM cluster (login node)
- Use `ssh software` if downloading/installing packages

**Jobs fail immediately:**
- Check logs in `/logs/b_lora_training/`
- Verify conda environment `B-LoRA_2` exists
- Ensure SDXL model available in huggingface cache

**Out of disk space:**
- Check output directory size
- Each LoRA is ~20-50 MB
- Consider using `/scratch` for temporary storage
