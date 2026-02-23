#!/bin/bash
#SBATCH --job-name=download-images
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=software
#SBATCH --time=00:10:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/download-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/download-%j.err

set -euo pipefail
OUTDIR="/home/eyavuz21/repos/MoLoRAs/lora_attention/test_images/novel_styles"
mkdir -p "$OUTDIR"

# Surrealism: Dalí – Persistence of Memory (public domain)
wget -q "https://upload.wikimedia.org/wikipedia/en/d/dd/The_Persistence_of_Memory.jpg" \
  -O "$OUTDIR/surrealism_dali_persistence.jpg" && echo "surrealism OK" || echo "surrealism FAILED"

# Pointillism: Seurat – Sunday on La Grande Jatte (public domain)
wget -q "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg/800px-A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg" \
  -O "$OUTDIR/pointillism_seurat_grande_jatte.jpg" && echo "pointillism OK" || echo "pointillism FAILED"

# Ukiyo-e: Hokusai – Great Wave (public domain)
wget -q "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg" \
  -O "$OUTDIR/ukiyo_e_hokusai_great_wave.jpg" && echo "ukiyo-e OK" || echo "ukiyo-e FAILED"

# Futurism: Boccioni – States of Mind (public domain)
wget -q "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Umberto_Boccioni_-_The_City_Rises.jpg/800px-Umberto_Boccioni_-_The_City_Rises.jpg" \
  -O "$OUTDIR/futurism_boccioni_city_rises.jpg" && echo "futurism OK" || echo "futurism FAILED"

echo ""
ls -la "$OUTDIR"
