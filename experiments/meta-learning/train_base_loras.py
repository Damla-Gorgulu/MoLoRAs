#!/usr/bin/env python3
"""
This script submits training jobs for B-LoRA training on a SLURM cluster.

For each style directory in data/b_lora_data, it submits a SLURM job
to train a B-LoRA weight using valar_scripts/train_b_lora.sh
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def discover_styles(data_dir: Path):
    """Discover all style directories with images."""
    styles = []
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return styles
    
    print(f"Scanning for styles in: {data_dir}")
    
    for style_dir in sorted(data_dir.iterdir()):
        if not style_dir.is_dir():
            continue
        
        # Find all image files
        images = (
            list(style_dir.glob("*.jpeg")) +
            list(style_dir.glob("*.jpg")) +
            list(style_dir.glob("*.png")) +
            list(style_dir.glob("*.JPEG")) +
            list(style_dir.glob("*.JPG")) +
            list(style_dir.glob("*.PNG"))
        )
        
        if len(images) == 0:
            print(f"  ⚠️  Skipping {style_dir.name} - no images found")
            continue
        
        if len(images) < 5:
            print(f"  ⚠️  Warning: {style_dir.name} has only {len(images)} images")
        
        styles.append({
            'name': style_dir.name,
            'path': str(style_dir.absolute()),
            'num_images': len(images)
        })
        print(f"  ✓ Found {style_dir.name} with {len(images)} images")
    
    return styles


def submit_training_job(instance_dir: str, output_dir: str, prompt: str, 
                       job_name: str, script_path: Path, dry_run: bool = False):
    """Submit a SLURM job for B-LoRA training."""
    
    cmd = [
        'sbatch',
        str(script_path),
        instance_dir,
        output_dir,
        prompt
    ]
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return "DRY_RUN_12345"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse job ID from output: "Submitted batch job 123456"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            job_id = match.group(1)
            return job_id
        else:
            print(f"  ⚠️  Warning: Could not parse job ID from: {result.stdout}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error submitting job: {e}")
        print(f"     stdout: {e.stdout}")
        print(f"     stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  ❌ Error: 'sbatch' command not found. Are you on a SLURM cluster?")
        sys.exit(1)


def save_manifest(styles, submissions, manifest_path: Path):
    """Save training manifest with submission details."""
    manifest = {
        'created_at': datetime.now().isoformat(),
        'total_styles': len(styles),
        'submissions': submissions
    }
    
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📝 Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Submit B-LoRA training jobs for all styles'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/b_lora_data'),
        help='Directory containing style subdirectories (default: data/b_lora_data)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/meta-learning/trained_loras'),
        help='Base directory for trained LoRA outputs'
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=8,
        help='LoRA rank (default: 8)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip styles that already have trained weights'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview jobs without actually submitting them'
    )
    parser.add_argument(
        '--script',
        type=Path,
        default=Path('valar_scripts/train_b_lora.sh'),
        help='Path to SLURM training script'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / args.data_dir
    output_base = repo_root / args.output_dir
    script_path = repo_root / args.script
    
    print("="*80)
    print("B-LoRA Training Job Submission")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_base}")
    print(f"Script: {script_path}")
    print(f"Rank: {args.rank}")
    print(f"Resume mode: {args.resume}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    print()
    
    # Validate script exists
    if not script_path.exists():
        print(f"❌ Error: Training script not found: {script_path}")
        sys.exit(1)
    
    # Discover styles
    styles = discover_styles(data_dir)
    
    if not styles:
        print("\n❌ No valid styles found. Exiting.")
        sys.exit(1)
    
    print(f"\n📊 Found {len(styles)} styles to train\n")
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Submit jobs
    submissions = []
    submitted_count = 0
    skipped_count = 0
    
    for style in styles:
        style_output_dir = output_base / f"{style['name']}_rank{args.rank}"
        weights_file = style_output_dir / "pytorch_lora_weights.safetensors"
        
        print(f"Processing: {style['name']}")
        
        # Check if already trained
        if args.resume and weights_file.exists():
            print(f"  ⏭️  Skipping - already trained")
            skipped_count += 1
            submissions.append({
                'style': style['name'],
                'status': 'skipped',
                'reason': 'already_trained',
                'output_dir': str(style_output_dir)
            })
            continue
        
        # Submit job
        job_name = f"BLoRA-{style['name']}"
        job_id = submit_training_job(
            instance_dir=style['path'],
            output_dir=str(style_output_dir),
            prompt=style['name'],
            job_name=job_name,
            script_path=script_path,
            dry_run=args.dry_run
        )
        
        if job_id:
            print(f"  ✅ Submitted job {job_id}")
            submitted_count += 1
            submissions.append({
                'style': style['name'],
                'status': 'submitted',
                'job_id': job_id,
                'instance_dir': style['path'],
                'output_dir': str(style_output_dir),
                'num_images': style['num_images'],
                'submitted_at': datetime.now().isoformat()
            })
        else:
            print(f"  ❌ Failed to submit")
            submissions.append({
                'style': style['name'],
                'status': 'failed',
                'reason': 'submission_error'
            })
        
        print()
    
    # Summary
    print("="*80)
    print("Submission Summary")
    print("="*80)
    print(f"Total styles: {len(styles)}")
    print(f"Submitted: {submitted_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(styles) - submitted_count - skipped_count}")
    print("="*80)
    
    # Save manifest
    manifest_path = output_base / 'training_manifest.json'
    save_manifest(styles, submissions, manifest_path)
    
    if not args.dry_run and submitted_count > 0:
        print(f"\n💡 Monitor jobs with: squeue -u $USER")
        print(f"💡 Check logs in: /logs/b_lora_training/")


if __name__ == '__main__':
    main()


