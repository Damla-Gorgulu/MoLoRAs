#!/bin/bash
# ============================================================
# run_all.sh — Submit all phases sequentially via SLURM dependencies
# ============================================================
#
# Usage:
#   bash run_all.sh          # Submit all 5 jobs with dependencies
#   bash run_all.sh --dry    # Print commands without submitting
#
# Job dependency chain:
#   Phase 0 Extract  →  Phase 0 Sanity  →  Phase 1  →  Phase 2  →  Phase 3
#
# Each phase only starts after the previous one completes successfully.
# ============================================================

EXPERIMENT_DIR=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition
DRY_RUN=false

if [ "$1" = "--dry" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not executed ==="
    echo ""
fi

submit() {
    local script=$1
    local dep=$2
    local cmd="sbatch"

    if [ -n "$dep" ]; then
        cmd="$cmd --dependency=afterok:$dep"
    fi
    cmd="$cmd $EXPERIMENT_DIR/$script"

    if [ "$DRY_RUN" = true ]; then
        echo "  $cmd"
        echo "  (would return job ID)"
        echo ""
        return 0
    fi

    output=$($cmd)
    job_id=$(echo "$output" | grep -oP '\d+$')
    echo "  Submitted: $script → Job $job_id"
    echo "$job_id"
}

echo "========================================"
echo "Linear Composition — Submit All Phases"
echo "========================================"
echo ""

# Phase 0a: Extract
echo "Phase 0 — ΔW Extraction:"
JOB0A=$(submit run_phase0_extract.sh)
echo ""

if [ "$DRY_RUN" = false ]; then
    # Phase 0b: Sanity check (depends on 0a)
    echo "Phase 0 — Sanity Check:"
    JOB0B=$(submit run_phase0_sanity.sh "$JOB0A")
    echo ""

    # Phase 1: Global reconstruction (depends on 0a — doesn't need sanity images)
    echo "Phase 1 — Global Reconstruction:"
    JOB1=$(submit run_phase1.sh "$JOB0A")
    echo ""

    # Phase 2: Layer-wise (depends on Phase 1 for best_methods.json)
    echo "Phase 2 — Layer-wise Reconstruction:"
    JOB2=$(submit run_phase2.sh "$JOB1")
    echo ""

    # Phase 3: Span analysis (depends on Phase 1 for target selection)
    echo "Phase 3 — Span Analysis:"
    JOB3=$(submit run_phase3.sh "$JOB1")
    echo ""

    echo "========================================"
    echo "All jobs submitted!"
    echo "========================================"
    echo ""
    echo "Dependency chain:"
    echo "  Phase 0 Extract  ($JOB0A)"
    echo "    ├── Phase 0 Sanity  ($JOB0B)"
    echo "    └── Phase 1         ($JOB1)"
    echo "          ├── Phase 2   ($JOB2)"
    echo "          └── Phase 3   ($JOB3)"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all:   scancel $JOB0A $JOB0B $JOB1 $JOB2 $JOB3"
else
    submit run_phase0_sanity.sh "<JOB0A>"
    submit run_phase1.sh "<JOB0A>"
    submit run_phase2.sh "<JOB1>"
    submit run_phase3.sh "<JOB1>"
fi
