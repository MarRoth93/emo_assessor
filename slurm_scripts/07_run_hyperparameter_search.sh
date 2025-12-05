#!/bin/bash
#SBATCH --job-name=hp_search_launcher
#SBATCH --output=/home/rothermm/new_assessor/logs/hp_search_launcher_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/hp_search_launcher_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# =============================================================================
# Hyperparameter Search Launcher
# =============================================================================
# This script launches the hyperparameter search for the PCA + Decoder model.
# It creates SLURM scripts for each experiment configuration and submits them.
#
# Usage:
#   sbatch slurm_scripts/07_run_hyperparameter_search.sh              # Run all experiments
#   sbatch slurm_scripts/07_run_hyperparameter_search.sh baseline     # Run specific experiment
#   bash slurm_scripts/07_run_hyperparameter_search.sh --dry-run      # Preview without submitting
# =============================================================================

# Print job information
echo "=================================================="
echo "Hyperparameter Search Launcher"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start Time: $(date)"
echo "=================================================="

# Load required modules
module load miniconda 2>/dev/null || true
source $CONDA_ROOT/bin/activate 2>/dev/null || true
conda activate mediaeval 2>/dev/null || true

# Navigate to project directory
cd /home/rothermm/new_assessor

# Create required directories
mkdir -p logs
mkdir -p outputs/experiments
mkdir -p slurm_scripts/experiments

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Working Directory: $(pwd)"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set default mode and experiments
MODE="both"           # Options: create, submit, both
SEQUENTIAL=false      # Submit jobs sequentially with dependencies
DRY_RUN=false         # Preview without submitting

# Parse command line arguments
EXPERIMENTS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --create-only)
            MODE="create"
            shift
            ;;
        --submit-only)
            MODE="submit"
            shift
            ;;
        *)
            # Assume it's an experiment name
            EXPERIMENTS="$EXPERIMENTS $1"
            shift
            ;;
    esac
done

# Build Python command
CMD="python scripts/07_run_hyperparameter_search.py"
CMD="$CMD --mode $MODE"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ "$SEQUENTIAL" = true ]; then
    CMD="$CMD --sequential"
fi

if [ -n "$EXPERIMENTS" ]; then
    CMD="$CMD --experiments $EXPERIMENTS"
fi

# =============================================================================
# RUN HYPERPARAMETER SEARCH
# =============================================================================
echo "=================================================="
echo "Running Hyperparameter Search"
echo "=================================================="
echo "Mode: $MODE"
echo "Sequential: $SEQUENTIAL"
echo "Dry Run: $DRY_RUN"
echo "Experiments: ${EXPERIMENTS:-all}"
echo ""
echo "Command: $CMD"
echo ""

# Execute
eval $CMD

# Print completion info
echo ""
echo "=================================================="
echo "Launcher completed at: $(date)"
echo "=================================================="
echo ""
echo "Available experiments:"
echo "  baseline, high_reg, very_high_reg, efficientnet_b0,"
echo "  efficientnet_b0_high_reg, low_lr, very_low_lr, sgd_optimizer,"
echo "  efficientnet_b3, best_combo, resnet101, high_aux_loss,"
echo "  mse_loss, vit_b_16"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  ./outputs/experiments/<experiment_name>/"
echo ""
