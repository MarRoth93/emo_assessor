#!/bin/bash
#SBATCH --job-name=analyze_experiments
#SBATCH --output=/home/rothermm/new_assessor/logs/analyze_experiments_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/analyze_experiments_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# =============================================================================
# Hyperparameter Experiment Analysis
# =============================================================================
# This script analyzes and compares all hyperparameter search experiments.
# It generates comparison tables, visualizations, and recommendations.
#
# Usage:
#   sbatch slurm_scripts/08_analyze_experiments.sh
#   bash slurm_scripts/08_analyze_experiments.sh                    # Run locally
#   bash slurm_scripts/08_analyze_experiments.sh --top_k 5          # Show top 5
#   bash slurm_scripts/08_analyze_experiments.sh --experiments_dir ./outputs/custom
# =============================================================================

# Print job information
echo "=================================================="
echo "Hyperparameter Experiment Analysis"
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
mkdir -p results/experiment_analysis

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Working Directory: $(pwd)"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================
EXPERIMENTS_DIR="./outputs/experiments"
OUTPUT_DIR="./results/experiment_analysis"
TOP_K=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments_dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--experiments_dir DIR] [--output_dir DIR] [--top_k N]"
            exit 1
            ;;
    esac
done

# =============================================================================
# RUN ANALYSIS
# =============================================================================
echo "=================================================="
echo "Analyzing Experiments"
echo "=================================================="
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Top K experiments to plot: $TOP_K"
echo ""

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "ERROR: Experiments directory not found: $EXPERIMENTS_DIR"
    echo ""
    echo "Please run hyperparameter search first:"
    echo "  sbatch slurm_scripts/07_run_hyperparameter_search.sh"
    exit 1
fi

# Count available experiments (check for both CSV and JSON formats)
EXP_COUNT_CSV=$(find "$EXPERIMENTS_DIR" -name "training_history.csv" 2>/dev/null | wc -l)
EXP_COUNT_JSON=$(find "$EXPERIMENTS_DIR" -name "training_history.json" 2>/dev/null | wc -l)
EXP_COUNT=$((EXP_COUNT_CSV + EXP_COUNT_JSON))
echo "Found $EXP_COUNT experiment(s) with training history"
echo ""

if [ "$EXP_COUNT" -eq 0 ]; then
    echo "ERROR: No completed experiments found."
    echo "Please wait for experiments to complete or check the directory."
    exit 1
fi

# Run analysis
python scripts/08_analyze_experiments.py \
    --experiments_dir "$EXPERIMENTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --top_k "$TOP_K"

# Print completion info
echo ""
echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - experiment_comparison.csv    : Full comparison table"
echo "  - training_curves_comparison.png : Training curves for top experiments"
echo "  - hyperparameter_analysis.png  : Impact of each hyperparameter"
echo "  - experiment_ranking.png       : Visual ranking of all experiments"
echo "  - recommendations.txt          : Best hyperparameters to use"
echo ""
echo "Next steps:"
echo "  1. Review recommendations.txt for best hyperparameters"
echo "  2. Check training_curves_comparison.png for overfitting"
echo "  3. Use best checkpoint for evaluation:"
echo "     - Edit slurm_scripts/04_evaluate_model.sh with best model path"
echo "     - sbatch slurm_scripts/04_evaluate_model.sh"
echo ""
echo "Job completed at: $(date)"
echo "=================================================="
