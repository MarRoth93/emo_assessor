#!/bin/bash
#SBATCH --job-name=compute_pca
#SBATCH --output=/home/rothermm/new_assessor/logs/compute_pca_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/compute_pca_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Load required modules
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

# Navigate to project directory
cd /home/rothermm/new_assessor

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results/pca_analysis

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set paths
RATINGS_FILE="./ratings/per_image_Slider_mean_sd_from_wide.csv"
OUTPUT_DIR="./results/pca_analysis"

# PCA parameters
N_COMPONENTS=4
TRAIN_RATIO=0.7
VAL_RATIO=0.15

# Run PCA computation
echo "=================================================="
echo "Computing PCA on Dimensional Ratings (NO LEAKAGE)"
echo "=================================================="
echo "Ratings: $RATINGS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Components: $N_COMPONENTS"
echo "Split ratios: train=$TRAIN_RATIO, val=$VAL_RATIO, test=$(echo "1 - $TRAIN_RATIO - $VAL_RATIO" | bc)"
echo ""
echo "IMPORTANT: PCA fits on TRAIN data only!"
echo ""

python scripts/02_compute_pca.py \
    --ratings "$RATINGS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n_components "$N_COMPONENTS" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --random_state 42

# Print completion info
echo ""
echo "=================================================="
echo "PCA computation complete!"
echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - PCA model: pca_targets.joblib"
echo "  - Train ratings: ratings_pca_${N_COMPONENTS}comp_train.csv"
echo "  - Val ratings: ratings_pca_${N_COMPONENTS}comp_val.csv"
echo "  - Test ratings: ratings_pca_${N_COMPONENTS}comp_test.csv"
echo "  - Loadings sorted: pca_loadings_PC*_sorted.csv"
echo "  - Split indices: split_indices.json"
echo "  - Visualizations: *.png"
echo ""
echo "Next step: Train model with decoder"
echo "  sbatch slurm_scripts/train_assessor_pca.sh"
echo ""
echo "Job completed at: $(date)"
echo "=================================================="
