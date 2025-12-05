#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --output=/home/rothermm/new_assessor/logs/evaluate_model_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/evaluate_model_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# ==========================================
# CONFIGURATION - EDIT THESE PATHS
# ==========================================

# Model type: 'ensemble' for ViT+ResNet, 'single' for single backbone, 'auto' to detect
MODEL_TYPE="ensemble"

# Path to the trained model checkpoint
# For ensemble model:
CHECKPOINT="./results/ensemble_v2/best_model.pt"
# For single backbone model:
# CHECKPOINT="./outputs/experiments/vit_b_16/run_XXXXXX/best_model.pth"

# Use PCA mode? (model predicts PC scores)
USE_PCA=true

# Use decoder for reconstruction?
USE_DECODER=false  # Set to true if you want to evaluate reconstructed 14-dim outputs

# PCA model path (required if USE_DECODER=true)
PCA_MODEL="./results/pca_analysis/pca_targets.joblib"

# Ratings file - use TEST split for final evaluation
if [ "$USE_PCA" = true ]; then
    RATINGS_FILE="./results/pca_analysis/ratings_pca_4comp_test.csv"
    echo "Evaluating on PCA TEST set"
else
    RATINGS_FILE="./ratings/per_image_Slider_mean_sd_from_wide.csv"
    echo "Evaluating on original dimensions"
fi

IMAGES_DIR="./images"

# Automatically set output dir to checkpoint dir + /evaluation
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
OUTPUT_DIR="${CHECKPOINT_DIR}/evaluation"

# Evaluation parameters
BATCH_SIZE=32
NUM_WORKERS=4

# Build command flags
FLAGS=""
if [ "$USE_PCA" = true ]; then
    FLAGS="$FLAGS --use_pca"
fi
if [ "$USE_DECODER" = true ]; then
    FLAGS="$FLAGS --use_decoder"
fi
# Always provide PCA model path - script will auto-detect if decoder is needed
FLAGS="$FLAGS --pca_model $PCA_MODEL"

# ==========================================
# RUN EVALUATION
# ==========================================

echo "=================================================="
echo "Evaluating Trained Model"
echo "=================================================="
echo "Model Type: $MODEL_TYPE"
echo "Checkpoint: $CHECKPOINT"
echo "Images: $IMAGES_DIR"
echo "Ratings: $RATINGS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Use PCA: $USE_PCA"
echo "Use Decoder: $USE_DECODER"
echo ""

python scripts/09_evaluate_model.py \
    --checkpoint "$CHECKPOINT" \
    --images_dir "$IMAGES_DIR" \
    --ratings_file "$RATINGS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --model_type "$MODEL_TYPE" \
    $FLAGS

# Print completion info
echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - Metrics: test_metrics.csv"
echo "  - Predictions: detailed_predictions.csv"
echo "  - Visualizations: *.png"
echo ""
echo "Job completed at: $(date)"
echo "=================================================="
