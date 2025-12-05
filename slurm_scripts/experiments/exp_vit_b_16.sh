#!/bin/bash
#SBATCH --job-name=exp_vit_b_16
#SBATCH --output=/home/rothermm/new_assessor/logs/exp_vit_b_16_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/exp_vit_b_16_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Experiment: Vision Transformer (ViT-B/16)
# Architecture: PCA + Decoder
echo "=================================================="
echo "Experiment: Vision Transformer (ViT-B/16)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Load environment
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

# Navigate to project directory
cd /home/rothermm/new_assessor

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# PCA Pipeline Configuration
IMAGES_DIR="./images"
RATINGS_FILE="./results/pca_analysis/ratings_pca_4comp_train.csv"
RATINGS_FILE_14="./ratings/per_image_Slider_mean_sd_from_wide.csv"
PCA_MODEL="./results/pca_analysis/pca_targets.joblib"
OUTPUT_DIR="./outputs/experiments/vit_b_16"

echo "=================================================="
echo "Training with PCA + Decoder Architecture"
echo "=================================================="
echo "Images: $IMAGES_DIR"
echo "Ratings: $RATINGS_FILE"
echo "PCA Model: $PCA_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python scripts/03_train_model.py \
    --images_dir "$IMAGES_DIR" \
    --ratings_file "$RATINGS_FILE" \
    --ratings_file_14 "$RATINGS_FILE_14" \
    --output_dir "$OUTPUT_DIR" \
    --use_pca \
    --pca_model "$PCA_MODEL" \
    --use_decoder \
    --backbone vit_b_16 \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-05 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --scheduler cosine \
    --dropout 0.5 \
    --aux_loss_weight 0.2 \
    --grad_clip 1.0 \
    --loss_fn smooth_l1 \
    --image_size 224 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --num_workers 8 \
    --early_stopping 15 \
    --save_every 10 \
    --use_amp \
    --pretrained

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="
