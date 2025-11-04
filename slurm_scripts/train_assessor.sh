#!/bin/bash
#SBATCH --job-name=train_assessor
#SBATCH --output=../logs/train_assessor_%j.out
#SBATCH --error=../logs/train_assessor_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Load required modules (adjust based on your cluster)
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
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Set paths relative to project root
IMAGES_DIR="./images"
RATINGS_FILE="./ratings/per_image_Slider_mean_sd_from_wide.csv"
OUTPUT_DIR="./outputs"

# Training parameters
BACKBONE="resnet50"          # Options: resnet50, resnet101, efficientnet_b0, efficientnet_b3, vit_b_16
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.0001
OPTIMIZER="adamw"            # Options: adam, adamw, sgd
SCHEDULER="cosine"           # Options: cosine, step, plateau, none
IMAGE_SIZE=224
TRAIN_RATIO=0.7              # 70% for training
VAL_RATIO=0.15               # 15% for validation (remaining 15% for test)
DROPOUT=0.5

# Run training
echo "Starting training..."
echo "Images: $IMAGES_DIR"
echo "Ratings: $RATINGS_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

python scripts/train_assessor.py \
    --images_dir "$IMAGES_DIR" \
    --ratings_file "$RATINGS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --image_size "$IMAGE_SIZE" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --dropout "$DROPOUT" \
    --num_workers 8 \
    --early_stopping 15 \
    --save_every 10 \
    --use_amp \
    --pretrained

# Print completion info
echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="