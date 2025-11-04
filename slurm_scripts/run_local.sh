#!/bin/bash

# Local training script (no SLURM)
# Use this for testing or training on a local machine

echo "=================================================="
echo "Local Training Script"
echo "Start Time: $(date)"
echo "=================================================="

# Navigate to project directory
cd /home/rothermm/new_assessor

# Create necessary directories
mkdir -p logs
mkdir -p outputs

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    USE_AMP="--use_amp"
else
    echo "⚠️  No GPU detected - training will be slow!"
    USE_AMP=""
fi
echo ""

# Set paths
IMAGES_DIR="./images"
RATINGS_FILE="./ratings/per_image_Slider_mean_sd_from_wide.csv"
OUTPUT_DIR="./outputs"

# Training parameters (lighter for local testing)
BACKBONE="resnet50"
EPOCHS=30
BATCH_SIZE=16           # Smaller batch for local
LEARNING_RATE=0.0001
OPTIMIZER="adamw"
SCHEDULER="cosine"
IMAGE_SIZE=224
TRAIN_RATIO=0.8
DROPOUT=0.5

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
    --dropout "$DROPOUT" \
    --num_workers 4 \
    --early_stopping 10 \
    --save_every 5 \
    --pretrained \
    $USE_AMP \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "Training completed at: $(date)"
echo "=================================================="