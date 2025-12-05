#!/bin/bash
#SBATCH --job-name=ensemble_2pc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ensemble_2pc_%j.out
#SBATCH --error=logs/ensemble_2pc_%j.err

# Load environment
source ~/.bashrc
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

cd /home/rothermm/new_assessor
mkdir -p logs

echo "=================================================="
echo "Training 2-Component Ensemble (No Decoder)"
echo "=================================================="

# Step 1: Compute 2-component PCA (if not already done)
if [ ! -f "results/pca_2comp/pca_2comp.joblib" ]; then
    echo "Computing 2-component PCA..."
    python scripts/02b_compute_pca_2comp.py \
        --ratings_file ratings/per_image_Slider_mean_sd_from_wide.csv \
        --output_dir results/pca_2comp \
        --split_file results/pca_analysis/split_indices.json
fi

# Step 2: Train ensemble WITHOUT decoder
# Just predict the 2 PC scores directly
python scripts/03b_train_ensemble.py \
    --train_csv results/pca_2comp/ratings_2comp_train.csv \
    --val_csv results/pca_2comp/ratings_2comp_val.csv \
    --pca_model results/pca_2comp/pca_2comp.joblib \
    --pca_components 2 \
    --use_pca \
    --freeze_backbones \
    --augmentation moderate \
    --use_mixup \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --loss_fn smooth_l1 \
    --use_amp \
    --output_dir results/ensemble_2comp \
    --seed 42

echo "Training complete!"