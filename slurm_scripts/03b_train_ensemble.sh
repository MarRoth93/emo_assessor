#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rothermm/new_assessor/logs/ensemble_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/ensemble_%j.err

# Load environment
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

# Navigate to project directory
cd /home/rothermm/new_assessor

# Create logs directory
mkdir -p logs

# Print environment info
echo "=================================================="
echo "Ensemble Training: ViT + ResNet"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="
echo ""
echo "Environment:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run ensemble training with moderate augmentation
python scripts/03b_train_ensemble.py \
    --train_csv results/pca_analysis/ratings_pca_4comp_train.csv \
    --val_csv results/pca_analysis/ratings_pca_4comp_val.csv \
    --image_dir images \
    --use_pca \
    --use_decoder \
    --pca_model results/pca_analysis/pca_targets.joblib \
    --pca_components 4 \
    --freeze_decoder \
    --freeze_backbones \
    --augmentation moderate \
    --use_mixup \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --loss_fn smooth_l1 \
    --aux_loss_weight 0.2 \
    --use_amp \
    --output_dir results/ensemble_v2 \
    --seed 42

echo "Training complete!"