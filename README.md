# Psychological Rating Assessor

A deep learning model that predicts 14 psychological dimensions from images.

## Dimensions Predicted

The model predicts the following psychological dimensions:
- **Approach**: Tendency to approach or engage with the stimulus
- **Arousal**: Level of emotional/physiological activation
- **Attention**: Degree of attentional capture
- **Certainty**: Feeling of certainty or confidence
- **Commitment**: Level of commitment or dedication
- **Control**: Sense of control over the situation
- **Dominance**: Feeling of dominance or power
- **Effort**: Amount of effort required or expended
- **Fairness**: Perception of fairness
- **Identity**: Relevance to personal identity
- **Obstruction**: Sense of obstruction or blocking
- **Safety**: Feeling of safety or security
- **Upswing**: Positive momentum or uplift
- **Valence**: Overall positive/negative emotional tone

## Project Structure

```
new_assessor/
├── train_assessor.py          # Main training script
├── predict_assessor.py         # Inference script for new images
├── train_assessor.sh          # SLURM batch script for cluster
├── run_local.sh               # Simple script for local training
├── requirements.txt           # Python dependencies
├── images/                    # Training images (600 images)
├── ratings/                   # CSV file with ratings
│   └── per_image_Slider_mean_sd_from_wide.csv
└── outputs/                   # Training outputs (created automatically)
    └── run_YYYYMMDD_HHMMSS/
        ├── best_model.pth
        ├── training_history.csv
        ├── training_history.png
        ├── per_dimension_performance.png
        └── ...
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using conda:
```bash
conda install pytorch torchvision -c pytorch
pip install pandas scikit-learn matplotlib seaborn tqdm pillow
```

## Training

### Option 1: SLURM Cluster (Recommended)

Submit the job to the cluster:
```bash
sbatch train_assessor.sh
```

Monitor the job:
```bash
squeue -u $USER
tail -f logs/train_assessor_JOBID.out
```

### Option 2: Local Training

Make scripts executable:
```bash
chmod +x train_assessor.sh run_local.sh
```

Run locally:
```bash
./run_local.sh
```

### Option 3: Custom Training

Run with custom parameters:
```bash
python train_assessor.py \
    --images_dir ./images \
    --ratings_file ./ratings/per_image_Slider_mean_sd_from_wide.csv \
    --output_dir ./outputs \
    --backbone resnet50 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001
```

## Training Parameters

### Model Architectures (--backbone)
- `resnet50` (default): Good balance of speed and accuracy
- `resnet101`: Deeper ResNet, more capacity
- `efficientnet_b0`: Efficient and fast
- `efficientnet_b3`: More accurate EfficientNet
- `vit_b_16`: Vision Transformer (slower but potentially more accurate)

### Key Hyperparameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--optimizer`: Optimizer (adam, adamw, sgd) (default: adamw)
- `--scheduler`: LR scheduler (cosine, step, plateau, none) (default: cosine)
- `--dropout`: Dropout rate (default: 0.5)
- `--train_ratio`: Train/val split ratio (default: 0.8)
- `--early_stopping`: Early stopping patience (default: 15)

## Data Augmentation

The training script uses extensive data augmentation to improve model robustness:

1. **Geometric Transformations**:
   - Random cropping
   - Random horizontal flipping (50% probability)
   - Random rotation (±15°)
   - Random affine transformations (translation, scaling, shearing)
   - Random perspective distortion

2. **Color Augmentations**:
   - Brightness adjustment (±30%)
   - Contrast adjustment (±30%)
   - Saturation adjustment (±30%)
   - Hue adjustment (±10%)
   - Random grayscale conversion (10% probability)

3. **Other Augmentations**:
   - Random erasing (20% probability) - simulates occlusion

All augmentations are only applied during training. Validation uses center cropping only.

## Inference

Use the trained model to predict ratings for new images:

### Predict for a directory of images:
```bash
python predict_assessor.py \
    --checkpoint outputs/run_YYYYMMDD_HHMMSS/best_model.pth \
    --image_dir path/to/new/images \
    --output predictions.csv
```

### Predict for a list of images:
```bash
python predict_assessor.py \
    --checkpoint outputs/run_YYYYMMDD_HHMMSS/best_model.pth \
    --image_list image_paths.txt \
    --output predictions.csv
```

The output CSV will contain:
- `image_path`: Full path to the image
- `image_name`: Image filename
- One column for each dimension with predicted ratings

## Outputs

Each training run creates a timestamped directory in `outputs/` containing:

1. **Model Files**:
   - `best_model.pth`: Best model based on validation correlation
   - `checkpoint_epoch_N.pth`: Periodic checkpoints

2. **Training Logs**:
   - `training_history.csv`: Loss and metrics per epoch
   - `args.json`: Training configuration

3. **Visualizations**:
   - `training_history.png`: Loss, correlation, and MAE curves
   - `per_dimension_performance.png`: Bar charts for each dimension
   - `best_model_per_dimension_performance.csv`: Detailed per-dimension metrics

4. **Other**:
   - `target_scaler.pkl`: Scaler for denormalization (if --normalize_targets used)

## Performance Metrics

The model is evaluated using:
1. **Pearson Correlation**: Correlation between predicted and actual ratings
2. **Mean Absolute Error (MAE)**: Average absolute difference
3. **Mean Squared Error (MSE)**: Loss function

Metrics are computed both overall (averaged across dimensions) and per-dimension.

## Tips for Better Performance

1. **Use a GPU**: Training on CPU is very slow
2. **Increase batch size**: If you have GPU memory, use larger batches (64, 128)
3. **Try different backbones**: EfficientNet or ViT might work better for your data
4. **Enable mixed precision**: Use `--use_amp` for faster training on modern GPUs
5. **Normalize targets**: Use `--normalize_targets` if rating scales vary widely
6. **Increase epochs**: 50 epochs might not be enough - try 100+
7. **Early stopping**: Prevents overfitting by stopping when validation stops improving

## Dataset Information

- **Total images**: 600
- **Training images**: 480 (80%)
- **Validation images**: 120 (20%)
- **Dimensions**: 14 psychological ratings per image
- **Rating scale**: Typically 1-8 or similar (check your data)

## Troubleshooting

### Out of Memory (OOM) Error
```bash
# Reduce batch size
python train_assessor.py --batch_size 16 ...
```

### Training too slow
```bash
# Use a smaller/faster model
python train_assessor.py --backbone efficientnet_b0 ...

# Enable mixed precision (requires GPU)
python train_assessor.py --use_amp ...
```

### Model overfitting
```bash
# Increase dropout
python train_assessor.py --dropout 0.7 ...

# Reduce learning rate
python train_assessor.py --lr 0.00005 ...

# Enable early stopping (already default)
python train_assessor.py --early_stopping 10 ...
```

## Citation

If you use this code, please cite your work appropriately.

## License

See LICENSE file for details.
