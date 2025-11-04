# Training Script Summary

## Overview
A complete deep learning pipeline to train an image-based psychological rating assessor that predicts 14 dimensions from images.

## Files Created

### 1. Main Training Script: `train_assessor.py`
**Purpose**: Full-featured training script with extensive data augmentation

**Key Features**:
- Multi-output regression model (predicts 14 dimensions simultaneously)
- Multiple backbone options (ResNet50/101, EfficientNet B0/B3, ViT)
- Extensive data augmentation:
  - Geometric: rotation, flipping, cropping, affine, perspective
  - Color: brightness, contrast, saturation, hue adjustments
  - Random grayscale and random erasing
- Mixed precision training support
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate schedulers (Cosine, Step, Plateau)
- Early stopping
- Comprehensive logging and visualization
- Per-dimension performance tracking

**Usage**:
```bash
python train_assessor.py \
    --images_dir ./images \
    --ratings_file ./ratings/per_image_Slider_mean_sd_from_wide.csv \
    --output_dir ./outputs \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 32
```

### 2. SLURM Batch Script: `train_assessor.sh`
**Purpose**: Submit training job to SLURM cluster

**Configuration**:
- 24 hours max time
- 32GB RAM
- 8 CPUs
- 1 GPU
- Automatic GPU detection and logging

**Usage**:
```bash
sbatch train_assessor.sh
```

### 3. Local Run Script: `run_local.sh`
**Purpose**: Simple script for local training (without SLURM)

**Usage**:
```bash
./run_local.sh
```

### 4. Inference Script: `predict_assessor.py`
**Purpose**: Use trained model to predict ratings for new images

**Features**:
- Batch prediction support
- Automatic denormalization
- Support for directory or file list input
- Summary statistics output

**Usage**:
```bash
python predict_assessor.py \
    --checkpoint outputs/run_YYYYMMDD_HHMMSS/best_model.pth \
    --image_dir path/to/images \
    --output predictions.csv
```

### 5. Setup Verification: `verify_setup.py`
**Purpose**: Test environment and data before training

**Checks**:
- Python version
- Required packages
- CUDA/GPU availability
- Data files existence
- Data loading functionality
- Model creation

**Usage**:
```bash
python verify_setup.py
```

### 6. Requirements: `requirements.txt`
**Purpose**: List of Python dependencies

**Install**:
```bash
pip install -r requirements.txt
```

### 7. Documentation: `README.md`
**Purpose**: Comprehensive user guide

**Contents**:
- Project overview
- Installation instructions
- Training guide
- Parameter descriptions
- Inference guide
- Troubleshooting tips

## Data Augmentation Strategy

### Training Augmentations (Applied):
1. **Random Crop**: From 256x256 to 224x224
2. **Random Horizontal Flip**: 50% probability
3. **Random Rotation**: ±15 degrees
4. **Random Affine**: Translation (±10%), Scale (90-110%), Shear (±10°)
5. **Color Jitter**: Brightness/Contrast/Saturation (±30%), Hue (±10%)
6. **Random Grayscale**: 10% probability
7. **Random Perspective**: 20% distortion, 30% probability
8. **Random Erasing**: 20% probability, 2-15% area

### Validation Augmentations:
- Resize to 224x224 (center crop)
- Normalize only

## Model Architecture

```
Input Image (3 x 224 x 224)
        ↓
Pretrained Backbone (ResNet/EfficientNet/ViT)
        ↓
Feature Vector (2048 for ResNet50)
        ↓
Linear(2048 → 512) + ReLU + Dropout(0.5)
        ↓
Linear(512 → 256) + ReLU + Dropout(0.5)
        ↓
Linear(256 → 14)
        ↓
Output: 14 psychological ratings
```

## Training Process

1. **Data Split**: 80% train, 20% validation
2. **Loss Function**: Mean Squared Error (MSE)
3. **Optimization**: AdamW with cosine learning rate schedule
4. **Evaluation**: Pearson correlation + MAE per dimension
5. **Best Model**: Saved based on highest validation correlation
6. **Early Stopping**: Stops if no improvement for 15 epochs

## Expected Outputs

Each training run creates a timestamped folder in `outputs/` containing:

```
outputs/run_20251104_123456/
├── best_model.pth                              # Best model checkpoint
├── checkpoint_epoch_10.pth                     # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── args.json                                   # Training configuration
├── training_history.csv                        # Epoch-by-epoch metrics
├── training_history.png                        # Loss/correlation curves
├── per_dimension_performance.png               # Per-dimension bar charts
├── best_model_per_dimension_performance.csv    # Detailed metrics
└── target_scaler.pkl                          # (if normalization used)
```

## Performance Metrics

The model is evaluated using:
1. **Overall Correlation**: Average Pearson correlation across all dimensions
2. **Per-Dimension Correlation**: Individual correlations for each dimension
3. **Mean Absolute Error (MAE)**: Average prediction error
4. **Mean Squared Error (MSE)**: Training loss

## Typical Training Time

- **ResNet50**: ~2-3 minutes per epoch (with GPU)
- **EfficientNet-B3**: ~3-4 minutes per epoch
- **ViT**: ~4-5 minutes per epoch
- **50 epochs**: ~2-4 hours total (with GPU)
- **CPU**: 10-20x slower (not recommended)

## Hardware Requirements

### Minimum:
- GPU: 6GB VRAM (e.g., GTX 1060)
- RAM: 16GB
- Storage: 5GB

### Recommended:
- GPU: 11GB+ VRAM (e.g., RTX 2080 Ti, RTX 3080)
- RAM: 32GB
- Storage: 10GB

## Quick Start

1. **Verify setup**:
   ```bash
   python verify_setup.py
   ```

2. **Start training**:
   ```bash
   # On cluster:
   sbatch train_assessor.sh
   
   # Locally:
   ./run_local.sh
   ```

3. **Monitor training**:
   ```bash
   tail -f logs/train_assessor_*.out
   ```

4. **Use trained model**:
   ```bash
   python predict_assessor.py \
       --checkpoint outputs/run_*/best_model.pth \
       --image_dir new_images/
   ```

## Tips for Success

1. **Start with default parameters** - they're well-tuned
2. **Use a GPU** - training on CPU is impractically slow
3. **Monitor overfitting** - watch validation metrics
4. **Try different backbones** - EfficientNet often works well
5. **Increase epochs if needed** - 50 may not be enough for convergence
6. **Use early stopping** - prevents wasting time on overfitting

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch_size` to 16 or 8 |
| Training too slow | Use `--backbone efficientnet_b0` or enable `--use_amp` |
| Overfitting | Increase `--dropout` to 0.7 or reduce learning rate |
| Poor performance | Try different backbone, increase epochs, or check data quality |

## Next Steps

After training completes:
1. Check `training_history.png` for convergence
2. Review `per_dimension_performance.png` for per-dimension quality
3. Use best model for inference on new images
4. If performance is poor, try different hyperparameters or backbone
5. Consider ensemble methods (train multiple models and average predictions)
