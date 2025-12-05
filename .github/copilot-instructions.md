# Copilot Instructions - EMO Assessor

## Project Overview

This project trains deep learning models to predict 14 psychological dimensions (Approach, Arousal, Valence, etc.) from images. It uses a **PCA + Decoder architecture** to reduce dimensionality (14 dims → 4 PCs) for more stable training, with optional reconstruction back to 14 dimensions.

## Architecture Pipeline

```
1. 01_analyze_dimensions.py  → Analyze rating distributions
2. 02_compute_pca.py         → Fit PCA on train set ONLY (prevents data leakage)
3. 03_train_model.py         → Train CNN backbone + regression head
4. 08_analyze_experiments.py → Compare hyperparameter experiments
5. 09_evaluate_model.py      → Evaluate on test set
```

## Critical Data Flow

- **Raw ratings**: `ratings/per_image_Slider_mean_sd_from_wide.csv` - 14 dimensions with `_mean` and `_sd` columns
- **PCA-transformed**: `results/pca_analysis/ratings_pca_4comp_{train,val,test}.csv` - 4 PC scores
- **PCA model**: `results/pca_analysis/pca_targets.joblib` - Contains StandardScaler + PCA fitted on train only
- **Images**: `images/` directory (referenced as `stimuli/image_XXX.jpg` in CSVs)

## Training Modes

1. **PCA + Decoder (recommended)**: Predict 4 PC scores, reconstruct 14 dims via frozen/trainable decoder
   ```bash
   python scripts/03_train_model.py --use_pca --use_decoder --pca_model ./results/pca_analysis/pca_targets.joblib
   ```

2. **Direct 14-dim**: Predict all 14 dimensions directly (higher variance)

## Model Architecture

- **Backbones**: `resnet50`, `resnet101`, `efficientnet_b0`, `efficientnet_b3`, `vit_b_16`
- **Head**: 2-layer MLP (512 → 256 → n_outputs) with dropout
- **Decoder**: Linear layer initialized with PCA components (can be frozen or finetuned)

## Key Conventions

### Loss Functions
- Use `--loss_fn smooth_l1` for robustness to outliers (default in experiments)
- Auxiliary reconstruction loss (`--aux_loss_weight 0.2`) compares decoder output to original 14-dim ratings

### Transforms
- Training: Conservative augmentation (RandomCrop, HorizontalFlip, ColorJitter)
- Validation/Test: Only resize + normalize using torchvision weights' canonical transforms

### SLURM Execution
```bash
# Single experiment
sbatch slurm_scripts/experiments/exp_baseline.sh

# Full hyperparameter search
python scripts/07_run_hyperparameter_search.py --submit
```

## Common Patterns

### Loading Ratings CSV (quirky header format)
```python
with open(ratings_file, 'r') as f:
    lines = f.readlines()
header = lines[0].strip().replace('""', '"').replace('"', '')
lines[0] = header + '\n'
df = pd.read_csv(io.StringIO(''.join(lines)))
```

### Creating Train/Val/Test Splits
Splits are stratified by mean rating quintiles - always use `split_indices.json` for consistency:
```python
# results/pca_analysis/split_indices.json contains fixed indices
```

## Environment

- **Conda env**: `mediaeval`
- **GPU**: Required for reasonable training time; uses AMP (`--use_amp`)
- **Key deps**: PyTorch ≥2.0, torchvision, scikit-learn, pandas

## Best Performing Config (from experiments)

From `results/experiment_analysis/recommendations.txt`:
- **Backbone**: `vit_b_16`
- **LR**: 5e-05
- **Optimizer**: AdamW with cosine scheduler
- **Loss**: smooth_l1 with aux_loss_weight=0.2
