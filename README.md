# EMO Assessor

Deep learning models to predict 14 psychological dimensions from images. Uses a **PCA + Decoder architecture** to reduce dimensionality (14 dims → 4 PCs) for more stable training with small datasets.

## 🎯 Task

Predict psychological ratings for images across 14 dimensions:
- **Affective**: Valence, Arousal, Approach
- **Cognitive**: Attention, Certainty, Control, Dominance
- **Social**: Fairness, Identity, Commitment
- **Situational**: Effort, Obstruction, Safety, Upswing

## 📁 Project Structure

```
├── scripts/
│   ├── 01_analyze_dimensions.py    # Analyze rating distributions
│   ├── 02_compute_pca.py           # Fit PCA (train set only)
│   ├── 03_train_model.py           # Train single-backbone models
│   ├── 03b_train_ensemble.py       # Train ViT+ResNet ensemble
│   ├── 05_predict_new_images.py    # Inference on new images
│   ├── 07_run_hyperparameter_search.py  # Launch experiments
│   ├── 08_analyze_experiments.py   # Compare experiment results
│   ├── 09_evaluate_model.py        # Evaluate on test set
│   ├── models/
│   │   └── ensemble_model.py       # ViT+ResNet ensemble
│   └── datasets/
│       ├── augmentation.py         # Augmentation strategies
│       └── image_dataset.py        # Dataset class
├── slurm_scripts/                  # SLURM job scripts
├── ratings/                        # Raw rating CSVs
├── results/
│   ├── pca_analysis/              # PCA model & transformed ratings
│   └── experiment_analysis/       # Hyperparameter search results
├── outputs/                        # Trained models
└── images/                         # Image dataset
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda activate mediaeval
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Analyze rating distributions
python scripts/01_analyze_dimensions.py

# Compute PCA on training set (prevents data leakage)
python scripts/02_compute_pca.py --n_components 4
```

### 3. Training

**Single Backbone (ResNet, ViT, EfficientNet):**
```bash
python scripts/03_train_model.py \
    --images_dir ./images \
    --ratings_file ./results/pca_analysis/ratings_pca_4comp_train.csv \
    --use_pca --use_decoder \
    --pca_model ./results/pca_analysis/pca_targets.joblib \
    --backbone vit_b_16 \
    --loss_fn smooth_l1 \
    --use_amp
```

**Ensemble Model (ViT + ResNet):**
```bash
python scripts/03b_train_ensemble.py \
    --train_csv ./results/pca_analysis/ratings_pca_4comp_train.csv \
    --val_csv ./results/pca_analysis/ratings_pca_4comp_val.csv \
    --use_pca --use_decoder \
    --augmentation moderate \
    --use_mixup
```

**SLURM Cluster:**
```bash
sbatch slurm_scripts/03b_train_ensemble.sh
```

### 4. Evaluation

```bash
python scripts/09_evaluate_model.py \
    --checkpoint ./results/ensemble_v2/best_model.pt \
    --ratings_file ./results/pca_analysis/ratings_pca_4comp_test.csv \
    --use_pca \
    --model_type ensemble
```

## 🏗️ Architecture

### PCA + Decoder Pipeline

```
Image → Backbone → 4 PC Scores → Decoder → 14 Dimensions
         ↓              ↓            ↓
    (frozen/        (primary      (aux loss
     finetune)       loss)        λ=0.2)
```

- **Why PCA?** 14 dimensions are highly correlated; PCA reduces to 4 independent components
- **Why Decoder?** Allows end-to-end training while outputting interpretable 14-dim ratings
- **Decoder initialization**: PCA components (linear projection)

### Ensemble Model

```
Image → ViT-B/16 ──┐
                   ├── Concat (2816 dims) → MLP → 4 PCs
Image → ResNet50 ──┘
```

- ViT captures global semantics
- ResNet captures local texture/color
- Combined features improve generalization on small datasets

## 📊 Key Files

| File | Description |
|------|-------------|
| `ratings/per_image_Slider_mean_sd_from_wide.csv` | Raw 14-dim ratings |
| `results/pca_analysis/pca_targets.joblib` | PCA model (scaler + components) |
| `results/pca_analysis/ratings_pca_4comp_{train,val,test}.csv` | Split PCA ratings |
| `results/pca_analysis/split_indices.json` | Fixed train/val/test split |

## ⚙️ Configuration

### Best Hyperparameters (from experiments)

| Parameter | Value |
|-----------|-------|
| Backbone | `vit_b_16` (single) or ensemble |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Loss | SmoothL1 |
| Aux Loss Weight | 0.2 |
| Dropout | 0.3-0.5 |
| Augmentation | Moderate |

### Augmentation Levels

- **Conservative**: RandomCrop, HorizontalFlip, light ColorJitter
- **Moderate**: + RandomRotation, RandomErasing, stronger ColorJitter
- **Strong**: + GaussianBlur, more aggressive crops

## 📈 Expected Performance

Given inter-rater variability in psychological ratings:

| Metric | Realistic Target | Theoretical Ceiling |
|--------|------------------|---------------------|
| R² | 0.35-0.45 | ~0.55-0.65 |
| Correlation | 0.60-0.70 | ~0.75-0.80 |

*Note: High rating SD (0.8-1.0 on 7-point scale) limits achievable accuracy.*

## 🔧 Troubleshooting

### CSV Header Issues
The ratings CSV has quirky formatting. Use this pattern:
```python
with open(ratings_file, 'r') as f:
    lines = f.readlines()
header = lines[0].strip().replace('""', '"').replace('"', '')
lines[0] = header + '\n'
df = pd.read_csv(io.StringIO(''.join(lines)))
```

### GPU Memory
- Use `--use_amp` for mixed precision
- Reduce `--batch_size` if OOM
- Freeze backbones with `--freeze_backbones`

## 📝 Citation

*Add citation info here*

## 📄 License

*Add license info here*
