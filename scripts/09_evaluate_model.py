#!/usr/bin/env python3
"""
Comprehensive evaluation script for the trained assessor model.
Evaluates model on test set and creates detailed visualizations.

Supports:
- Single backbone models (ResNet, ViT, EfficientNet)
- Ensemble models (ViT + ResNet)
- PCA mode with optional decoder
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights, ViT_B_16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add project root to path for ensemble model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class PsychologicalRatingDataset(Dataset):
    """Dataset for loading images and their psychological ratings."""
    
    def __init__(self, image_paths, ratings, transform=None):
        self.image_paths = image_paths
        self.ratings = ratings
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        rating = torch.FloatTensor(self.ratings[idx])
        
        return image, rating, str(img_path)


def get_eval_transforms(image_size=224):
    """Get transforms for evaluation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


class RegressionHead(nn.Module):
    """
    Regression head with optional PCA decoder.
    """
    
    def __init__(self, in_channels, n_pc, n_out=14, use_decoder=False, pca_components=None):
        super().__init__()
        
        self.n_pc = n_pc
        self.n_out = n_out
        self.use_decoder = use_decoder
        
        # Main predictor: backbone features -> PC scores
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_pc)
        )
        
        # Optional decoder: PC scores -> reconstructed 14 dimensions
        if use_decoder:
            self.decoder = nn.Linear(n_pc, n_out, bias=False)
            
            if pca_components is not None:
                # nn.Linear expects weight shape (n_out, n_pc) for: y = x @ W^T + b
                # So we need to transpose
                self.decoder.weight.data = torch.from_numpy(pca_components.T).float()
                self.decoder.weight.requires_grad = False
        else:
            self.decoder = None
    
    def forward(self, x):
        pc_scores = self.predictor(x)
        
        if self.use_decoder:
            reconstructed = self.decoder(pc_scores)
            return pc_scores, reconstructed
        else:
            return pc_scores


class MultiOutputModel(nn.Module):
    """Multi-output regression model with optional PCA decoder."""
    
    def __init__(self, backbone='resnet50', num_outputs=14, dropout=0.5, pretrained=True,
                 use_pca=False, n_pc=4, pca_components=None, use_decoder=False):
        super().__init__()
        
        self.backbone_name = backbone
        self.use_pca = use_pca
        self.use_decoder = use_decoder
        
        # Load backbone with new API
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'vit_b_16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Regression head
        if use_pca and use_decoder:
            self.head = RegressionHead(
                num_features, n_pc, n_out=14,
                use_decoder=True,
                pca_components=pca_components
            )
        elif use_pca:
            self.head = RegressionHead(
                num_features, n_pc, n_out=14,
                use_decoder=False
            )
        else:
            self.head = RegressionHead(
                num_features, num_outputs, n_out=num_outputs,
                use_decoder=False
            )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


def load_data(images_dir, ratings_file, use_pca=False):
    """
    Load images and ratings.
    
    Args:
        images_dir: Directory containing images
        ratings_file: CSV file with ratings (original or PCA-transformed)
        use_pca: If True, expects PCA-transformed ratings (PC1, PC2, etc.)
    
    Returns:
        image_paths, ratings, dimension_names
    """
    print("Loading data...")
    
    # Read and fix the CSV header
    with open(ratings_file, 'r') as f:
        lines = f.readlines()
    
    # Fix the header - remove extra quotes
    header = lines[0].strip().replace('""', '"').replace('"', '')
    lines[0] = header + '\n'
    
    # Parse the fixed CSV
    import io
    fixed_csv = io.StringIO(''.join(lines))
    df = pd.read_csv(fixed_csv)
    
    # Extract dimension/component names based on data type
    if use_pca:
        # PCA-transformed data has columns like: image, PC1, PC2, PC3, ...
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        dimension_names = pc_cols
        dimension_cols = pc_cols
        print(f"Using PCA-transformed ratings with {len(dimension_names)} components")
    else:
        # Extract only the mean columns - these are the prediction targets
        # The _sd columns contain standard deviations of ratings, not targets
        mean_cols = [col for col in df.columns if col.endswith('_mean')]
        dimension_names = [col.replace('_mean', '') for col in mean_cols]
        dimension_cols = mean_cols
        print(f"Using original ratings with {len(dimension_names)} dimensions")
    
    # Get image filenames
    if 'filename' in df.columns:
        filenames = df['filename'].values
    elif 'image' in df.columns:
        filenames = df['image'].values
    elif 'Image' in df.columns:
        filenames = df['Image'].values
    else:
        raise ValueError(f"Could not find filename column in ratings file. Available columns: {df.columns.tolist()}")
    
    # Create full paths - handle both full paths and filenames
    image_paths = []
    for fname in filenames:
        # If fname is already a path with directory, get just the filename
        fname_only = os.path.basename(fname)
        full_path = os.path.join(images_dir, fname_only)
        image_paths.append(full_path)
    
    # Filter out missing images
    valid_indices = [i for i, path in enumerate(image_paths) if os.path.exists(path)]
    image_paths = [image_paths[i] for i in valid_indices]
    ratings = df[dimension_cols].values[valid_indices]
    
    print(f"Loaded {len(image_paths)} images")
    print(f"Predicting {len(dimension_names)} dimensions: {dimension_names}")
    
    return image_paths, ratings, dimension_names


def evaluate_model(model, dataloader, device, dimension_names, use_decoder=False):
    """Evaluate model and collect predictions. Handles decoder outputs if present."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_paths = []
    all_reconstructions = [] if use_decoder else None
    
    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # Ensemble model with decoder returns dict
                pc_pred = outputs['pc_scores']
                all_predictions.append(pc_pred.cpu().numpy())
                if use_decoder and 'reconstructed' in outputs:
                    all_reconstructions.append(outputs['reconstructed'].cpu().numpy())
            elif use_decoder and isinstance(outputs, tuple):
                # Original model with decoder returns tuple
                pc_pred, recon_14dim = outputs
                all_predictions.append(pc_pred.cpu().numpy())
                all_reconstructions.append(recon_14dim.cpu().numpy())
            else:
                # Direct tensor output
                all_predictions.append(outputs.cpu().numpy())
            
            all_targets.append(targets.cpu().numpy())
            all_paths.extend(paths)
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    reconstructions = np.vstack(all_reconstructions) if all_reconstructions else None
    
    return predictions, targets, all_paths, reconstructions


def compute_metrics(predictions, targets, dimension_names):
    """Compute comprehensive metrics for each dimension."""
    metrics = []
    
    for i, dim_name in enumerate(dimension_names):
        pred = predictions[:, i]
        true = targets[:, i]
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(pred, true)
        spearman_corr, spearman_p = spearmanr(pred, true)
        
        # Error metrics
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        
        # Relative error
        mean_true = np.mean(true)
        relative_mae = mae / mean_true if mean_true != 0 else np.inf
        
        metrics.append({
            'Dimension': dim_name,
            'Pearson_r': pearson_corr,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_corr,
            'Spearman_p': spearman_p,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Relative_MAE': relative_mae,
            'Mean_True': mean_true,
            'Mean_Pred': np.mean(pred),
            'Std_True': np.std(true),
            'Std_Pred': np.std(pred)
        })
    
    return pd.DataFrame(metrics)


def plot_predictions_vs_actual(predictions, targets, dimension_names, output_dir):
    """Create scatter plots of predictions vs actual for each dimension."""
    n_dims = len(dimension_names)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]
    
    for i, dim_name in enumerate(dimension_names):
        ax = axes[i]
        pred = predictions[:, i]
        true = targets[:, i]
        
        # Scatter plot
        ax.scatter(true, pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Compute correlation
        corr, _ = pearsonr(pred, true)
        mae = mean_absolute_error(true, pred)
        
        ax.set_xlabel('Actual Rating', fontsize=10)
        ax.set_ylabel('Predicted Rating', fontsize=10)
        ax.set_title(f'{dim_name}\nPearson r={corr:.3f}, MAE={mae:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions vs actual plot to {output_dir}")


def plot_residuals(predictions, targets, dimension_names, output_dir):
    """Plot residuals (errors) for each dimension."""
    n_dims = len(dimension_names)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]
    
    for i, dim_name in enumerate(dimension_names):
        ax = axes[i]
        pred = predictions[:, i]
        true = targets[:, i]
        residuals = pred - true
        
        # Histogram of residuals
        ax.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        
        ax.set_xlabel('Residual (Predicted - Actual)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{dim_name}\nMean={np.mean(residuals):.3f}, Std={np.std(residuals):.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals plot to {output_dir}")


def plot_metrics_summary(metrics_df, output_dir):
    """Create summary visualizations of all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sort by correlation
    metrics_sorted = metrics_df.sort_values('Pearson_r', ascending=False)
    
    # 1. Correlation comparison
    ax = axes[0, 0]
    x = np.arange(len(metrics_sorted))
    width = 0.35
    ax.bar(x - width/2, metrics_sorted['Pearson_r'], width, label='Pearson', alpha=0.8)
    ax.bar(x + width/2, metrics_sorted['Spearman_r'], width, label='Spearman', alpha=0.8)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Correlation by Dimension', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_sorted['Dimension'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.8)
    
    # 2. MAE and RMSE
    ax = axes[0, 1]
    x = np.arange(len(metrics_sorted))
    width = 0.35
    ax.bar(x - width/2, metrics_sorted['MAE'], width, label='MAE', alpha=0.8)
    ax.bar(x + width/2, metrics_sorted['RMSE'], width, label='RMSE', alpha=0.8)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Metrics by Dimension', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_sorted['Dimension'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. R² scores
    ax = axes[0, 2]
    colors = ['green' if r2 > 0 else 'red' for r2 in metrics_sorted['R2']]
    ax.barh(metrics_sorted['Dimension'], metrics_sorted['R2'], color=colors, alpha=0.7)
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title('R² Score by Dimension', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Relative MAE
    ax = axes[1, 0]
    ax.barh(metrics_sorted['Dimension'], metrics_sorted['Relative_MAE']*100, alpha=0.7, color='orange')
    ax.set_xlabel('Relative MAE (%)', fontsize=12)
    ax.set_title('Relative Error by Dimension', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Mean predictions vs actual
    ax = axes[1, 1]
    x = np.arange(len(metrics_sorted))
    width = 0.35
    ax.bar(x - width/2, metrics_sorted['Mean_True'], width, label='Actual', alpha=0.8)
    ax.bar(x + width/2, metrics_sorted['Mean_Pred'], width, label='Predicted', alpha=0.8)
    ax.set_ylabel('Mean Rating', fontsize=12)
    ax.set_title('Mean Ratings: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_sorted['Dimension'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Prediction variance
    ax = axes[1, 2]
    x = np.arange(len(metrics_sorted))
    width = 0.35
    ax.bar(x - width/2, metrics_sorted['Std_True'], width, label='Actual', alpha=0.8)
    ax.bar(x + width/2, metrics_sorted['Std_Pred'], width, label='Predicted', alpha=0.8)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('Rating Variance: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_sorted['Dimension'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics summary plot to {output_dir}")


def save_predictions(predictions, targets, paths, dimension_names, output_dir):
    """Save detailed predictions to CSV."""
    data = {'image_path': paths}
    
    for i, dim_name in enumerate(dimension_names):
        data[f'{dim_name}_actual'] = targets[:, i]
        data[f'{dim_name}_predicted'] = predictions[:, i]
        data[f'{dim_name}_error'] = predictions[:, i] - targets[:, i]
        data[f'{dim_name}_abs_error'] = np.abs(predictions[:, i] - targets[:, i])
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, 'detailed_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved detailed predictions to {output_path}")
    
    return df


def analyze_worst_predictions(predictions_df, dimension_names, output_dir, n_worst=10):
    """Identify and save worst predictions for each dimension."""
    worst_predictions = {}
    
    for dim_name in dimension_names:
        error_col = f'{dim_name}_abs_error'
        worst = predictions_df.nlargest(n_worst, error_col)[
            ['image_path', f'{dim_name}_actual', f'{dim_name}_predicted', error_col]
        ]
        worst_predictions[dim_name] = worst
    
    # Save to file
    output_path = os.path.join(output_dir, 'worst_predictions_analysis.txt')
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WORST PREDICTIONS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for dim_name in dimension_names:
            f.write(f"\n{dim_name} - Top {n_worst} Worst Predictions:\n")
            f.write("-" * 80 + "\n")
            worst = worst_predictions[dim_name]
            for idx, row in worst.iterrows():
                f.write(f"  Image: {os.path.basename(row['image_path'])}\n")
                f.write(f"    Actual: {row[f'{dim_name}_actual']:.3f}\n")
                f.write(f"    Predicted: {row[f'{dim_name}_predicted']:.3f}\n")
                f.write(f"    Error: {row[f'{dim_name}_abs_error']:.3f}\n\n")
    
    print(f"Saved worst predictions analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained assessor model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--images_dir', type=str, default='./images',
                       help='Directory containing images')
    parser.add_argument('--ratings_file', type=str, 
                       default='./ratings/per_image_Slider_mean_sd_from_wide.csv',
                       help='Path to ratings CSV file (original or PCA-transformed)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as checkpoint dir)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_pca', action='store_true', default=False,
                       help='Use PCA-transformed ratings (expects PC1, PC2, etc. columns)')
    parser.add_argument('--pca_model', type=str, default=None,
                       help='Path to PCA model joblib file (required if use_decoder=True)')
    parser.add_argument('--use_decoder', action='store_true', default=False,
                       help='Evaluate model with decoder (requires pca_model)')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'single', 'ensemble'],
                       help='Model type: auto-detect, single backbone, or ensemble')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(checkpoint_dir, 'evaluation')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("EVALUATING TRAINED MODEL")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {args.images_dir}")
    print(f"Ratings: {args.ratings_file}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Detect model type and decoder usage from checkpoint
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    is_ensemble = False
    has_decoder = False
    
    if args.model_type == 'auto':
        # Check if state_dict has ensemble-specific keys
        # EnsembleWithDecoder has 'encoder.vit' prefix, EnsembleRegressor has 'vit' directly
        if any('encoder.vit' in k or 'encoder.resnet' in k for k in state_dict_keys):
            is_ensemble = True
            has_decoder = True  # encoder. prefix means EnsembleWithDecoder
            print("Auto-detected: ENSEMBLE model with DECODER (EnsembleWithDecoder)")
        elif any(k.startswith('vit.') or k.startswith('resnet.') for k in state_dict_keys):
            is_ensemble = True
            has_decoder = False  # No encoder. prefix means plain EnsembleRegressor
            print("Auto-detected: ENSEMBLE model without decoder (EnsembleRegressor)")
        else:
            print("Auto-detected: SINGLE backbone model")
    elif args.model_type == 'ensemble':
        is_ensemble = True
        # Still auto-detect decoder from keys
        has_decoder = any('encoder.vit' in k for k in state_dict_keys)
        print(f"Ensemble model, decoder detected: {has_decoder}")
    
    # Override with explicit args if provided
    if args.use_decoder:
        has_decoder = True
    
    # Load PCA model if needed
    pca_dict = None
    if has_decoder or args.pca_model:
        if args.pca_model:
            print(f"\nLoading PCA model from {args.pca_model}...")
            pca_dict = joblib.load(args.pca_model)
            print(f"  PCA: {pca_dict['n_components']} components, {pca_dict['variance_explained']:.1%} variance")
        elif has_decoder:
            raise ValueError("Model was trained with decoder but --pca_model not provided. "
                           "Please specify --pca_model ./results/pca_analysis/pca_targets.joblib")
    
    # Load data
    image_paths, ratings, dimension_names = load_data(
        args.images_dir, 
        args.ratings_file,
        use_pca=args.use_pca
    )
    
    # Create dataset and loader
    transform = get_eval_transforms()
    dataset = PsychologicalRatingDataset(image_paths, ratings, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    
    print(f"\nEvaluating on {len(dataset)} samples with {len(dimension_names)} dimensions/components")
    
    # Create and load model based on type
    print("\nCreating model...")
    
    if is_ensemble:
        # Import ensemble model
        from scripts.models.ensemble_model import EnsembleRegressor, EnsembleWithDecoder
        
        n_pca_components = len(dimension_names)  # Should be 4 for PCA mode
        
        if has_decoder and pca_dict is not None:
            pca_components = torch.tensor(pca_dict['pca'].components_, dtype=torch.float32)
            pca_mean = torch.tensor(pca_dict['scaler_y'].mean_, dtype=torch.float32)
            
            model = EnsembleWithDecoder(
                n_pca_components=n_pca_components,
                n_original_dims=14,
                pca_components=pca_components,
                pca_mean=pca_mean,
                freeze_decoder=True,
                freeze_backbones=True,
                dropout=0.3,
                hidden_dim=512
            )
        else:
            model = EnsembleRegressor(
                n_outputs=n_pca_components,
                freeze_backbones=True,
                dropout=0.3,
                hidden_dim=512
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("Ensemble model loaded successfully")
        
    else:
        # Original single-backbone model loading
        model_config = checkpoint.get('model_config', {})
        checkpoint_args = checkpoint.get('args', {})
        
        backbone = model_config.get('backbone', checkpoint_args.get('backbone', 'resnet50'))
        num_outputs = checkpoint.get('dimension_names', model_config.get('num_outputs', 14))
        if isinstance(num_outputs, list):
            num_outputs = len(num_outputs)
        dropout = model_config.get('dropout', checkpoint_args.get('dropout', 0.5))
        
        use_pca_model = checkpoint.get('use_pca', checkpoint_args.get('use_pca', False))
        use_decoder_model = checkpoint.get('use_decoder', checkpoint_args.get('use_decoder', False))
        n_pc = checkpoint.get('n_pc', None)
        
        print(f"Model config: backbone={backbone}, outputs={num_outputs}, dropout={dropout}")
        
        pca_components = np.asarray(pca_dict['components'], dtype=np.float32) if pca_dict else None
        
        model = MultiOutputModel(
            backbone=backbone,
            num_outputs=num_outputs,
            dropout=dropout,
            pretrained=False,
            use_pca=use_pca_model or args.use_pca,
            n_pc=n_pc if n_pc else len(dimension_names),
            pca_components=pca_components,
            use_decoder=use_decoder_model or args.use_decoder
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("Single backbone model loaded successfully")
    
    # Evaluate
    print("\nRunning evaluation...")
    use_eval_decoder = args.use_decoder
    predictions, targets, paths, reconstructions = evaluate_model(
        model, dataloader, device, dimension_names, use_decoder=use_eval_decoder
    )
    print("Evaluation complete\n")
    
    # Compute metrics
    print("Computing metrics...")
    metrics_df = compute_metrics(predictions, targets, dimension_names)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}\n")
    
    # Print summary
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print()
    
    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Mean Pearson r: {metrics_df['Pearson_r'].mean():.4f} (±{metrics_df['Pearson_r'].std():.4f})")
    print(f"Mean MAE: {metrics_df['MAE'].mean():.4f} (±{metrics_df['MAE'].std():.4f})")
    print(f"Mean RMSE: {metrics_df['RMSE'].mean():.4f} (±{metrics_df['RMSE'].std():.4f})")
    print(f"Mean R²: {metrics_df['R2'].mean():.4f} (±{metrics_df['R2'].std():.4f})")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    plot_predictions_vs_actual(predictions, targets, dimension_names, args.output_dir)
    plot_residuals(predictions, targets, dimension_names, args.output_dir)
    plot_metrics_summary(metrics_df, args.output_dir)
    
    # Save detailed predictions
    print("\nSaving detailed predictions...")
    predictions_df = save_predictions(predictions, targets, paths, dimension_names, args.output_dir)
    
    # Analyze worst predictions
    print("Analyzing worst predictions...")
    analyze_worst_predictions(predictions_df, dimension_names, args.output_dir)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
