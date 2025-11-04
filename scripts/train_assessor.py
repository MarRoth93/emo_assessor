#!/usr/bin/env python3
"""
Train a deep learning model to predict psychological ratings from images.
This script trains a multi-output regression model that predicts 14 psychological dimensions.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class PsychologicalRatingDataset(Dataset):
    """Dataset for loading images and their psychological ratings."""
    
    def __init__(self, image_paths, ratings, transform=None, dimension_names=None):
        """
        Args:
            image_paths: List of paths to images
            ratings: numpy array of shape (n_samples, n_dimensions)
            transform: torchvision transforms to apply
            dimension_names: List of dimension names
        """
        self.image_paths = image_paths
        self.ratings = ratings
        self.transform = transform
        self.dimension_names = dimension_names
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        rating = torch.FloatTensor(self.ratings[idx])
        
        return image, rating


def get_train_transforms(image_size=224):
    """
    Extensive data augmentation for training.
    Applies various transformations to make the model more robust.
    """
    return transforms.Compose([
        # Geometric transformations
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        
        # Color augmentations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        
        # Additional augmentations
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Normalization (ImageNet stats)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Random erasing (dropout-like for images)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size=224):
    """Minimal transformations for validation/test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class MultiOutputRegressionModel(nn.Module):
    """
    Multi-output regression model using a pre-trained backbone.
    Predicts multiple psychological dimensions simultaneously.
    """
    
    def __init__(self, num_outputs=14, backbone='resnet50', pretrained=True, dropout=0.5):
        super(MultiOutputRegressionModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Regression head with dropout
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output


def load_data(images_dir, ratings_file):
    """Load and prepare the dataset."""
    print("Loading data...")
    
    # Read ratings CSV
    with open(ratings_file, 'r') as f:
        lines = f.readlines()
    
    # Fix the header
    header = lines[0].strip().replace('""', '"').replace('"', '')
    lines[0] = header + '\n'
    
    import io
    fixed_csv = io.StringIO(''.join(lines))
    df = pd.read_csv(fixed_csv)
    
    print(f"Loaded {len(df)} samples")
    
    # Extract dimension names
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    dimension_names = [col.replace('_mean', '') for col in mean_cols]
    
    print(f"Predicting {len(dimension_names)} dimensions: {dimension_names}")
    
    # Prepare image paths and ratings
    image_paths = []
    ratings = []
    
    for idx, row in df.iterrows():
        img_name = Path(row['image']).name
        img_path = Path(images_dir) / img_name
        
        if img_path.exists():
            image_paths.append(str(img_path))
            rating_values = [row[col] for col in mean_cols]
            ratings.append(rating_values)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    image_paths = np.array(image_paths)
    ratings = np.array(ratings, dtype=np.float32)
    
    print(f"Final dataset: {len(image_paths)} samples with {ratings.shape[1]} dimensions")
    print(f"Ratings range: [{ratings.min():.2f}, {ratings.max():.2f}]")
    
    return image_paths, ratings, dimension_names


def stratified_split(image_paths, ratings, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    Create stratified train/val/test split based on rating distributions.
    Stratifies based on the mean of all dimensions to ensure balanced splits.
    
    Args:
        image_paths: Array of image paths
        ratings: Array of ratings (n_samples, n_dimensions)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings
    """
    import numpy as np
    
    # Calculate mean rating across all dimensions for each image
    mean_ratings = ratings.mean(axis=1)
    
    # Create stratification bins (quintiles)
    n_bins = 5
    bins = np.percentile(mean_ratings, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 0.001  # Ensure last value is included
    strata = np.digitize(mean_ratings, bins[1:-1])
    
    # First split: separate test set
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio > 0:
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Split off test set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        train_val_idx, test_idx = next(sss.split(image_paths, strata))
        
        # Split train and val from the remaining data
        train_val_paths = image_paths[train_val_idx]
        train_val_ratings = ratings[train_val_idx]
        train_val_strata = strata[train_val_idx]
        
        test_paths = image_paths[test_idx]
        test_ratings = ratings[test_idx]
        
        # Adjust val_ratio relative to remaining data
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=random_state)
        train_idx, val_idx = next(sss.split(train_val_paths, train_val_strata))
        
        train_paths = train_val_paths[train_idx]
        train_ratings = train_val_ratings[train_idx]
        
        val_paths = train_val_paths[val_idx]
        val_ratings = train_val_ratings[val_idx]
    else:
        # Only train/val split
        from sklearn.model_selection import StratifiedShuffleSplit
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=random_state)
        train_idx, val_idx = next(sss.split(image_paths, strata))
        
        train_paths = image_paths[train_idx]
        train_ratings = ratings[train_idx]
        
        val_paths = image_paths[val_idx]
        val_ratings = ratings[val_idx]
        
        test_paths = np.array([])
        test_ratings = np.array([])
    
    return train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings


def print_split_statistics(train_ratings, val_ratings, test_ratings, dimension_names):
    """Print statistics about the data split to verify balance."""
    print("\n" + "="*80)
    print("DATA SPLIT STATISTICS")
    print("="*80)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_ratings)} samples ({len(train_ratings)/(len(train_ratings)+len(val_ratings)+len(test_ratings))*100:.1f}%)")
    print(f"  Validation: {len(val_ratings)} samples ({len(val_ratings)/(len(train_ratings)+len(val_ratings)+len(test_ratings))*100:.1f}%)")
    if len(test_ratings) > 0:
        print(f"  Test:       {len(test_ratings)} samples ({len(test_ratings)/(len(train_ratings)+len(val_ratings)+len(test_ratings))*100:.1f}%)")
    
    print(f"\nMean ratings per split (averaged across all dimensions):")
    train_mean = train_ratings.mean()
    val_mean = val_ratings.mean()
    print(f"  Training:   {train_mean:.3f} (std: {train_ratings.std():.3f})")
    print(f"  Validation: {val_mean:.3f} (std: {val_ratings.std():.3f})")
    if len(test_ratings) > 0:
        test_mean = test_ratings.mean()
        print(f"  Test:       {test_mean:.3f} (std: {test_ratings.std():.3f})")
    
    # Check per-dimension balance
    print(f"\nPer-dimension mean ratings:")
    print(f"{'Dimension':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Max Diff':<10}")
    print("-" * 60)
    
    for i, dim in enumerate(dimension_names):
        train_dim_mean = train_ratings[:, i].mean()
        val_dim_mean = val_ratings[:, i].mean()
        
        if len(test_ratings) > 0:
            test_dim_mean = test_ratings[:, i].mean()
            max_diff = max(train_dim_mean, val_dim_mean, test_dim_mean) - min(train_dim_mean, val_dim_mean, test_dim_mean)
            print(f"{dim:<15} {train_dim_mean:<10.3f} {val_dim_mean:<10.3f} {test_dim_mean:<10.3f} {max_diff:<10.3f}")
        else:
            max_diff = abs(train_dim_mean - val_dim_mean)
            print(f"{dim:<15} {train_dim_mean:<10.3f} {val_dim_mean:<10.3f} {'N/A':<10} {max_diff:<10.3f}")
    
    print("="*80 + "\n")


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate correlation coefficient
    correlations = []
    for i in range(all_preds.shape[1]):
        corr = np.corrcoef(all_preds[:, i], all_targets[:, i])[0, 1]
        correlations.append(corr)
    avg_corr = np.mean(correlations)
    
    return avg_loss, avg_corr


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate per-dimension metrics
    correlations = []
    maes = []
    for i in range(all_preds.shape[1]):
        corr = np.corrcoef(all_preds[:, i], all_targets[:, i])[0, 1]
        mae = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
        correlations.append(corr)
        maes.append(mae)
    
    avg_corr = np.mean(correlations)
    avg_mae = np.mean(maes)
    
    return avg_loss, avg_corr, avg_mae, correlations, maes


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Correlation
    axes[1].plot(history['train_corr'], label='Train Corr', linewidth=2)
    axes[1].plot(history['val_corr'], label='Val Corr', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Correlation', fontsize=12)
    axes[1].set_title('Training and Validation Correlation', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # MAE
    axes[2].plot(history['val_mae'], label='Val MAE', linewidth=2, color='orange')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('MAE', fontsize=12)
    axes[2].set_title('Validation Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_dimension_performance(dimension_names, correlations, maes, save_path):
    """Plot per-dimension performance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlations
    axes[0].barh(dimension_names, correlations, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Correlation', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Dimensions', fontsize=12, fontweight='bold')
    axes[0].set_title('Per-Dimension Correlation', fontsize=14, fontweight='bold')
    axes[0].axvline(np.mean(correlations), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # MAE
    axes[1].barh(dimension_names, maes, color='coral', alpha=0.7)
    axes[1].set_xlabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dimensions', fontsize=12, fontweight='bold')
    axes[1].set_title('Per-Dimension MAE', fontsize=14, fontweight='bold')
    axes[1].axvline(np.mean(maes), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(maes):.3f}')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Output directory: {output_dir}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    image_paths, ratings, dimension_names = load_data(args.images_dir, args.ratings_file)
    
    # Normalize ratings (optional but recommended)
    if args.normalize_targets:
        print("Normalizing target values...")
        scaler = StandardScaler()
        ratings_normalized = scaler.fit_transform(ratings)
        # Save scaler
        import pickle
        with open(output_dir / 'target_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        ratings_normalized = ratings
        scaler = None
    
    # Split data with stratification
    print(f"\nSplitting data with stratification...")
    print(f"  Train: {args.train_ratio*100:.0f}%")
    print(f"  Val: {args.val_ratio*100:.0f}%")
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio > 0.001:  # If test set requested
        print(f"  Test: {test_ratio*100:.0f}%")
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings = stratified_split(
            image_paths, ratings_normalized,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=42
        )
    else:
        # No test set, just train/val
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings = stratified_split(
            image_paths, ratings_normalized,
            train_ratio=args.train_ratio,
            val_ratio=1.0 - args.train_ratio,
            random_state=42
        )
    
    print(f"\nFinal split:")
    print(f"  Train samples: {len(train_paths)}")
    print(f"  Val samples: {len(val_paths)}")
    if len(test_paths) > 0:
        print(f"  Test samples: {len(test_paths)}")
    
    # Print split statistics to verify balance
    print_split_statistics(train_ratings, val_ratings, test_ratings, dimension_names)
    
    # Create datasets
    train_dataset = PsychologicalRatingDataset(
        train_paths, train_ratings,
        transform=get_train_transforms(args.image_size),
        dimension_names=dimension_names
    )
    
    val_dataset = PsychologicalRatingDataset(
        val_paths, val_ratings,
        transform=get_val_transforms(args.image_size),
        dimension_names=dimension_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create test dataloader if test set exists
    if len(test_paths) > 0:
        test_dataset = PsychologicalRatingDataset(
            test_paths, test_ratings,
            transform=get_val_transforms(args.image_size),
            dimension_names=dimension_names
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"Test loader created with {len(test_dataset)} samples")
    else:
        test_loader = None
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = MultiOutputRegressionModel(
        num_outputs=len(dimension_names),
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # Mixed precision training
    scaler_amp = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    history = {
        'train_loss': [], 'train_corr': [],
        'val_loss': [], 'val_corr': [], 'val_mae': []
    }
    
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_corr = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler_amp
        )
        
        # Validate
        val_loss, val_corr, val_mae, val_correlations, val_maes = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_corr'].append(train_corr)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)
        history['val_mae'].append(val_mae)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Corr: {train_corr:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f} | Val MAE: {val_mae:.4f}")
        
        # Learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr,
                'val_mae': val_mae,
                'dimension_names': dimension_names,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            
            print(f"✓ Saved best model (corr: {val_corr:.4f})")
            
            # Save per-dimension performance
            perf_df = pd.DataFrame({
                'Dimension': dimension_names,
                'Correlation': val_correlations,
                'MAE': val_maes
            })
            perf_df.to_csv(output_dir / 'best_model_per_dimension_performance.csv', index=False)
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Load best model and evaluate
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_corr, val_mae, val_correlations, val_maes = validate(
        model, val_loader, criterion, device
    )
    
    # Plot per-dimension performance
    plot_per_dimension_performance(
        dimension_names, val_correlations, val_maes,
        output_dir / 'per_dimension_performance.png'
    )
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("\n" + "="*80)
        print("FINAL EVALUATION ON TEST SET")
        print("="*80)
        
        test_loss, test_corr, test_mae, test_correlations, test_maes = validate(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Set Performance:")
        print(f"  Overall Correlation: {test_corr:.4f}")
        print(f"  Overall MAE: {test_mae:.4f}")
        print(f"  Overall Loss: {test_loss:.4f}")
        
        print("\nPer-dimension test performance:")
        for dim, corr, mae in zip(dimension_names, test_correlations, test_maes):
            print(f"  {dim:15s}: Corr={corr:.4f}, MAE={mae:.4f}")
        
        # Save test set results
        test_perf_df = pd.DataFrame({
            'Dimension': dimension_names,
            'Test_Correlation': test_correlations,
            'Test_MAE': test_maes
        })
        test_perf_df.to_csv(output_dir / 'test_set_performance.csv', index=False)
        
        # Plot test set performance
        plot_per_dimension_performance(
            dimension_names, test_correlations, test_maes,
            output_dir / 'test_set_performance.png'
        )
        
        print(f"\nTest results saved to: {output_dir / 'test_set_performance.csv'}")
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS ON VALIDATION SET")
    print("="*80)
    print(f"Overall Correlation: {val_corr:.4f}")
    print(f"Overall MAE: {val_mae:.4f}")
    print("\nPer-dimension performance:")
    for dim, corr, mae in zip(dimension_names, val_correlations, val_maes):
        print(f"  {dim:15s}: Corr={corr:.4f}, MAE={mae:.4f}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train psychological rating assessor')
    
    # Data arguments
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--ratings_file', type=str, required=True,
                        help='CSV file with ratings')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 
                                'efficientnet_b3', 'vit_b_16'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    
    # Data arguments
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7 for 70%%)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data (default: 0.15 for 15%%). Test ratio = 1 - train - val')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--normalize_targets', action='store_true', default=False,
                        help='Normalize target values')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
