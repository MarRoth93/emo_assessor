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
from torchvision.models import ResNet50_Weights, ResNet101_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights, ViT_B_16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

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
    
    def __init__(self, image_paths, ratings, transform=None, dimension_names=None, return_index=False):
        """
        Args:
            image_paths: List of paths to images
            ratings: numpy array of shape (n_samples, n_dimensions)
            transform: torchvision transforms to apply
            dimension_names: List of dimension names
            return_index: If True, return (image, rating, idx) instead of (image, rating)
        """
        self.image_paths = image_paths
        self.ratings = ratings
        self.transform = transform
        self.dimension_names = dimension_names
        self.return_index = return_index
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        rating = torch.FloatTensor(self.ratings[idx])
        
        if self.return_index:
            return image, rating, idx
        return image, rating


def get_transforms(backbone, pretrained, image_size=224, train=True):
    """
    Get transforms using weight-specific preprocessing for best accuracy.
    Uses canonical transforms from torchvision weights when available.
    
    Args:
        backbone: Model backbone name
        pretrained: Whether using pretrained weights
        image_size: Target image size
        train: If True, add training augmentations
    
    Returns:
        Composed transforms
    """
    # Get the appropriate weights for each backbone
    weights = None
    if pretrained:
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V2
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        elif backbone == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        elif backbone == 'vit_b_16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1
    
    # Use canonical transforms from weights if available
    if weights is not None:
        base_transform = weights.transforms(antialias=True)
    else:
        # Fallback to manual transforms
        base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    if not train:
        # Validation/test: use base transforms only
        return base_transform
    
    # Training: add conservative geometric and color augmentations
    # Keep augmentations light to preserve semantic content for rating tasks
    aug_transforms = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    
    # Combine augmentations with base transform
    # Note: base_transform includes Resize, ToTensor, and Normalize
    return transforms.Compose([aug_transforms, base_transform])


def get_train_transforms(image_size=224):
    """
    Conservative data augmentation for training.
    Reduced from aggressive settings to preserve semantic content.
    
    DEPRECATED: Use get_transforms(backbone, pretrained, image_size, train=True) instead
    """
    return transforms.Compose([
        # Geometric transformations (reduced)
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),  # Reduced from 15
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # Reduced from 0.1
            scale=(0.95, 1.05),  # Reduced from (0.9, 1.1)
            shear=None  # Removed shear
        ),
        
        # Color augmentations (reduced)
        transforms.ColorJitter(
            brightness=0.1,  # Reduced from 0.3
            contrast=0.1,  # Reduced from 0.3
            saturation=0.1,  # Reduced from 0.3
            hue=0.05  # Reduced from 0.1
        ),
        
        # Keep minimal augmentations
        transforms.RandomGrayscale(p=0.05),  # Reduced from 0.1
        
        # Normalization (ImageNet stats)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Removed RandomErasing and RandomPerspective
    ])


def get_val_transforms(image_size=224):
    """
    Minimal transformations for validation/test.
    
    DEPRECATED: Use get_transforms(backbone, pretrained, image_size, train=False) instead
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class RegressionHead(nn.Module):
    """
    Regression head with optional PCA decoder.
    
    Predicts PC scores and optionally reconstructs original 14 dimensions.
    The decoder uses PCA components to transform PC scores back to original space.
    """
    
    def __init__(self, in_channels, n_pc, n_out=14, use_decoder=False, pca_components=None):
        """
        Args:
            in_channels: Number of input features from backbone
            n_pc: Number of PC components to predict
            n_out: Number of original dimensions (14)
            use_decoder: Whether to use decoder for reconstructing original dimensions
            pca_components: PCA components matrix (n_pc, n_out) for decoder initialization
        """
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
            
            # Initialize decoder with PCA components if provided
            if pca_components is not None:
                # pca_components shape: (n_pc, n_out)
                # nn.Linear expects weight shape (n_out, n_pc) for: y = x @ W^T + b
                # So we need to transpose: y_reconstructed = PC @ components^T
                self.decoder.weight.data = torch.from_numpy(pca_components.T).float()
                self.decoder.weight.requires_grad = False  # Start frozen
                print(f"  ✓ Initialized decoder with PCA components (frozen, transposed)")
            else:
                print(f"  ⚠ Decoder created but not initialized with PCA components")
        else:
            self.decoder = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            If use_decoder: (pc_scores, reconstructed_14dim)
            Otherwise: pc_scores
        """
        pc_scores = self.predictor(x)
        
        if self.use_decoder:
            reconstructed = self.decoder(pc_scores)
            return pc_scores, reconstructed
        else:
            return pc_scores
    
    def unfreeze_decoder(self):
        """Allow decoder weights to be trained."""
        if self.decoder is not None:
            self.decoder.weight.requires_grad = True
            print("  ✓ Decoder weights unfrozen")
    
    def freeze_decoder(self):
        """Freeze decoder weights."""
        if self.decoder is not None:
            self.decoder.weight.requires_grad = False
            print("  ✓ Decoder weights frozen")


class MultiOutputRegressionModel(nn.Module):
    """
    Multi-output regression model using a pre-trained backbone.
    Supports both direct 14-dim prediction and PCA-based prediction with decoder.
    """
    
    def __init__(self, num_outputs=14, backbone='resnet50', pretrained=True, dropout=0.5,
                 use_pca=False, n_pc=4, pca_components=None, use_decoder=False):
        """
        Args:
            num_outputs: Number of output dimensions (14 for direct, or n_pc for PCA)
            backbone: Backbone architecture name
            pretrained: Use pretrained weights
            dropout: Dropout rate
            use_pca: Whether using PCA mode
            n_pc: Number of PC components (only used if use_pca=True)
            pca_components: PCA components for decoder initialization
            use_decoder: Whether to use decoder for reconstruction
        """
        super(MultiOutputRegressionModel, self).__init__()
        
        self.backbone_name = backbone
        self.use_pca = use_pca
        self.use_decoder = use_decoder
        
        # Load pre-trained backbone with new API
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
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
            # PCA mode with decoder
            self.head = RegressionHead(
                num_features, n_pc, n_out=14,
                use_decoder=True,
                pca_components=pca_components
            )
        elif use_pca:
            # PCA mode without decoder
            self.head = RegressionHead(
                num_features, n_pc, n_out=14,
                use_decoder=False
            )
        else:
            # Direct 14-dim prediction
            self.head = RegressionHead(
                num_features, num_outputs, n_out=num_outputs,
                use_decoder=False
            )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


def load_pca_model(pca_model_path):
    """
    Load PCA model from joblib file.
    
    Returns:
        Dictionary containing scaler_y, pca, dimension_names, etc.
    """
    print(f"Loading PCA model from {pca_model_path}...")
    pca_dict = joblib.load(pca_model_path)
    
    print(f"  ✓ PCA has {pca_dict['n_components']} components")
    print(f"  ✓ Explains {pca_dict['variance_explained']:.1%} of variance")
    print(f"  ✓ Original dimensions: {pca_dict['dimension_names']}")
    
    return pca_dict


def load_data(images_dir, ratings_file, use_pca=False, pca_model_path=None, use_decoder=False):
    """
    Load and prepare the dataset.
    
    Args:
        images_dir: Directory containing images
        ratings_file: CSV file with ratings (original or PCA-transformed from compute_pca.py)
        use_pca: If True, expects PCA-transformed ratings (PC1, PC2, etc.)
        pca_model_path: Path to PCA model joblib file (required if use_decoder=True)
        use_decoder: Whether decoder will be used (requires loading original 14-dim ratings too)
    
    Returns:
        image_paths, ratings, dimension_names, (optional) ratings_14dim, pca_dict
    """
    print("Loading data...")
    
    # Load PCA model if needed
    pca_dict = None
    if pca_model_path:
        pca_dict = load_pca_model(pca_model_path)
    
    # Read ratings CSV
    with open(ratings_file, 'r') as f:
        lines = f.readlines()
    
    # Fix the header (for original ratings files with extra quotes)
    header = lines[0].strip().replace('""', '"').replace('"', '')
    lines[0] = header + '\n'
    
    import io
    fixed_csv = io.StringIO(''.join(lines))
    df = pd.read_csv(fixed_csv)
    
    print(f"Loaded {len(df)} samples")
    
    # Extract dimension/component names based on data type
    if use_pca:
        # PCA-transformed data has columns like: image, PC1, PC2, PC3, ...
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        dimension_names = pc_cols
        print(f"Using PCA-transformed ratings with {len(dimension_names)} components: {dimension_names}")
        rating_cols = dimension_names
    else:
        # Original data has columns like: image, Approach_mean, Arousal_mean, ...
        mean_cols = [col for col in df.columns if col.endswith('_mean')]
        dimension_names = [col.replace('_mean', '') for col in mean_cols]
        print(f"Using original ratings with {len(dimension_names)} dimensions: {dimension_names}")
        rating_cols = [f"{dim}_mean" for dim in dimension_names]
    
    # Prepare image paths and ratings
    image_paths = []
    ratings = []
    
    for idx, row in df.iterrows():
        img_name = Path(row['image']).name
        img_path = Path(images_dir) / img_name
        
        if img_path.exists():
            image_paths.append(str(img_path))
            rating_values = [row[col] for col in rating_cols]
            ratings.append(rating_values)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    image_paths = np.array(image_paths)
    ratings = np.array(ratings, dtype=np.float32)
    
    print(f"Final dataset: {len(image_paths)} samples with {ratings.shape[1]} outputs")
    print(f"Ratings range: [{ratings.min():.2f}, {ratings.max():.2f}]")
    
    # If using decoder, we also need the standardized 14-dim ratings for auxiliary loss
    ratings_14dim_standardized = None
    if use_decoder and pca_dict is not None and use_pca:
        print("\nLoading original 14-dim ratings for decoder auxiliary loss...")
        # Need to load the original ratings file and standardize with saved scaler
        # For now, we'll compute them on the fly during training
        # This will be handled in the training loop
        pass
    
    return image_paths, ratings, dimension_names, pca_dict


def stratified_split(image_paths, ratings, train_ratio=0.7, val_ratio=0.15, random_state=42, use_pc1_stratification=False):
    """
    Create stratified train/val/test split based on rating distributions.
    
    Args:
        image_paths: Array of image paths
        ratings: Array of ratings (n_samples, n_dimensions)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_state: Random seed for reproducibility
        use_pc1_stratification: If True, stratify on PC1 (first column).
                                If False, stratify on mean of all dimensions.
    
    Returns:
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings
    """
    import numpy as np
    
    # Calculate stratification values
    if use_pc1_stratification:
        # Stratify based on PC1 (first column) when using PCA
        strat_values = ratings[:, 0]
        print("  Stratifying on PC1 quantiles")
    else:
        # Stratify based on mean rating across all dimensions
        strat_values = ratings.mean(axis=1)
        print("  Stratifying on mean of all dimensions")
    
    # Create stratification bins (quintiles)
    n_bins = 5
    bins = np.percentile(strat_values, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 0.001  # Ensure last value is included
    strata = np.digitize(strat_values, bins[1:-1])
    
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


def compute_safe_correlation(pred, target):
    """Compute correlation with NaN guard for zero variance."""
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return 0.0  # Return 0 correlation if no variance
    corr = np.corrcoef(pred, target)[0, 1]
    return 0.0 if np.isnan(corr) else corr


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, 
                use_decoder=False, aux_loss_weight=0.2, ratings_14dim_z=None,
                grad_clip=1.0, use_smooth_l1=False):
    """
    Train for one epoch with optional decoder auxiliary loss.
    
    Args:
        model: The model to train
        dataloader: Training dataloader (should return indices if ratings_14dim_z is provided)
        criterion: Loss function (MSE or SmoothL1)
        optimizer: Optimizer
        device: Device
        scaler: AMP scaler
        use_decoder: Whether model uses decoder
        aux_loss_weight: Weight for auxiliary 14-dim reconstruction loss
        ratings_14dim_z: Standardized 14-dim ratings for auxiliary loss (if use_decoder)
        grad_clip: Gradient clipping value
        use_smooth_l1: Use SmoothL1Loss instead of MSE
    """
    model.train()
    total_loss = 0
    total_pc_loss = 0
    total_aux_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch - may include indices if dataset returns them
        if len(batch_data) == 3:
            images, targets, idxs = batch_data
        else:
            images, targets = batch_data
            idxs = None
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
                # Handle decoder case
                if use_decoder and isinstance(outputs, tuple):
                    pc_pred, recon_14dim = outputs
                    
                    # Main loss: PC space
                    pc_loss = criterion(pc_pred, targets)
                    
                    # Auxiliary loss: reconstructed 14-dim space
                    # Use sample indices to get correct targets
                    if ratings_14dim_z is not None and idxs is not None:
                        targets_14dim = torch.from_numpy(
                            ratings_14dim_z[idxs.cpu().numpy()]
                        ).float().to(device)
                        
                        aux_loss = criterion(recon_14dim, targets_14dim)
                        loss = pc_loss + aux_loss_weight * aux_loss
                        
                        total_pc_loss += pc_loss.item()
                        total_aux_loss += aux_loss.item()
                    else:
                        loss = pc_loss
                        total_pc_loss += pc_loss.item()
                    
                    outputs_for_metrics = pc_pred
                else:
                    # No decoder, standard loss
                    loss = criterion(outputs, targets)
                    outputs_for_metrics = outputs
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            # Handle decoder case
            if use_decoder and isinstance(outputs, tuple):
                pc_pred, recon_14dim = outputs
                
                # Main loss: PC space
                pc_loss = criterion(pc_pred, targets)
                
                # Auxiliary loss: reconstructed 14-dim space
                if ratings_14dim_z is not None and idxs is not None:
                    targets_14dim = torch.from_numpy(
                        ratings_14dim_z[idxs.cpu().numpy()]
                    ).float().to(device)
                    
                    aux_loss = criterion(recon_14dim, targets_14dim)
                    loss = pc_loss + aux_loss_weight * aux_loss
                    
                    total_pc_loss += pc_loss.item()
                    total_aux_loss += aux_loss.item()
                else:
                    loss = pc_loss
                    total_pc_loss += pc_loss.item()
                
                outputs_for_metrics = pc_pred
            else:
                # No decoder, standard loss
                loss = criterion(outputs, targets)
                outputs_for_metrics = outputs
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        
        all_preds.append(outputs_for_metrics.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
        if use_decoder and aux_loss_weight > 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pc': f'{total_pc_loss/(batch_idx+1):.4f}',
                'aux': f'{total_aux_loss/(batch_idx+1):.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate correlation coefficient with NaN guards
    correlations = []
    for i in range(all_preds.shape[1]):
        corr = compute_safe_correlation(all_preds[:, i], all_targets[:, i])
        correlations.append(corr)
    avg_corr = np.mean(correlations)
    
    return avg_loss, avg_corr


def validate(model, dataloader, criterion, device, use_decoder=False):
    """
    Validate the model.
    
    Returns metrics for PC space. If using decoder, also returns reconstruction metrics.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_recon = [] if use_decoder else None
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            # Unpack batch - may or may not include indices
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                images, targets, _ = batch_data  # ignore indices if present
            else:
                images, targets = batch_data[:2] if isinstance(batch_data, (list, tuple)) else (batch_data[0], batch_data[1])
            
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            # Handle decoder case
            if use_decoder and isinstance(outputs, tuple):
                pc_pred, recon_14dim = outputs
                loss = criterion(pc_pred, targets)  # Main loss on PC space
                
                all_preds.append(pc_pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_recon.append(recon_14dim.cpu().numpy())
            else:
                loss = criterion(outputs, targets)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    if use_decoder:
        all_recon = np.vstack(all_recon)
    
    # Calculate per-dimension metrics with NaN guards
    correlations = []
    maes = []
    for i in range(all_preds.shape[1]):
        corr = compute_safe_correlation(all_preds[:, i], all_targets[:, i])
        mae = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
        correlations.append(corr)
        maes.append(mae)
    
    avg_corr = np.mean(correlations)
    avg_mae = np.mean(maes)
    
    results = {
        'loss': avg_loss,
        'corr': avg_corr,
        'mae': avg_mae,
        'correlations': correlations,
        'maes': maes
    }
    
    # Add reconstruction metrics if using decoder
    if use_decoder:
        results['reconstructions'] = all_recon
    
    return results


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
    
    # Define use_decoder early to avoid undefined variable issues
    use_decoder = bool(getattr(args, 'use_decoder', False))
    
    # Validate that ratings_file_14 is provided when using decoder
    if use_decoder and args.use_pca:
        assert hasattr(args, 'ratings_file_14') and args.ratings_file_14 is not None, \
            "ERROR: --ratings_file_14 is required when using --use_decoder with PCA ratings"
    
    # Load data
    image_paths, ratings, dimension_names, pca_dict = load_data(
        args.images_dir, 
        args.ratings_file,
        use_pca=args.use_pca,
        pca_model_path=args.pca_model if hasattr(args, 'pca_model') else None,
        use_decoder=use_decoder
    )
    
    # Split data FIRST (before normalization to prevent leakage)
    print(f"\nSplitting data with stratification...")
    print(f"  Train: {args.train_ratio*100:.0f}%")
    print(f"  Val: {args.val_ratio*100:.0f}%")
    
    # Use PC1 stratification whenever using PCA, not just with decoder
    use_pc1_strat = args.use_pca
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio > 0.001:  # If test set requested
        print(f"  Test: {test_ratio*100:.0f}%")
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings = stratified_split(
            image_paths, ratings,  # Use unnormalized ratings for splitting
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=42,
            use_pc1_stratification=use_pc1_strat
        )
    else:
        # No test set, just train/val
        train_paths, val_paths, test_paths, train_ratings, val_ratings, test_ratings = stratified_split(
            image_paths, ratings,  # Use unnormalized ratings for splitting
            train_ratio=args.train_ratio,
            val_ratio=1.0 - args.train_ratio,
            random_state=42,
            use_pc1_stratification=use_pc1_strat
        )
    
    print(f"\nFinal split:")
    print(f"  Train samples: {len(train_paths)}")
    print(f"  Val samples: {len(val_paths)}")
    if len(test_paths) > 0:
        print(f"  Test samples: {len(test_paths)}")
    
    # Normalize ratings AFTER split to prevent data leakage
    # Fit StandardScaler on TRAIN ONLY, then transform val/test
    scaler = None
    if args.normalize_targets:
        print("\nNormalizing target values (fitting on train only)...")
        scaler = StandardScaler()
        train_ratings = scaler.fit_transform(train_ratings)
        val_ratings = scaler.transform(val_ratings)
        if len(test_ratings) > 0:
            test_ratings = scaler.transform(test_ratings)
        # Save scaler
        joblib.dump(scaler, output_dir / 'target_scaler.joblib')
        print(f"  ✓ Saved scaler to {output_dir / 'target_scaler.joblib'}")
    
    # Load 14-dim targets for auxiliary loss if using decoder
    Yz_train, Yz_val, Yz_test = None, None, None
    if use_decoder:
        print("\nLoading 14-dim targets for auxiliary reconstruction loss...")
        
        # Use ratings_file_14 if provided, otherwise try to use original ratings_file
        ratings_14_file = args.ratings_file_14 if hasattr(args, 'ratings_file_14') and args.ratings_file_14 else args.ratings_file
        
        # Load the 14-dim CSV
        with open(ratings_14_file, 'r') as f:
            lines14 = f.readlines()
        header14 = lines14[0].strip().replace('""', '"').replace('"', '')
        lines14[0] = header14 + '\n'
        import io
        fixed_csv14 = io.StringIO(''.join(lines14))
        df14 = pd.read_csv(fixed_csv14)
        
        # Set index to image name for lookups
        df14.index = df14['image'].apply(lambda p: Path(p).name)
        
        # Get dimension names from PCA model (order used in PCA fit)
        dim_order = pca_dict['dimension_names']
        print(f"  Using dimension order from PCA: {dim_order}")
        
        # Extract 14-dim ratings in same order as original data (image_paths)
        map_idx = [Path(p).name for p in image_paths]
        Y_14 = df14.loc[map_idx, [f"{d}_mean" for d in dim_order]].values.astype(np.float32)
        
        # Transform with the saved scaler_y from PCA (fitted on train only during PCA computation)
        scaler_y = pca_dict['scaler_y']
        Yz_all = scaler_y.transform(Y_14)
        
        # Create index mapping from image paths to positions
        path_to_idx = {Path(p).name: i for i, p in enumerate(image_paths)}
        
        # Extract targets for each split using the same indices
        train_idx = np.array([path_to_idx[Path(p).name] for p in train_paths])
        val_idx = np.array([path_to_idx[Path(p).name] for p in val_paths])
        
        Yz_train = Yz_all[train_idx]
        Yz_val = Yz_all[val_idx]
        
        if len(test_paths) > 0:
            test_idx = np.array([path_to_idx[Path(p).name] for p in test_paths])
            Yz_test = Yz_all[test_idx]
        
        print(f"  ✓ Loaded 14-dim targets: train={Yz_train.shape}, val={Yz_val.shape}")
        if Yz_test is not None:
            print(f"  ✓ Test 14-dim targets: {Yz_test.shape}")
    
    # Print split statistics to verify balance
    print_split_statistics(train_ratings, val_ratings, test_ratings, dimension_names)
    
    # Get weight-specific transforms for best accuracy
    train_transforms = get_transforms(args.backbone, args.pretrained, args.image_size, train=True)
    val_transforms = get_transforms(args.backbone, args.pretrained, args.image_size, train=False)
    
    # Create datasets with return_index=True for proper aux loss indexing
    train_dataset = PsychologicalRatingDataset(
        train_paths, train_ratings,
        transform=train_transforms,
        dimension_names=dimension_names,
        return_index=True  # Need indices for aux loss
    )
    
    val_dataset = PsychologicalRatingDataset(
        val_paths, val_ratings,
        transform=val_transforms,
        dimension_names=dimension_names,
        return_index=False  # Don't need indices in validation (saves memory)
    )
    
    # Create dataloaders with determinism
    g = torch.Generator()
    g.manual_seed(42)
    
    def worker_init_fn(worker_id):
        """Initialize worker with unique but reproducible seed."""
        import random
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if args.num_workers > 0 else False  # Speed up dataloading
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create test dataloader if test set exists
    if len(test_paths) > 0:
        test_dataset = PsychologicalRatingDataset(
            test_paths, test_ratings,
            transform=val_transforms,
            dimension_names=dimension_names,
            return_index=False  # Don't need indices in test
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        print(f"Test loader created with {len(test_dataset)} samples")
    else:
        test_loader = None
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    
    # Critical: Use n_components from PCA model, not CSV column count
    if args.use_pca:
        # Check PCA consistency
        if pca_dict is not None:
            n_pc_model = int(pca_dict['n_components'])
            n_pc_csv = len(dimension_names)
            
            # Verify CSV PC columns match PCA model
            if n_pc_csv != n_pc_model:
                print(f"⚠ WARNING: CSV has {n_pc_csv} PCs, PCA model expects {n_pc_model}")
                if not use_decoder:
                    print(f"  Proceeding without decoder (CSV/model mismatch)")
            
            # Additional check: ensure PC columns exist
            if all(d.startswith('PC') for d in dimension_names):
                n_pc_cols = len([c for c in dimension_names if c.startswith('PC')])
                assert n_pc_cols == pca_dict['n_components'], \
                    f"PC columns in CSV ({n_pc_cols}) do not match PCA model n_components ({pca_dict['n_components']})"
        
        if use_decoder:
            assert pca_dict is not None, "pca_model is required when use_decoder=True"
            n_pc = int(pca_dict['n_components'])
            pca_components = np.asarray(pca_dict['components'], dtype=np.float32)
            
            # Verify consistency
            assert pca_components.shape[0] == n_pc, f"PCA components shape mismatch: {pca_components.shape[0]} != {n_pc}"
            assert pca_components.shape[1] == 14, f"PCA components should map to 14 dims, got {pca_components.shape[1]}"
            print(f"  ✓ Using {n_pc} PCs from PCA model")
            print(f"  ✓ Decoder will reconstruct 14 dimensions")
        else:
            # Without decoder, use column count from CSV (should match n_components)
            n_pc = len(dimension_names)
            pca_components = None
            print(f"  ✓ Predicting {n_pc} PC scores (no decoder)")
    else:
        n_pc = None
        pca_components = None
    
    model = MultiOutputRegressionModel(
        num_outputs=len(dimension_names),
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
        use_pca=args.use_pca,
        n_pc=n_pc,
        pca_components=pca_components,
        use_decoder=use_decoder
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    loss_fn = getattr(args, 'loss_fn', 'mse')
    if loss_fn == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
        print("Using SmoothL1Loss for robustness")
    else:
        criterion = nn.MSELoss()
        print("Using MSELoss")
    
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
            model, train_loader, criterion, optimizer, device, scaler_amp,
            use_decoder=use_decoder,
            aux_loss_weight=getattr(args, 'aux_loss_weight', 0.2),
            ratings_14dim_z=Yz_train if use_decoder else None,
            grad_clip=getattr(args, 'grad_clip', 1.0),
            use_smooth_l1=(loss_fn == 'smooth_l1')
        )
        
        # Validate
        val_results = validate(
            model, val_loader, criterion, device, use_decoder=use_decoder
        )
        val_loss = val_results['loss']
        val_corr = val_results['corr']
        val_mae = val_results['mae']
        val_correlations = val_results['correlations']
        val_maes = val_results['maes']
        
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
        
        # Save best model based on val_loss (not correlation)
        # This ensures we save the model that best fits the PC space
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_corr = val_corr  # Track corr for reporting
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr,
                'val_mae': val_mae,
                'dimension_names': dimension_names,
                'use_pca': args.use_pca,
                'use_decoder': use_decoder,
                'n_pc': n_pc,
                'pca_model_path': getattr(args, 'pca_model', None),
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            
            print(f"✓ Saved best model (loss: {val_loss:.4f}, corr: {val_corr:.4f})")
            
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
    checkpoint = torch.load(output_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_results = validate(
        model, val_loader, criterion, device, use_decoder=use_decoder
    )
    val_loss = val_results['loss']
    val_corr = val_results['corr']
    val_mae = val_results['mae']
    val_correlations = val_results['correlations']
    val_maes = val_results['maes']
    
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
        
        test_results = validate(
            model, test_loader, criterion, device, use_decoder=use_decoder
        )
        test_loss = test_results['loss']
        test_corr = test_results['corr']
        test_mae = test_results['mae']
        test_correlations = test_results['correlations']
        test_maes = test_results['maes']
        
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
                        help='CSV file with ratings (original or PCA-transformed)')
    parser.add_argument('--ratings_file_14', type=str, default=None,
                        help='CSV file with original 14-dim ratings (required if use_decoder=True for aux loss)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--use_pca', action='store_true', default=False,
                        help='Use PCA-transformed ratings (expects PC1, PC2, etc. columns)')
    parser.add_argument('--pca_model', type=str, default=None,
                        help='Path to PCA model joblib file (required if use_decoder=True)')
    parser.add_argument('--use_decoder', action='store_true', default=False,
                        help='Use decoder to reconstruct 14 dimensions from PC scores')
    parser.add_argument('--aux_loss_weight', type=float, default=0.2,
                        help='Weight for auxiliary 14-dim reconstruction loss (default: 0.2)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0, set to 0 to disable)')
    parser.add_argument('--loss_fn', type=str, default='mse',
                        choices=['mse', 'smooth_l1'],
                        help='Loss function: mse or smooth_l1 (more robust)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 
                                'efficientnet_b3', 'vit_b_16'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained weights')
    parser.set_defaults(pretrained=True)
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
