#!/usr/bin/env python
"""
Train ensemble model (ViT + ResNet) with configurable augmentation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.models.ensemble_model import EnsembleRegressor, EnsembleWithDecoder
from scripts.datasets.augmentation import get_augmentation, get_validation_transform, MixUpCutMix
from scripts.datasets.image_dataset import ImageRegressionDataset


def plot_training_curves(history, output_dir):
    """Generate and save training curve plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_epoch}')
    
    # R² curve
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_r2'], 'g-', label='Val R²', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('Validation R²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    best_r2_epoch = np.argmax(history['val_r2']) + 1
    best_r2 = max(history['val_r2'])
    ax2.axvline(x=best_r2_epoch, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=best_r2, color='r', linestyle=':', alpha=0.5)
    ax2.annotate(f'Best: {best_r2:.4f}', xy=(best_r2_epoch, best_r2), 
                xytext=(best_r2_epoch + 2, best_r2 + 0.02))
    
    # Correlation curve
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_corr'], 'm-', label='Val Correlation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Validation Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax4 = axes[1, 1]
    gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
    ax4.plot(epochs, gap, 'orange', label='Val - Train Loss', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Gap')
    ax4.set_title('Overfitting Gap (Val - Train)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(epochs, 0, gap, alpha=0.3, color='orange')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training curves saved to {output_dir}/training_curves.png")


def parse_args():
    parser = argparse.ArgumentParser(description='Train ensemble model')
    
    # Data
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='images')
    
    # PCA / Decoder
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--use_decoder', action='store_true')
    parser.add_argument('--pca_model', type=str, default='results/pca_analysis/pca_targets.joblib')
    parser.add_argument('--pca_components', type=int, default=4)
    parser.add_argument('--freeze_decoder', action='store_true', default=True)
    
    # Model
    parser.add_argument('--freeze_backbones', action='store_true', default=True)
    parser.add_argument('--unfreeze_layers', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    
    # Augmentation
    parser.add_argument('--augmentation', type=str, default='moderate',
                       choices=['conservative', 'moderate', 'strong'])
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_fn', type=str, default='smooth_l1',
                       choices=['mse', 'smooth_l1', 'huber'])
    parser.add_argument('--aux_loss_weight', type=float, default=0.2,
                       help='Weight for reconstruction loss when using decoder')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--output_dir', type=str, default='results/ensemble')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def get_loss_fn(name: str):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif name == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


def train_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp, 
                mixup_fn=None, use_decoder=False, aux_loss_weight=0.0, original_targets_key=None):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Training', leave=False):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Get original targets for aux loss if using decoder
        if use_decoder and original_targets_key and original_targets_key in batch:
            original_targets = batch[original_targets_key].to(device)
        else:
            original_targets = None
        
        # Apply MixUp if enabled
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_decoder:
                outputs = model(images)
                pc_loss = loss_fn(outputs['pc_scores'], targets)
                
                if original_targets is not None:
                    recon_loss = loss_fn(outputs['reconstructed'], original_targets)
                    loss = pc_loss + aux_loss_weight * recon_loss
                else:
                    loss = pc_loss
            else:
                outputs = model(images)
                loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_decoder=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(loader, desc='Validation', leave=False):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        if use_decoder:
            outputs = model(images)
            preds = outputs['pc_scores']
        else:
            preds = model(images)
        
        loss = loss_fn(preds, targets)
        total_loss += loss.item()
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    
    # R² per dimension
    ss_res = np.sum((all_targets - all_preds) ** 2, axis=0)
    ss_tot = np.sum((all_targets - all_targets.mean(axis=0)) ** 2, axis=0)
    r2_per_dim = 1 - ss_res / (ss_tot + 1e-8)
    r2_mean = np.mean(r2_per_dim)
    
    # Correlation per dimension
    correlations = []
    for i in range(all_preds.shape[1]):
        corr = np.corrcoef(all_preds[:, i], all_targets[:, i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    corr_mean = np.mean(correlations)
    
    return {
        'loss': total_loss / len(loader),
        'mse': mse,
        'r2_mean': r2_mean,
        'r2_per_dim': r2_per_dim.tolist(),
        'corr_mean': corr_mean,
        'corr_per_dim': correlations
    }


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load PCA if needed
    pca_components = None
    pca_mean = None
    if args.use_decoder:
        pca_data = joblib.load(args.pca_model)
        pca = pca_data['pca']
        scaler = pca_data['scaler_y']  # Note: key is 'scaler_y' not 'scaler'
        pca_components = torch.tensor(pca.components_, dtype=torch.float32)
        pca_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    
    # Determine output dimensions
    if args.use_pca:
        n_outputs = args.pca_components
    else:
        n_outputs = 14  # All original dimensions
    
    # Create model
    if args.use_decoder:
        model = EnsembleWithDecoder(
            n_pca_components=args.pca_components,
            n_original_dims=14,
            pca_components=pca_components,
            pca_mean=pca_mean,
            freeze_decoder=args.freeze_decoder,
            freeze_backbones=args.freeze_backbones,
            unfreeze_layers=args.unfreeze_layers,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim
        )
    else:
        model = EnsembleRegressor(
            n_outputs=n_outputs,
            freeze_backbones=args.freeze_backbones,
            unfreeze_layers=args.unfreeze_layers,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Datasets
    train_transform = get_augmentation(mode=args.augmentation, img_size=224)
    val_transform = get_validation_transform(img_size=224)
    
    train_dataset = ImageRegressionDataset(
        csv_file=args.train_csv,
        image_dir=args.image_dir,
        transform=train_transform,
        use_pca=args.use_pca,
        n_components=args.pca_components
    )
    
    val_dataset = ImageRegressionDataset(
        csv_file=args.val_csv,
        image_dir=args.image_dir,
        transform=val_transform,
        use_pca=args.use_pca,
        n_components=args.pca_components
    )
    
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
    
    # MixUp
    mixup_fn = MixUpCutMix(mixup_alpha=args.mixup_alpha) if args.use_mixup else None
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    loss_fn = get_loss_fn(args.loss_fn)
    
    # Training loop
    best_r2 = -float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_corr': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, args.use_amp,
            mixup_fn=mixup_fn, use_decoder=args.use_decoder, 
            aux_loss_weight=args.aux_loss_weight
        )
        
        val_metrics = validate(model, val_loader, loss_fn, device, args.use_decoder)
        
        scheduler.step()
        
        # Log
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | MSE: {val_metrics['mse']:.4f}")
        print(f"  Val R²: {val_metrics['r2_mean']:.4f} | Corr: {val_metrics['corr_mean']:.4f}")
        
        # Convert to Python floats for JSON serialization
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_r2'].append(float(val_metrics['r2_mean']))
        history['val_corr'].append(float(val_metrics['corr_mean']))
        
        # Save best model
        if val_metrics['r2_mean'] > best_r2:
            best_r2 = float(val_metrics['r2_mean'])
            # Convert val_metrics to JSON-serializable format
            val_metrics_save = {}
            for k, v in val_metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    val_metrics_save[k] = float(v)
                elif isinstance(v, np.ndarray):
                    val_metrics_save[k] = v.tolist()
                elif isinstance(v, list):
                    val_metrics_save[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    val_metrics_save[k] = v
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_r2': best_r2,
                'val_metrics': val_metrics_save
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ New best model saved (R² = {best_r2:.4f})")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate training curves plot
    plot_training_curves(history, args.output_dir)
    
    print(f"\nTraining complete. Best R²: {best_r2:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()