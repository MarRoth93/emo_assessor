#!/usr/bin/env python3
"""
Compute PCA on 14-dimensional rating matrix with proper train/val/test split handling.

This script:
1. Loads the ratings data and image lists for train/val/test splits
2. Fits StandardScaler on TRAIN data only
3. Transforms all splits with the fitted scaler
4. Fits PCA on TRAIN data only
5. Transforms all splits with the fitted PCA
6. Saves PCA components, scaler, and transformed ratings
7. Creates comprehensive visualizations
8. Saves interpretable loadings sorted by absolute value

CRITICAL: Prevents data leakage by fitting only on training data.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_ratings_data(ratings_file):
    """Load ratings data with proper CSV header handling."""
    print("Loading ratings data...")
    
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
    
    # Extract only the mean columns - these are the prediction targets
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    dimension_names = [col.replace('_mean', '') for col in mean_cols]
    
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  ✓ Found {len(dimension_names)} dimensions: {dimension_names}")
    
    # Extract ratings matrix
    ratings_matrix = df[mean_cols].values
    image_names = df['image'].values
    
    return ratings_matrix, dimension_names, image_names, df


def create_stratified_splits(ratings_matrix, image_names, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    Create stratified train/val/test splits based on rating distributions.
    
    Args:
        ratings_matrix: (n_samples, n_dimensions) array
        image_names: Array of image filenames
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_state: Random seed
    
    Returns:
        train/val/test indices
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Calculate mean rating across all dimensions for stratification
    mean_ratings = ratings_matrix.mean(axis=1)
    
    # Create stratification bins (quintiles)
    n_bins = 5
    bins = np.percentile(mean_ratings, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 0.001  # Ensure last value is included
    strata = np.digitize(mean_ratings, bins[1:-1])
    
    test_ratio = 1.0 - train_ratio - val_ratio
    indices = np.arange(len(image_names))
    
    # Split off test set
    if test_ratio > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        train_val_idx, test_idx = next(sss.split(indices, strata))
        
        # Split train and val from remaining
        train_val_strata = strata[train_val_idx]
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=random_state)
        train_idx_local, val_idx_local = next(sss.split(train_val_idx, train_val_strata))
        
        train_idx = train_val_idx[train_idx_local]
        val_idx = train_val_idx[val_idx_local]
    else:
        # Only train/val split
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=random_state)
        train_idx, val_idx = next(sss.split(indices, strata))
        test_idx = np.array([], dtype=int)
    
    return train_idx, val_idx, test_idx


def fit_pca_on_train(train_ratings, n_components=4, random_state=42):
    """
    Fit StandardScaler and PCA on training data ONLY.
    
    Args:
        train_ratings: (n_train, n_dimensions) array - TRAIN SET ONLY
        n_components: Number of principal components
        random_state: Random seed
    
    Returns:
        scaler_y: Fitted StandardScaler
        pca: Fitted PCA
        train_Yz: Standardized train ratings
        train_PC: PC scores for train set
    """
    print(f"\nFitting StandardScaler on {len(train_ratings)} training samples...")
    scaler_y = StandardScaler()
    train_Yz = scaler_y.fit_transform(train_ratings)
    
    print(f"  ✓ Train mean: {train_Yz.mean(axis=0)[:3]}... (should be ~0)")
    print(f"  ✓ Train std: {train_Yz.std(axis=0)[:3]}... (should be ~1)")
    
    print(f"\nFitting PCA with {n_components} components on training data...")
    pca = PCA(n_components=n_components, random_state=random_state)
    train_PC = pca.fit_transform(train_Yz)
    
    variance_explained = np.sum(pca.explained_variance_ratio_)
    print(f"  ✓ Variance explained: {variance_explained:.4f}")
    print(f"  ✓ Per-component variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"     PC{i+1}: {var:.4f}")
    
    return scaler_y, pca, train_Yz, train_PC


def transform_splits(scaler_y, pca, train_ratings, val_ratings, test_ratings):
    """
    Transform validation and test sets using fitted scaler and PCA.
    
    Args:
        scaler_y: Fitted StandardScaler
        pca: Fitted PCA
        train_ratings, val_ratings, test_ratings: Rating matrices
    
    Returns:
        Standardized ratings and PC scores for all splits
    """
    print("\nTransforming all splits with fitted scaler and PCA...")
    
    # Train (already done, but included for completeness)
    train_Yz = scaler_y.transform(train_ratings)
    train_PC = pca.transform(train_Yz)
    
    # Validation
    val_Yz = scaler_y.transform(val_ratings)
    val_PC = pca.transform(val_Yz)
    print(f"  ✓ Transformed {len(val_ratings)} validation samples")
    
    # Test (if exists)
    if len(test_ratings) > 0:
        test_Yz = scaler_y.transform(test_ratings)
        test_PC = pca.transform(test_Yz)
        print(f"  ✓ Transformed {len(test_ratings)} test samples")
    else:
        test_Yz = np.array([])
        test_PC = np.array([])
    
    return (train_Yz, train_PC), (val_Yz, val_PC), (test_Yz, test_PC)


def save_pca_artifacts(output_dir, scaler_y, pca, dimension_names):
    """
    Save the fitted scaler and PCA model using joblib.
    
    Args:
        output_dir: Output directory path
        scaler_y: Fitted StandardScaler
        pca: Fitted PCA
        dimension_names: List of original dimension names
    """
    print("\nSaving PCA artifacts...")
    
    # Save as dictionary with all metadata
    pca_dict = {
        'scaler_y': scaler_y,
        'pca': pca,
        'dimension_names': dimension_names,
        'n_components': pca.n_components,
        'variance_explained': np.sum(pca.explained_variance_ratio_),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_
    }
    
    model_path = os.path.join(output_dir, 'pca_targets.joblib')
    joblib.dump(pca_dict, model_path)
    print(f"  ✓ Saved PCA model to {model_path}")
    
    # Also save loadings as CSV sorted by absolute value per PC
    print("\nSaving PCA loadings sorted by absolute value...")
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=dimension_names,
        index=[f'PC{i+1}' for i in range(pca.n_components)]
    )
    
    # Save raw loadings
    loadings_path = os.path.join(output_dir, 'pca_loadings.csv')
    loadings_df.to_csv(loadings_path)
    print(f"  ✓ Saved raw loadings to {loadings_path}")
    
    # Save sorted loadings for each PC
    for pc_idx in range(pca.n_components):
        pc_name = f'PC{pc_idx + 1}'
        sorted_loadings = loadings_df.T.sort_values(by=pc_name, key=np.abs, ascending=False)
        
        sorted_path = os.path.join(output_dir, f'pca_loadings_{pc_name}_sorted.csv')
        sorted_loadings.to_csv(sorted_path)
        print(f"  ✓ Saved {pc_name} sorted loadings to {sorted_path}")
        
        # Print top contributors
        print(f"\n    Top contributors to {pc_name} ({pca.explained_variance_ratio_[pc_idx]:.1%} variance):")
        for dim in sorted_loadings.index[:5]:
            loading = sorted_loadings.loc[dim, pc_name]
            print(f"      {dim:20s}: {loading:+.3f}")


def save_transformed_ratings(output_dir, image_names, pc_scores, split_name):
    """
    Save PC scores as CSV.
    
    Args:
        output_dir: Output directory
        image_names: Array of image filenames
        pc_scores: (n_samples, n_components) array
        split_name: 'train', 'val', or 'test'
    """
    n_components = pc_scores.shape[1]
    
    df = pd.DataFrame(
        pc_scores,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    df.insert(0, 'image', image_names)
    
    output_path = os.path.join(output_dir, f'ratings_pca_{n_components}comp_{split_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {split_name} PC scores to {output_path}")


def create_visualizations(pca, train_PC, dimension_names, output_dir):
    """Create comprehensive PCA visualizations."""
    print("\nCreating visualizations...")
    
    n_components = pca.n_components
    n_dims = len(dimension_names)
    
    # 1. Explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(f'Explained Variance by Component\n(Total: {np.sum(pca.explained_variance_ratio_):.1%})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, n_components + 1), cumvar, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Component loadings heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, n_components * 1.5)))
    
    sns.heatmap(pca.components_,
                xticklabels=dimension_names,
                yticklabels=[f'PC{i+1}' for i in range(n_components)],
                cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Loading'},
                ax=ax)
    
    ax.set_title(f'PCA Component Loadings ({n_components} components)\n'
                 f'Variance Explained: {np.sum(pca.explained_variance_ratio_):.1%}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Original Dimensions', fontsize=12)
    ax.set_ylabel('Principal Components', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_loadings_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top loadings per component
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    for pc_idx in range(n_components):
        ax = axes[pc_idx]
        loadings = pca.components_[pc_idx, :]
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        sorted_loadings = loadings[sorted_indices]
        sorted_dims = [dimension_names[i] for i in sorted_indices]
        
        colors = ['red' if x < 0 else 'blue' for x in sorted_loadings]
        ax.barh(range(len(sorted_dims)), sorted_loadings, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_dims)))
        ax.set_yticklabels(sorted_dims)
        ax.set_xlabel('Loading', fontsize=10)
        ax.set_title(f'PC{pc_idx + 1}\n({pca.explained_variance_ratio_[pc_idx]:.1%} var)',
                     fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_top_loadings.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 2D projection (if at least 2 components)
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(train_PC[:, 0], train_PC[:, 1],
                           alpha=0.5, s=30, c=range(len(train_PC)),
                           cmap='viridis')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.set_title('Training Data in PC Space (2D Projection)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_2d_projection.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  ✓ All visualizations saved")


def print_summary(scaler_y, pca, dimension_names, train_idx, val_idx, test_idx):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("PCA COMPUTATION SUMMARY")
    print("=" * 80)
    
    print(f"\nData splits:")
    print(f"  Training:   {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    if len(test_idx) > 0:
        print(f"  Test:       {len(test_idx)} samples")
    
    print(f"\nPCA configuration:")
    print(f"  Original dimensions: {len(dimension_names)}")
    print(f"  PC components: {pca.n_components}")
    print(f"  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    print(f"\nPer-component variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumvar = np.sum(pca.explained_variance_ratio_[:i+1])
        print(f"  PC{i+1}: {var:.1%} (cumulative: {cumvar:.1%})")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compute PCA on ratings with proper train/val/test split handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This script fits StandardScaler and PCA ONLY on training data,
then transforms validation and test sets. This prevents data leakage.

Examples:
  # Basic usage with default settings
  python scripts/compute_pca.py
  
  # Specify number of components
  python scripts/compute_pca.py --n_components 4
  
  # Custom split ratios
  python scripts/compute_pca.py --train_ratio 0.7 --val_ratio 0.15
  
  # Custom paths
  python scripts/compute_pca.py --ratings ratings/my_ratings.csv --output results/pca/
        """
    )
    
    parser.add_argument('--ratings', type=str,
                       default='./ratings/per_image_Slider_mean_sd_from_wide.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--output_dir', type=str, default='./results/pca_analysis',
                       help='Output directory for PCA results')
    parser.add_argument('--n_components', type=int, default=4,
                       help='Number of principal components')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be < 1.0 to leave room for test set")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PCA ANALYSIS WITH PROPER SPLIT HANDLING")
    print("=" * 80)
    print(f"Ratings file: {args.ratings}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of components: {args.n_components}")
    print(f"Split ratios: train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, "
          f"test={1-args.train_ratio-args.val_ratio:.0%}")
    
    # Load data
    ratings_matrix, dimension_names, image_names, df = load_ratings_data(args.ratings)
    
    print(f"\nRatings matrix shape: {ratings_matrix.shape}")
    print(f"  {ratings_matrix.shape[0]} images")
    print(f"  {ratings_matrix.shape[1]} dimensions")
    
    # Create stratified splits
    print(f"\nCreating stratified splits...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        ratings_matrix, image_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )
    
    train_ratings = ratings_matrix[train_idx]
    val_ratings = ratings_matrix[val_idx]
    test_ratings = ratings_matrix[test_idx] if len(test_idx) > 0 else np.array([])
    
    train_images = image_names[train_idx]
    val_images = image_names[val_idx]
    test_images = image_names[test_idx] if len(test_idx) > 0 else np.array([])
    
    print(f"  ✓ Train: {len(train_idx)} samples")
    print(f"  ✓ Val: {len(val_idx)} samples")
    if len(test_idx) > 0:
        print(f"  ✓ Test: {len(test_idx)} samples")
    
    # Fit PCA on train data only
    scaler_y, pca, train_Yz, train_PC = fit_pca_on_train(
        train_ratings,
        n_components=args.n_components,
        random_state=args.random_state
    )
    
    # Transform all splits
    (train_Yz, train_PC), (val_Yz, val_PC), (test_Yz, test_PC) = transform_splits(
        scaler_y, pca, train_ratings, val_ratings, test_ratings
    )
    
    # Save PCA artifacts
    save_pca_artifacts(args.output_dir, scaler_y, pca, dimension_names)
    
    # Save transformed ratings for each split
    print("\nSaving PC scores for all splits...")
    save_transformed_ratings(args.output_dir, train_images, train_PC, 'train')
    save_transformed_ratings(args.output_dir, val_images, val_PC, 'val')
    if len(test_PC) > 0:
        save_transformed_ratings(args.output_dir, test_images, test_PC, 'test')
    
    # Create visualizations
    create_visualizations(pca, train_PC, dimension_names, args.output_dir)
    
    # Print summary
    print_summary(scaler_y, pca, dimension_names, train_idx, val_idx, test_idx)
    
    # Save split indices for reproducibility
    split_indices = {
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist() if len(test_idx) > 0 else []
    }
    
    import json
    with open(os.path.join(args.output_dir, 'split_indices.json'), 'w') as f:
        json.dump(split_indices, f, indent=2)
    print(f"\n✓ Saved split indices to split_indices.json")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n1. PCA model saved: {args.output_dir}/pca_targets.joblib")
    print(f"2. PC scores saved: ratings_pca_{args.n_components}comp_{{train,val,test}}.csv")
    print(f"3. Loadings saved and sorted by absolute value per PC")
    print("\n4. Train model with PC scores as targets:")
    print(f"   python scripts/train_assessor.py \\")
    print(f"       --ratings_file {args.output_dir}/ratings_pca_{args.n_components}comp_train.csv \\")
    print(f"       --use_pca --pca_model {args.output_dir}/pca_targets.joblib")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
