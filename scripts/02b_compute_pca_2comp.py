#!/usr/bin/env python3
"""
Compute 2-component PCA for simplified affect prediction.
Outputs interpretable dimensions (e.g., Valence, Arousal).
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ratings(ratings_file):
    """Load and parse ratings CSV with quirky header format."""
    import io
    with open(ratings_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().replace('""', '"').replace('"', '')
    lines[0] = header + '\n'
    df = pd.read_csv(io.StringIO(''.join(lines)))
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute 2-component PCA')
    parser.add_argument('--ratings_file', type=str, 
                       default='ratings/per_image_Slider_mean_sd_from_wide.csv')
    parser.add_argument('--output_dir', type=str, default='results/pca_2comp')
    parser.add_argument('--split_file', type=str, 
                       default='results/pca_analysis/split_indices.json')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading ratings...")
    df = load_ratings(args.ratings_file)
    
    # Get mean columns (the 14 dimensions)
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    feature_names = [col.replace('_mean', '') for col in mean_cols]
    print(f"Found {len(mean_cols)} dimensions: {feature_names}")
    
    # Get image column
    if 'filename' in df.columns:
        img_col = 'filename'
    elif 'image' in df.columns:
        img_col = 'image'
    else:
        img_col = df.columns[0]
    
    # Load split indices
    with open(args.split_file, 'r') as f:
        splits = json.load(f)
    
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    
    # Extract features
    X = df[mean_cols].values
    
    # Fit scaler and PCA on TRAINING data only
    print("\nFitting StandardScaler and PCA on training data...")
    X_train = X[train_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    
    # Print component interpretation
    print("\n" + "=" * 60)
    print("PCA COMPONENT INTERPRETATION")
    print("=" * 60)
    
    component_names = []
    for i in range(2):
        print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]:.1%} variance):")
        loadings = pca.components_[i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        
        print("  Top positive loadings:")
        pos_idx = [j for j in sorted_idx if loadings[j] > 0][:3]
        for j in pos_idx:
            print(f"    {feature_names[j]:20s}: {loadings[j]:+.3f}")
        
        print("  Top negative loadings:")
        neg_idx = [j for j in sorted_idx if loadings[j] < 0][:3]
        for j in neg_idx:
            print(f"    {feature_names[j]:20s}: {loadings[j]:+.3f}")
        
        # Suggest interpretation
        if i == 0:
            # PC1 is typically valence
            pos_terms = [feature_names[j] for j in pos_idx]
            if any(t in ['Approach', 'Valence', 'Pleasure'] for t in pos_terms):
                name = "Valence_Pleasantness"
            else:
                name = "PC1_Affective"
        else:
            # PC2 is typically arousal
            pos_terms = [feature_names[j] for j in pos_idx]
            if any(t in ['Arousal', 'Dominance', 'Excitement'] for t in pos_terms):
                name = "Arousal_Activation"
            else:
                name = "PC2_Activation"
        
        component_names.append(name)
        print(f"  → Suggested label: {name}")
    
    total_var = pca.explained_variance_ratio_.sum()
    print(f"\nTotal variance explained by 2 components: {total_var:.1%}")
    
    # Transform all data
    print("\nTransforming all splits...")
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Create output DataFrames for each split
    for split_name, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = pd.DataFrame({
            'image': df[img_col].iloc[indices].values,
            component_names[0]: X_pca[indices, 0],
            component_names[1]: X_pca[indices, 1],
        })
        
        output_path = os.path.join(args.output_dir, f'ratings_2comp_{split_name}.csv')
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split_name}: {len(split_df)} samples → {output_path}")
    
    # Save PCA model
    pca_data = {
        'pca': pca,
        'scaler_y': scaler,
        'n_components': 2,
        'feature_names': feature_names,
        'component_names': component_names,
        'variance_explained': total_var,
        'variance_per_component': pca.explained_variance_ratio_.tolist(),
        'loadings': pca.components_.tolist(),
    }
    
    model_path = os.path.join(args.output_dir, 'pca_2comp.joblib')
    joblib.dump(pca_data, model_path)
    print(f"\nSaved PCA model to {model_path}")
    
    # Create loading visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        loadings = pca.components_[i]
        sorted_idx = np.argsort(loadings)
        
        colors = ['#d73027' if l < 0 else '#1a9850' for l in loadings[sorted_idx]]
        
        ax.barh(np.array(feature_names)[sorted_idx], loadings[sorted_idx], color=colors)
        ax.set_xlabel('Loading')
        ax.set_title(f'{name}\n({pca.explained_variance_ratio_[i]:.1%} variance)')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'component_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loading plot to {args.output_dir}/component_loadings.png")
    
    # Save interpretation summary
    summary_path = os.path.join(args.output_dir, 'component_interpretation.txt')
    with open(summary_path, 'w') as f:
        f.write("2-COMPONENT PCA INTERPRETATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total variance explained: {total_var:.1%}\n\n")
        
        for i, name in enumerate(component_names):
            f.write(f"\n{name} (PC{i+1}): {pca.explained_variance_ratio_[i]:.1%} variance\n")
            f.write("-" * 40 + "\n")
            loadings = pca.components_[i]
            sorted_idx = np.argsort(np.abs(loadings))[::-1]
            for j in sorted_idx:
                f.write(f"  {feature_names[j]:20s}: {loadings[j]:+.3f}\n")
    
    print(f"Saved interpretation to {summary_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()