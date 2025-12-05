#!/usr/bin/env python3
"""
Investigate low-performing dimensions.
Analyzes data quality and distribution for problematic dimensions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_ratings_data(ratings_file):
    """Load ratings data with proper CSV header handling."""
    # Read the file to fix the header
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
    # The _sd columns contain standard deviations of ratings, not dimensions to predict
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    dimension_names = [col.replace('_mean', '') for col in mean_cols]
    
    print(f"Found {len(dimension_names)} dimensions to predict: {dimension_names}")
    
    # Also get sd columns for analysis purposes
    sd_cols = [col for col in df.columns if col.endswith('_sd')]
    
    return df, mean_cols, sd_cols, dimension_names


def analyze_distribution(df, dimension_cols, output_dir):
    """Analyze rating distributions for each dimension."""
    n_dims = len(dimension_cols)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    # Properly flatten axes array regardless of shape
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    stats_data = []
    
    for i, dim in enumerate(dimension_cols):
        ax = axes[i]
        values = df[dim].values
        
        # Histogram with KDE
        ax.hist(values, bins=30, alpha=0.6, color='blue', edgecolor='black', density=True)
        
        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        # Statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        
        # Skewness and kurtosis
        skewness = stats.skew(values)
        kurt = stats.kurtosis(values)
        
        # Normality test
        _, p_value = stats.shapiro(values) if len(values) < 5000 else stats.normaltest(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel('Rating Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{dim}\nStd={std_val:.2f}, Range={range_val:.2f}, Skew={skewness:.2f}', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Store statistics
        stats_data.append({
            'Dimension': dim,
            'Mean': mean_val,
            'Median': median_val,
            'Std': std_val,
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Normality_p': p_value,
            'CV': std_val / mean_val if mean_val != 0 else np.inf,  # Coefficient of variation
        })
    
    # Hide empty subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plots")
    
    return pd.DataFrame(stats_data)


def analyze_correlations(df, dimension_cols, output_dir):
    """Analyze correlations between dimensions."""
    # Compute correlation matrix
    corr_matrix = df[dimension_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Between Dimensions', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix")
    
    # Find highly correlated pairs
    high_corr = []
    for i in range(len(dimension_cols)):
        for j in range(i+1, len(dimension_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Strong correlation threshold
                high_corr.append({
                    'Dimension_1': dimension_cols[i],
                    'Dimension_2': dimension_cols[j],
                    'Correlation': corr_val
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', ascending=False, key=abs)
        return high_corr_df
    else:
        return None


def analyze_data_quality(df, dimension_cols, output_dir):
    """Check for data quality issues."""
    issues = []
    
    for dim in dimension_cols:
        values = df[dim].values
        
        # Check for missing values
        n_missing = df[dim].isna().sum()
        if n_missing > 0:
            issues.append({
                'Dimension': dim,
                'Issue': 'Missing values',
                'Details': f'{n_missing} missing values ({n_missing/len(df)*100:.1f}%)'
            })
        
        # Check for constant/near-constant values
        if np.std(values) < 0.1:
            issues.append({
                'Dimension': dim,
                'Issue': 'Low variance',
                'Details': f'Std={np.std(values):.3f}, may be too uniform'
            })
        
        # Check for outliers (using IQR method)
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        outliers = ((values < Q1 - 1.5*IQR) | (values > Q3 + 1.5*IQR)).sum()
        if outliers > len(values) * 0.1:  # More than 10% outliers
            issues.append({
                'Dimension': dim,
                'Issue': 'Many outliers',
                'Details': f'{outliers} outliers ({outliers/len(values)*100:.1f}%)'
            })
        
        # Check for extreme skewness
        skewness = stats.skew(values)
        if abs(skewness) > 1.0:
            issues.append({
                'Dimension': dim,
                'Issue': 'High skewness',
                'Details': f'Skewness={skewness:.2f}, distribution is highly skewed'
            })
    
    if issues:
        return pd.DataFrame(issues)
    else:
        return None


def plot_scatter_matrix(df, dimension_cols, output_dir, n_samples=None):
    """Create scatter plot matrix for problematic dimensions."""
    # If too many dimensions, focus on a subset
    if len(dimension_cols) > 6:
        print("Too many dimensions for scatter matrix, showing first 6...")
        dimension_cols = dimension_cols[:6]
    
    # Sample data if too large
    if n_samples and len(df) > n_samples:
        df_sample = df[dimension_cols].sample(n_samples)
    else:
        df_sample = df[dimension_cols]
    
    # Create scatter matrix
    pd.plotting.scatter_matrix(df_sample, figsize=(15, 15), alpha=0.5, diagonal='hist')
    plt.suptitle('Scatter Plot Matrix of Dimensions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter matrix")


def analyze_specific_dimension(df, dimension, output_dir):
    """Deep dive into a specific problematic dimension."""
    values = df[dimension].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram with more detail
    ax = axes[0, 0]
    ax.hist(values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
    ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
    ax.set_xlabel('Rating Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{dimension} - Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax = axes[0, 1]
    ax.boxplot(values, vert=True)
    ax.set_ylabel('Rating Value', fontsize=12)
    ax.set_title(f'{dimension} - Box Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plot (normality check)
    ax = axes[1, 0]
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f'{dimension} - Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_values = np.sort(values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, cumulative, linewidth=2)
    ax.set_xlabel('Rating Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(f'{dimension} - Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dimension}_detailed_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed analysis for {dimension}")


def main():
    parser = argparse.ArgumentParser(description='Investigate low-performing dimensions')
    parser.add_argument('--ratings_file', type=str, 
                       default='./ratings/per_image_Slider_mean_sd_from_wide.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/dimension_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--focus_dims', nargs='+', type=str, default=None,
                       help='Specific dimensions to analyze in detail (e.g., Obstruction Effort)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DIMENSION QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Ratings file: {args.ratings_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load data
    print("Loading ratings data...")
    df, mean_cols, sd_cols, dimension_names = load_ratings_data(args.ratings_file)
    print(f"Loaded {len(df)} samples with {len(dimension_names)} dimensions")
    print(f"Dimensions: {dimension_names}")
    print()
    
    # Analyze distributions for mean columns (prediction targets)
    print("Analyzing rating distributions for prediction targets (means)...")
    stats_df = analyze_distribution(df, mean_cols, args.output_dir)
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'dimension_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}\n")
    
    # Print statistics
    print("=" * 80)
    print("DIMENSION STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print()
    
    # Identify problematic dimensions
    print("=" * 80)
    print("POTENTIAL ISSUES")
    print("=" * 80)
    
    # Low variance
    low_var = stats_df[stats_df['Std'] < 0.5]
    if not low_var.empty:
        print("\nLow Variance Dimensions (Std < 0.5):")
        for _, row in low_var.iterrows():
            print(f"  - {row['Dimension']}: Std={row['Std']:.3f}")
    
    # High skewness
    high_skew = stats_df[abs(stats_df['Skewness']) > 1.0]
    if not high_skew.empty:
        print("\nHighly Skewed Dimensions (|Skew| > 1.0):")
        for _, row in high_skew.iterrows():
            print(f"  - {row['Dimension']}: Skewness={row['Skewness']:.3f}")
    
    # Limited range
    limited_range = stats_df[stats_df['Range'] < 3.0]
    if not limited_range.empty:
        print("\nLimited Range Dimensions (Range < 3.0):")
        for _, row in limited_range.iterrows():
            print(f"  - {row['Dimension']}: Range={row['Range']:.3f} ({row['Min']:.2f} to {row['Max']:.2f})")
    
    print()
    
    # Analyze correlations (only for mean columns - the prediction targets)
    print("Analyzing inter-dimension correlations...")
    high_corr_df = analyze_correlations(df, mean_cols, args.output_dir)
    
    if high_corr_df is not None:
        print("\nHighly Correlated Dimension Pairs (|r| > 0.7):")
        print(high_corr_df.to_string(index=False))
        high_corr_path = os.path.join(args.output_dir, 'high_correlations.csv')
        high_corr_df.to_csv(high_corr_path, index=False)
        print(f"Saved to {high_corr_path}")
    else:
        print("No highly correlated dimension pairs found.")
    print()
    
    # Check data quality
    print("Checking data quality...")
    issues_df = analyze_data_quality(df, mean_cols, args.output_dir)
    
    if issues_df is not None:
        print("\nData Quality Issues:")
        print(issues_df.to_string(index=False))
        issues_path = os.path.join(args.output_dir, 'data_quality_issues.csv')
        issues_df.to_csv(issues_path, index=False)
        print(f"Saved to {issues_path}")
    else:
        print("No significant data quality issues found.")
    print()
    
    # Create scatter matrix
    print("Creating scatter plot matrix...")
    plot_scatter_matrix(df, mean_cols, args.output_dir, n_samples=200)
    print()
    
    # Detailed analysis for specific dimensions
    if args.focus_dims:
        print("Creating detailed analyses for specified dimensions...")
        for dim in args.focus_dims:
            dim_col = f"{dim}_mean"
            if dim_col in mean_cols:
                print(f"  Analyzing {dim}...")
                analyze_specific_dimension(df, dim_col, args.output_dir)
            else:
                print(f"  Warning: {dim} not found in data")
    else:
        # Analyze dimensions with lowest variance or highest skewness
        print("Creating detailed analyses for potentially problematic dimensions...")
        problematic = set()
        
        # Add low variance dimensions
        if not low_var.empty:
            problematic.update(low_var['Dimension'].tolist()[:3])
        
        # Add high skewness dimensions
        if not high_skew.empty:
            problematic.update(high_skew['Dimension'].tolist()[:3])
        
        for dim in problematic:
            print(f"  Analyzing {dim}...")
            analyze_specific_dimension(df, dim, args.output_dir)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {args.output_dir}")
    print()
    print("RECOMMENDATIONS:")
    print("- Check dimensions with low variance - they may be hard to predict")
    print("- Consider normalizing or transforming highly skewed dimensions")
    print("- Review highly correlated dimensions - they may provide redundant information")
    print("- Inspect data quality issues before training")


if __name__ == '__main__':
    main()
