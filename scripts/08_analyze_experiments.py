#!/usr/bin/env python3
"""
Analyze and compare hyperparameter search experiments.

This script:
1. Scans all experiment output directories for results
2. Extracts training metrics, validation metrics, and configurations
3. Generates comparison tables and visualizations
4. Ranks experiments by various metrics
5. Provides recommendations for best hyperparameters

Usage:
    python scripts/08_analyze_experiments.py
    python scripts/08_analyze_experiments.py --experiments_dir ./outputs/experiments
    python scripts/08_analyze_experiments.py --metric val_loss --top_k 5
"""

import os
import sys
import argparse
import json
import glob
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def find_experiment_runs(experiments_dir):
    """Find all experiment run directories with results."""
    experiments = {}
    
    exp_dir = Path(experiments_dir)
    if not exp_dir.exists():
        print(f"  ✗ Experiments directory not found: {exp_dir}")
        return experiments
    
    # Look for experiment folders
    for exp_folder in sorted(exp_dir.iterdir()):
        if not exp_folder.is_dir():
            continue
        
        exp_name = exp_folder.name
        
        # Find run folders within each experiment
        run_folders = list(exp_folder.glob('run_*'))
        
        if not run_folders:
            # Maybe results are directly in the experiment folder
            # Check for both CSV (actual) and JSON (legacy) formats
            if (exp_folder / 'training_history.csv').exists() or (exp_folder / 'training_history.json').exists():
                run_folders = [exp_folder]
        
        for run_folder in run_folders:
            # Check for required files - support both CSV (actual) and JSON (legacy)
            history_file_csv = run_folder / 'training_history.csv'
            history_file_json = run_folder / 'training_history.json'
            history_file = history_file_csv if history_file_csv.exists() else history_file_json
            
            # Support both args.json (actual) and config.json (legacy)
            config_file_args = run_folder / 'args.json'
            config_file_config = run_folder / 'config.json'
            config_file = config_file_args if config_file_args.exists() else config_file_config
            
            if history_file.exists():
                run_id = f"{exp_name}/{run_folder.name}" if run_folder != exp_folder else exp_name
                experiments[run_id] = {
                    'path': run_folder,
                    'exp_name': exp_name,
                    'history_file': history_file,
                    'config_file': config_file if config_file.exists() else None,
                }
    
    return experiments


def load_experiment_data(exp_info):
    """Load training history and config for an experiment."""
    data = {
        'path': str(exp_info['path']),
        'exp_name': exp_info['exp_name'],
    }
    
    # Load training history - support both CSV and JSON formats
    try:
        history_file = exp_info['history_file']
        
        if str(history_file).endswith('.csv'):
            # Load CSV format (actual format from training script)
            history_df = pd.read_csv(history_file)
            # Convert DataFrame to dict of lists for consistent handling
            history = {col: history_df[col].tolist() for col in history_df.columns}
        else:
            # Load JSON format (legacy)
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        data['history'] = history
        
        # Extract key metrics from history
        if 'val_loss' in history:
            data['best_val_loss'] = min(history['val_loss'])
            data['best_val_epoch'] = history['val_loss'].index(data['best_val_loss']) + 1
            data['final_val_loss'] = history['val_loss'][-1]
        
        if 'train_loss' in history:
            data['best_train_loss'] = min(history['train_loss'])
            data['final_train_loss'] = history['train_loss'][-1]
        
        # Check for per-component metrics (PC1, PC2, etc.)
        for i in range(1, 5):
            key = f'val_r2_PC{i}'
            if key in history:
                data[f'best_r2_PC{i}'] = max(history[key])
                data[f'final_r2_PC{i}'] = history[key][-1]
        
        # Check for correlation metrics
        if 'val_corr' in history:
            data['best_val_corr'] = max(history['val_corr'])
            data['final_val_corr'] = history['val_corr'][-1]
        
        # Check for MAE metrics
        if 'val_mae' in history:
            data['best_val_mae'] = min(history['val_mae'])
            data['final_val_mae'] = history['val_mae'][-1]
        
        # Mean R2 across components
        if 'val_r2_mean' in history:
            data['best_r2_mean'] = max(history['val_r2_mean'])
            data['final_r2_mean'] = history['val_r2_mean'][-1]
        
        data['epochs_trained'] = len(history.get('train_loss', []))
        
    except Exception as e:
        print(f"  ✗ Error loading history for {exp_info['exp_name']}: {e}")
        data['history'] = {}
    
    # Load config - support both args.json and config.json
    if exp_info['config_file']:
        try:
            with open(exp_info['config_file'], 'r') as f:
                config = json.load(f)
            data['config'] = config
            
            # Extract key hyperparameters
            for key in ['backbone', 'lr', 'dropout', 'weight_decay', 'optimizer', 
                       'scheduler', 'batch_size', 'aux_loss_weight', 'loss_fn']:
                if key in config:
                    data[key] = config[key]
                    
        except Exception as e:
            print(f"  ✗ Error loading config for {exp_info['exp_name']}: {e}")
            data['config'] = {}
    
    return data


def create_comparison_table(experiments_data):
    """Create a DataFrame comparing all experiments."""
    rows = []
    
    for exp_id, data in experiments_data.items():
        row = {
            'experiment': data['exp_name'],
            'run_id': exp_id,
            'epochs': data.get('epochs_trained', 0),
            'best_val_loss': data.get('best_val_loss', np.nan),
            'final_val_loss': data.get('final_val_loss', np.nan),
            'best_train_loss': data.get('best_train_loss', np.nan),
            'final_train_loss': data.get('final_train_loss', np.nan),
            'best_epoch': data.get('best_val_epoch', 0),
            'best_val_corr': data.get('best_val_corr', np.nan),
            'best_r2_mean': data.get('best_r2_mean', np.nan),
        }
        
        # Add per-component R2
        for i in range(1, 5):
            row[f'best_r2_PC{i}'] = data.get(f'best_r2_PC{i}', np.nan)
        
        # Add hyperparameters
        for key in ['backbone', 'lr', 'dropout', 'weight_decay', 'optimizer', 
                   'aux_loss_weight', 'loss_fn']:
            row[key] = data.get(key, 'N/A')
        
        # Calculate overfitting gap
        if not np.isnan(row['final_val_loss']) and not np.isnan(row['final_train_loss']):
            row['overfit_gap'] = row['final_val_loss'] - row['final_train_loss']
        else:
            row['overfit_gap'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by best validation loss
    if 'best_val_loss' in df.columns:
        df = df.sort_values('best_val_loss', ascending=True)
    
    return df


def plot_training_curves(experiments_data, output_dir, top_k=10):
    """Plot training curves for top experiments."""
    
    # Get experiments sorted by best val loss
    sorted_exps = sorted(
        experiments_data.items(),
        key=lambda x: x[1].get('best_val_loss', float('inf'))
    )[:top_k]
    
    if not sorted_exps:
        print("  ✗ No experiments with valid training history found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_exps)))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for (exp_id, data), color in zip(sorted_exps, colors):
        history = data.get('history', {})
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label=data['exp_name'], color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for (exp_id, data), color in zip(sorted_exps, colors):
        history = data.get('history', {})
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label=data['exp_name'], color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Correlation (if available)
    ax = axes[1, 0]
    has_corr = False
    for (exp_id, data), color in zip(sorted_exps, colors):
        history = data.get('history', {})
        if 'val_corr' in history:
            ax.plot(history['val_corr'], label=data['exp_name'], color=color, alpha=0.8)
            has_corr = True
        elif 'val_r2_mean' in history:
            ax.plot(history['val_r2_mean'], label=data['exp_name'], color=color, alpha=0.8)
            has_corr = True
    if has_corr:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation R² / Correlation')
        ax.set_title('Validation Performance Curves')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No correlation data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Overfitting Analysis (Train vs Val gap)
    ax = axes[1, 1]
    for (exp_id, data), color in zip(sorted_exps, colors):
        history = data.get('history', {})
        if 'train_loss' in history and 'val_loss' in history:
            gap = np.array(history['val_loss']) - np.array(history['train_loss'])
            ax.plot(gap, label=data['exp_name'], color=color, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss - Train Loss')
    ax.set_title('Overfitting Gap (lower = less overfitting)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved training curves comparison")


def plot_hyperparameter_analysis(df, output_dir):
    """Analyze impact of different hyperparameters."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Backbone comparison
    ax = axes[0, 0]
    if 'backbone' in df.columns and df['backbone'].nunique() > 1:
        backbone_perf = df.groupby('backbone')['best_val_loss'].agg(['mean', 'std', 'min']).reset_index()
        bars = ax.bar(backbone_perf['backbone'], backbone_perf['mean'], 
                     yerr=backbone_perf['std'], capsize=5, alpha=0.7)
        ax.scatter(backbone_perf['backbone'], backbone_perf['min'], 
                  color='red', s=100, zorder=5, label='Best')
        ax.set_xlabel('Backbone')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Performance by Backbone')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'Single backbone used', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Learning Rate vs Performance
    ax = axes[0, 1]
    if 'lr' in df.columns:
        valid_lr = df[df['lr'] != 'N/A'].copy()
        if len(valid_lr) > 0:
            valid_lr['lr'] = valid_lr['lr'].astype(float)
            ax.scatter(valid_lr['lr'], valid_lr['best_val_loss'], s=100, alpha=0.7)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate (log scale)')
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('Learning Rate vs Performance')
            
            # Add experiment labels
            for _, row in valid_lr.iterrows():
                ax.annotate(row['experiment'], (row['lr'], row['best_val_loss']),
                           fontsize=7, alpha=0.7)
    
    # Plot 3: Dropout vs Performance
    ax = axes[0, 2]
    if 'dropout' in df.columns:
        valid_drop = df[df['dropout'] != 'N/A'].copy()
        if len(valid_drop) > 0:
            valid_drop['dropout'] = valid_drop['dropout'].astype(float)
            ax.scatter(valid_drop['dropout'], valid_drop['best_val_loss'], s=100, alpha=0.7)
            ax.set_xlabel('Dropout Rate')
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('Dropout vs Performance')
    
    # Plot 4: Weight Decay vs Performance
    ax = axes[1, 0]
    if 'weight_decay' in df.columns:
        valid_wd = df[df['weight_decay'] != 'N/A'].copy()
        if len(valid_wd) > 0:
            valid_wd['weight_decay'] = valid_wd['weight_decay'].astype(float)
            ax.scatter(valid_wd['weight_decay'], valid_wd['best_val_loss'], s=100, alpha=0.7)
            ax.set_xscale('log')
            ax.set_xlabel('Weight Decay (log scale)')
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('Weight Decay vs Performance')
    
    # Plot 5: Aux Loss Weight vs Performance
    ax = axes[1, 1]
    if 'aux_loss_weight' in df.columns:
        valid_aux = df[df['aux_loss_weight'] != 'N/A'].copy()
        if len(valid_aux) > 0:
            valid_aux['aux_loss_weight'] = valid_aux['aux_loss_weight'].astype(float)
            ax.scatter(valid_aux['aux_loss_weight'], valid_aux['best_val_loss'], s=100, alpha=0.7)
            ax.set_xlabel('Auxiliary Loss Weight')
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('Aux Loss Weight vs Performance')
    
    # Plot 6: Per-component R2 comparison (top 5 experiments)
    ax = axes[1, 2]
    top_5 = df.head(5)
    pc_cols = [f'best_r2_PC{i}' for i in range(1, 5) if f'best_r2_PC{i}' in df.columns]
    if pc_cols:
        x = np.arange(len(top_5))
        width = 0.2
        for i, col in enumerate(pc_cols):
            if col in top_5.columns:
                ax.bar(x + i*width, top_5[col], width, label=col.replace('best_r2_', ''))
        ax.set_xlabel('Experiment')
        ax.set_ylabel('R² Score')
        ax.set_title('Per-Component R² (Top 5 Experiments)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(top_5['experiment'], rotation=45, ha='right')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No per-component R² data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved hyperparameter analysis")


def plot_ranking_chart(df, output_dir):
    """Create a horizontal bar chart ranking all experiments."""
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
    
    # Sort by best validation loss
    df_sorted = df.sort_values('best_val_loss', ascending=True)
    
    # Create color gradient
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df_sorted)))
    
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['best_val_loss'], color=colors, alpha=0.8)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['best_val_loss'] + 0.001, i, f"{row['best_val_loss']:.4f}", 
                va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['experiment'])
    ax.set_xlabel('Best Validation Loss')
    ax.set_title('Experiment Ranking (lower is better)')
    ax.invert_yaxis()  # Best at top
    
    # Add a vertical line at the best value
    best_val = df_sorted['best_val_loss'].min()
    ax.axvline(x=best_val, color='green', linestyle='--', alpha=0.7, label=f'Best: {best_val:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved experiment ranking chart")


def generate_recommendations(df, experiments_data):
    """Generate recommendations based on experiment results."""
    
    recommendations = []
    
    if len(df) == 0:
        return ["No completed experiments found. Please run experiments first."]
    
    # Best overall experiment
    best_exp = df.iloc[0]
    recommendations.append(f"🏆 BEST EXPERIMENT: {best_exp['experiment']}")
    recommendations.append(f"   Best Validation Loss: {best_exp['best_val_loss']:.4f}")
    recommendations.append(f"   Achieved at epoch: {best_exp['best_epoch']}")
    recommendations.append("")
    
    # Best hyperparameters
    recommendations.append("📋 RECOMMENDED HYPERPARAMETERS:")
    for key in ['backbone', 'lr', 'dropout', 'weight_decay', 'optimizer', 
               'aux_loss_weight', 'loss_fn']:
        if key in best_exp and best_exp[key] != 'N/A':
            recommendations.append(f"   {key}: {best_exp[key]}")
    recommendations.append("")
    
    # Overfitting analysis
    if 'overfit_gap' in df.columns and not df['overfit_gap'].isna().all():
        least_overfit = df.loc[df['overfit_gap'].idxmin()]
        most_overfit = df.loc[df['overfit_gap'].idxmax()]
        
        recommendations.append("📊 OVERFITTING ANALYSIS:")
        recommendations.append(f"   Least overfitting: {least_overfit['experiment']} (gap: {least_overfit['overfit_gap']:.4f})")
        recommendations.append(f"   Most overfitting: {most_overfit['experiment']} (gap: {most_overfit['overfit_gap']:.4f})")
        recommendations.append("")
    
    # Per-component analysis
    pc_cols = [f'best_r2_PC{i}' for i in range(1, 5) if f'best_r2_PC{i}' in df.columns]
    if pc_cols:
        recommendations.append("🎯 PER-COMPONENT PERFORMANCE (Best Experiment):")
        for col in pc_cols:
            pc_name = col.replace('best_r2_', '')
            value = best_exp[col]
            if not np.isnan(value):
                recommendations.append(f"   {pc_name}: R² = {value:.4f}")
        recommendations.append("")
    
    # Top 3 alternatives
    if len(df) > 1:
        recommendations.append("🥈 TOP 3 ALTERNATIVES:")
        for i, (_, row) in enumerate(df.head(4).iloc[1:].iterrows(), 2):
            recommendations.append(f"   #{i}: {row['experiment']} (val_loss: {row['best_val_loss']:.4f})")
        recommendations.append("")
    
    # Observations
    recommendations.append("💡 OBSERVATIONS:")
    
    # Check if regularization helped
    if 'dropout' in df.columns:
        high_reg = df[df['dropout'].apply(lambda x: float(x) if x != 'N/A' else 0) >= 0.6]
        low_reg = df[df['dropout'].apply(lambda x: float(x) if x != 'N/A' else 0) < 0.6]
        if len(high_reg) > 0 and len(low_reg) > 0:
            high_mean = high_reg['best_val_loss'].mean()
            low_mean = low_reg['best_val_loss'].mean()
            if high_mean < low_mean:
                recommendations.append("   • Higher dropout appears beneficial for this dataset")
            else:
                recommendations.append("   • Lower dropout appears sufficient for this dataset")
    
    # Check backbone performance
    if 'backbone' in df.columns and df['backbone'].nunique() > 1:
        backbone_perf = df.groupby('backbone')['best_val_loss'].min()
        best_backbone = backbone_perf.idxmin()
        recommendations.append(f"   • Best performing backbone: {best_backbone}")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search experiments')
    parser.add_argument('--experiments_dir', type=str, 
                       default='./outputs/experiments',
                       help='Directory containing experiment outputs')
    parser.add_argument('--output_dir', type=str,
                       default='./results/experiment_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top experiments to show in plots')
    
    args = parser.parse_args()
    
    # Setup paths
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HYPERPARAMETER EXPERIMENT ANALYSIS")
    print("=" * 80)
    print(f"Experiments directory: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find all experiments
    print("Step 1: Finding experiment runs...")
    experiments = find_experiment_runs(experiments_dir)
    print(f"  ✓ Found {len(experiments)} experiment runs")
    
    if not experiments:
        print("\n  ✗ No experiments found. Please check:")
        print(f"    - Directory exists: {experiments_dir}")
        print("    - Experiments have training_history.json files")
        sys.exit(1)
    
    for exp_id in experiments:
        print(f"    - {exp_id}")
    print()
    
    # Load all experiment data
    print("Step 2: Loading experiment data...")
    experiments_data = {}
    for exp_id, exp_info in experiments.items():
        data = load_experiment_data(exp_info)
        experiments_data[exp_id] = data
    print(f"  ✓ Loaded data for {len(experiments_data)} experiments")
    print()
    
    # Create comparison table
    print("Step 3: Creating comparison table...")
    df = create_comparison_table(experiments_data)
    
    # Save to CSV
    csv_path = output_dir / 'experiment_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved comparison table to {csv_path}")
    print()
    
    # Print summary table
    print("Step 4: Experiment Summary (sorted by best validation loss)")
    print("-" * 100)
    summary_cols = ['experiment', 'best_val_loss', 'final_val_loss', 'best_epoch', 
                   'epochs', 'backbone', 'lr', 'dropout']
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df[summary_cols].to_string(index=False))
    print("-" * 100)
    print()
    
    # Create visualizations
    print("Step 5: Creating visualizations...")
    plot_training_curves(experiments_data, output_dir, top_k=args.top_k)
    plot_hyperparameter_analysis(df, output_dir)
    plot_ranking_chart(df, output_dir)
    print()
    
    # Generate recommendations
    print("Step 6: Generating recommendations...")
    recommendations = generate_recommendations(df, experiments_data)
    
    # Save recommendations
    rec_path = output_dir / 'recommendations.txt'
    with open(rec_path, 'w') as f:
        f.write("HYPERPARAMETER SEARCH RECOMMENDATIONS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(recommendations))
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    for line in recommendations:
        print(line)
    print()
    
    # Final summary
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - experiment_comparison.csv : Full comparison table")
    print(f"  - training_curves_comparison.png : Training curves")
    print(f"  - hyperparameter_analysis.png : HP impact analysis")
    print(f"  - experiment_ranking.png : Ranking chart")
    print(f"  - recommendations.txt : Best hyperparameters")
    print()
    
    # Return best experiment info for scripting
    if len(df) > 0:
        best = df.iloc[0]
        print(f"Best model checkpoint: {experiments_data[best['run_id']]['path']}/best_model.pth")
    print()


if __name__ == '__main__':
    main()
