#!/usr/bin/env python3
"""
Automated experiment runner for assessor model training with PCA pipeline.
Runs multiple experiments with different configurations.

This script:
1. Defines multiple hyperparameter configurations to test
2. Generates SLURM scripts for each configuration
3. Submits jobs to the cluster (optionally in parallel or sequentially)

IMPORTANT: This version is configured for the PCA + Decoder architecture.
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


# =============================================================================
# PCA PIPELINE CONFIGURATION
# =============================================================================
# These are the fixed paths for the PCA pipeline (shared across all experiments)
PCA_CONFIG = {
    'images_dir': './images',
    'ratings_file': './results/pca_analysis/ratings_pca_4comp_train.csv',
    'ratings_file_14': './ratings/per_image_Slider_mean_sd_from_wide.csv',
    'pca_model': './results/pca_analysis/pca_targets.joblib',
    'use_pca': True,
    'use_decoder': True,
    'use_amp': True,
    'pretrained': True,
    'num_workers': 8,
    'save_every': 10,
    'image_size': 224,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================
# Each experiment overrides specific hyperparameters
EXPERIMENTS = {
    'baseline': {
        'name': 'Baseline (ResNet50 + Decoder)',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.0001,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'high_reg': {
        'name': 'High Regularization',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.7,
            'lr': 0.0001,
            'weight_decay': 0.03,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'very_high_reg': {
        'name': 'Very High Regularization',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.7,
            'lr': 0.0001,
            'weight_decay': 0.05,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.3,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0 (smaller model)',
        'args': {
            'backbone': 'efficientnet_b0',
            'dropout': 0.5,
            'lr': 0.0001,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'efficientnet_b0_high_reg': {
        'name': 'EfficientNet-B0 + High Reg',
        'args': {
            'backbone': 'efficientnet_b0',
            'dropout': 0.6,
            'lr': 0.0001,
            'weight_decay': 0.02,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'low_lr': {
        'name': 'Lower Learning Rate',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.00003,
            'weight_decay': 0.01,
            'epochs': 100,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 20,
        }
    },
    'very_low_lr': {
        'name': 'Very Low Learning Rate',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.00001,
            'weight_decay': 0.01,
            'epochs': 100,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 20,
        }
    },
    'sgd_optimizer': {
        'name': 'SGD Optimizer',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.01,
            'weight_decay': 0.0001,
            'epochs': 50,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'efficientnet_b3': {
        'name': 'EfficientNet-B3',
        'args': {
            'backbone': 'efficientnet_b3',
            'dropout': 0.5,
            'lr': 0.0001,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'best_combo': {
        'name': 'Best Combination (small model + high reg + low LR)',
        'args': {
            'backbone': 'efficientnet_b0',
            'dropout': 0.6,
            'lr': 0.00005,
            'weight_decay': 0.02,
            'epochs': 80,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.25,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 12,
        }
    },
    'resnet101': {
        'name': 'ResNet101 (larger model)',
        'args': {
            'backbone': 'resnet101',
            'dropout': 0.6,
            'lr': 0.00005,
            'weight_decay': 0.02,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'high_aux_loss': {
        'name': 'Higher Auxiliary Loss Weight',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.0001,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.4,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
    'mse_loss': {
        'name': 'MSE Loss Function',
        'args': {
            'backbone': 'resnet50',
            'dropout': 0.5,
            'lr': 0.0001,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'mse',
            'early_stopping': 15,
        }
    },
    'vit_b_16': {
        'name': 'Vision Transformer (ViT-B/16)',
        'args': {
            'backbone': 'vit_b_16',
            'dropout': 0.5,
            'lr': 0.00005,
            'weight_decay': 0.01,
            'epochs': 50,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'aux_loss_weight': 0.2,
            'grad_clip': 1.0,
            'loss_fn': 'smooth_l1',
            'early_stopping': 15,
        }
    },
}


def create_slurm_script(exp_name, exp_config, base_dir, slurm_dir):
    """Create a SLURM script for the experiment with PCA + Decoder architecture."""
    
    # Merge PCA_CONFIG with experiment-specific args
    args = exp_config['args']
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=exp_{exp_name}
#SBATCH --output={base_dir}/logs/exp_{exp_name}_%j.out
#SBATCH --error={base_dir}/logs/exp_{exp_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Experiment: {exp_config['name']}
# Architecture: PCA + Decoder
echo "=================================================="
echo "Experiment: {exp_config['name']}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Load environment
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

# Navigate to project directory
cd {base_dir}

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# PCA Pipeline Configuration
IMAGES_DIR="{PCA_CONFIG['images_dir']}"
RATINGS_FILE="{PCA_CONFIG['ratings_file']}"
RATINGS_FILE_14="{PCA_CONFIG['ratings_file_14']}"
PCA_MODEL="{PCA_CONFIG['pca_model']}"
OUTPUT_DIR="./outputs/experiments/{exp_name}"

echo "=================================================="
echo "Training with PCA + Decoder Architecture"
echo "=================================================="
echo "Images: $IMAGES_DIR"
echo "Ratings: $RATINGS_FILE"
echo "PCA Model: $PCA_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python scripts/03_train_model.py \\
    --images_dir "$IMAGES_DIR" \\
    --ratings_file "$RATINGS_FILE" \\
    --ratings_file_14 "$RATINGS_FILE_14" \\
    --output_dir "$OUTPUT_DIR" \\
    --use_pca \\
    --pca_model "$PCA_MODEL" \\
    --use_decoder \\
    --backbone {args.get('backbone', 'resnet50')} \\
    --epochs {args.get('epochs', 50)} \\
    --batch_size {args.get('batch_size', 32)} \\
    --lr {args.get('lr', 0.0001)} \\
    --weight_decay {args.get('weight_decay', 0.01)} \\
    --optimizer {args.get('optimizer', 'adamw')} \\
    --scheduler {args.get('scheduler', 'cosine')} \\
    --dropout {args.get('dropout', 0.5)} \\
    --aux_loss_weight {args.get('aux_loss_weight', 0.2)} \\
    --grad_clip {args.get('grad_clip', 1.0)} \\
    --loss_fn {args.get('loss_fn', 'smooth_l1')} \\
    --image_size {PCA_CONFIG['image_size']} \\
    --train_ratio {PCA_CONFIG['train_ratio']} \\
    --val_ratio {PCA_CONFIG['val_ratio']} \\
    --num_workers {PCA_CONFIG['num_workers']} \\
    --early_stopping {args.get('early_stopping', 15)} \\
    --save_every {PCA_CONFIG['save_every']} \\
    --use_amp \\
    --pretrained

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="
"""
    
    # Save script
    script_path = os.path.join(slurm_dir, f'exp_{exp_name}.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Run multiple training experiments')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS.keys()) + ['all'],
                       default=['all'],
                       help='Which experiments to run (default: all)')
    parser.add_argument('--mode', choices=['create', 'submit', 'both'], default='both',
                       help='Create scripts only, submit only, or both (default: both)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without actually doing it')
    parser.add_argument('--sequential', action='store_true',
                       help='Submit jobs sequentially (with dependencies) instead of all at once')

    args = parser.parse_args()
    
    # Determine base directory
    base_dir = Path(__file__).parent.parent.absolute()
    slurm_dir = base_dir / 'slurm_scripts' / 'experiments'
    slurm_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    (base_dir / 'logs').mkdir(exist_ok=True)
    (base_dir / 'outputs' / 'experiments').mkdir(parents=True, exist_ok=True)
    
    # Select experiments
    if 'all' in args.experiments:
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = args.experiments
    
    print("=" * 80)
    print("EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"SLURM scripts will be saved to: {slurm_dir}")
    print(f"Mode: {args.mode}")
    print(f"Dry run: {args.dry_run}")
    print(f"Sequential: {args.sequential}")
    print(f"\nExperiments to run ({len(experiments_to_run)}):")
    for exp_name in experiments_to_run:
        print(f"  - {exp_name}: {EXPERIMENTS[exp_name]['name']}")
    print()
    
    # Create experiment info file
    exp_info = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {name: EXPERIMENTS[name] for name in experiments_to_run}
    }
    info_path = slurm_dir / 'experiment_info.json'
    with open(info_path, 'w') as f:
        json.dump(exp_info, f, indent=2)
    print(f"Saved experiment info to: {info_path}\n")
    
    # Create scripts
    script_paths = {}
    if args.mode in ['create', 'both']:
        print("Creating SLURM scripts...")
        for exp_name in experiments_to_run:
            exp_config = EXPERIMENTS[exp_name]
            script_path = create_slurm_script(exp_name, exp_config, str(base_dir), str(slurm_dir))
            script_paths[exp_name] = script_path
            print(f"  ✓ Created: {script_path}")
        print()
    
    # Submit jobs
    job_ids = {}
    if args.mode in ['submit', 'both'] and not args.dry_run:
        print("Submitting jobs...")
        prev_job_id = None
        
        for exp_name in experiments_to_run:
            script_path = script_paths.get(exp_name) or os.path.join(slurm_dir, f'exp_{exp_name}.sh')
            
            if args.sequential and prev_job_id:
                # Submit with dependency
                cmd = ['sbatch', f'--dependency=afterany:{prev_job_id}', script_path]
            else:
                cmd = ['sbatch', script_path]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                # Extract job ID from output like "Submitted batch job 12345"
                job_id = output.split()[-1]
                job_ids[exp_name] = job_id
                
                dep_str = f" (depends on {prev_job_id})" if args.sequential and prev_job_id else ""
                print(f"  ✓ Submitted {exp_name}: Job ID {job_id}{dep_str}")
                
                prev_job_id = job_id
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to submit {exp_name}: {e.stderr}")
        print()
    
    elif args.dry_run:
        print("DRY RUN - Would submit these jobs:")
        prev_job_id = "XXXXX"
        for exp_name in experiments_to_run:
            script_path = script_paths.get(exp_name) or os.path.join(slurm_dir, f'exp_{exp_name}.sh')
            dep_str = f" --dependency=afterany:{prev_job_id}" if args.sequential and prev_job_id != "XXXXX" else ""
            print(f"  sbatch{dep_str} {script_path}")
            prev_job_id = "YYYYY"
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Scripts created: {len(script_paths)}")
    print(f"Jobs submitted: {len(job_ids)}")
    
    if job_ids:
        print("\nJob IDs:")
        for exp_name, job_id in job_ids.items():
            print(f"  {exp_name}: {job_id}")
        
        print("\nTo check status:")
        print(f"  squeue -u $USER")
        print("\nTo cancel all jobs:")
        print(f"  scancel {' '.join(job_ids.values())}")
    
    if args.mode in ['create', 'both']:
        print(f"\nSLURM scripts saved to: {slurm_dir}")
        print("To submit manually:")
        print(f"  cd {slurm_dir}")
        print(f"  sbatch exp_<experiment_name>.sh")
    
    print("\nResults will be saved to:")
    print(f"  {base_dir}/outputs/experiments/")
    print()


if __name__ == '__main__':
    main()
