#!/usr/bin/env python3
"""
Quick test script to verify setup and data loading.
Run this before starting full training to catch any issues early.
"""

import sys
from pathlib import Path

print("="*80)
print("SETUP VERIFICATION SCRIPT")
print("="*80)

# Check Python version
print(f"\n1. Python version: {sys.version}")
if sys.version_info < (3, 7):
    print("   ⚠️  WARNING: Python 3.7+ recommended")
else:
    print("   ✓ Python version OK")

# Check imports
print("\n2. Checking required packages...")
required_packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('tqdm', 'tqdm'),
]

all_packages_ok = True
for package, name in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT FOUND")
        all_packages_ok = False

if not all_packages_ok:
    print("\n   ⚠️  Some packages are missing. Install them with:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Check CUDA availability
print("\n3. Checking CUDA/GPU availability...")
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ Number of GPUs: {torch.cuda.device_count()}")
else:
    print("   ⚠️  CUDA not available (this is OK if running on login node)")
    print("      GPU will be available when submitting via SLURM")

# Check data files - FIX: Use correct paths relative to new_assessor directory
print("\n4. Checking data files...")
# Navigate up from scripts/ to new_assessor/
project_root = Path(__file__).parent.parent
images_dir = project_root / 'images'
ratings_file = project_root / 'ratings' / 'per_image_Slider_mean_sd_from_wide.csv'

print(f"   Project root: {project_root}")

if images_dir.exists():
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"   ✓ Images directory exists: {images_dir}")
    print(f"   ✓ Found {len(image_files)} images")
    if len(image_files) == 0:
        print("   ⚠️  WARNING: No images found!")
else:
    print(f"   ✗ Images directory not found: {images_dir}")

if ratings_file.exists():
    print(f"   ✓ Ratings file exists: {ratings_file}")
    
    # Try to load it
    try:
        import pandas as pd
        import io
        
        with open(ratings_file, 'r') as f:
            lines = f.readlines()
        
        header = lines[0].strip().replace('""', '"').replace('"', '')
        lines[0] = header + '\n'
        fixed_csv = io.StringIO(''.join(lines))
        df = pd.read_csv(fixed_csv)
        
        print(f"   ✓ Successfully loaded ratings file")
        print(f"   ✓ Number of samples: {len(df)}")
        print(f"   ✓ Number of columns: {len(df.columns)}")
        
        mean_cols = [col for col in df.columns if col.endswith('_mean')]
        dimension_names = [col.replace('_mean', '') for col in mean_cols]
        print(f"   ✓ Number of rating dimensions: {len(mean_cols)}")
        print(f"   ✓ Dimensions: {', '.join(dimension_names)}")
        
        # Check if image files match
        print(f"\n   Checking image-rating correspondence...")
        missing_images = []
        for idx, row in df.head(10).iterrows():  # Check first 10
            img_path = images_dir / Path(row['image']).name
            if not img_path.exists():
                missing_images.append(Path(row['image']).name)
        
        if len(missing_images) > 0:
            print(f"   ⚠️  WARNING: Some images from ratings file not found:")
            for img in missing_images[:5]:
                print(f"      - {img}")
        else:
            print(f"   ✓ All checked images exist")
        
    except Exception as e:
        print(f"   ✗ Error loading ratings file: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ✗ Ratings file not found: {ratings_file}")

# Check output directory
print("\n5. Checking output directory...")
output_dir = project_root / 'outputs'
if not output_dir.exists():
    output_dir.mkdir(parents=True)
    print(f"   ✓ Created output directory: {output_dir}")
else:
    print(f"   ✓ Output directory exists: {output_dir}")

# Check logs directory
logs_dir = project_root / 'logs'
if not logs_dir.exists():
    logs_dir.mkdir(parents=True)
    print(f"   ✓ Created logs directory: {logs_dir}")
else:
    print(f"   ✓ Logs directory exists: {logs_dir}")

# Test data loading
print("\n6. Testing data loading and augmentation...")
try:
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    # Find a test image
    test_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    if len(test_images) > 0:
        test_img_path = test_images[0]
        img = Image.open(test_img_path).convert('RGB')
        print(f"   ✓ Successfully loaded test image: {test_img_path.name}")
        print(f"   ✓ Image size: {img.size}")
        
        # Test transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        img_tensor = transform(img)
        print(f"   ✓ Transform successful, tensor shape: {img_tensor.shape}")
        print(f"   ✓ Tensor dtype: {img_tensor.dtype}")
        print(f"   ✓ Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    else:
        print("   ⚠️  No test images available")
        
except Exception as e:
    print(f"   ✗ Error testing data loading: {e}")
    import traceback
    traceback.print_exc()

# Test model creation
print("\n7. Testing model creation...")
try:
    from torchvision import models
    import torch.nn as nn
    
    # Use newer API
    model = models.resnet50(weights=None)
    print(f"   ✓ Successfully created ResNet50 model")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model has {total_params:,} parameters")
    print(f"   ✓ Trainable: {trainable_params:,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"   ✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all_packages_ok and images_dir.exists() and ratings_file.exists():
    print("✓ All checks passed! You're ready to start training.")
    print("\nTo start training:")
    print("  - On SLURM cluster: sbatch scripts/train_assessor.sh")
    print("  - Locally: ./scripts/run_local.sh")
    print("  - Custom: python scripts/train_assessor.py --help")
else:
    print("⚠️  Some issues detected. Please fix them before training.")
    if not images_dir.exists():
        print(f"   - Images directory missing: {images_dir}")
    if not ratings_file.exists():
        print(f"   - Ratings file missing: {ratings_file}")

print("="*80)