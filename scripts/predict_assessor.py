#!/usr/bin/env python3
"""
Inference script for the trained psychological rating assessor.
Use this to predict ratings for new images.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pickle


class MultiOutputRegressionModel(nn.Module):
    """Same model architecture as in training."""
    
    def __init__(self, num_outputs=14, backbone='resnet50', pretrained=False, dropout=0.5):
        super(MultiOutputRegressionModel, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
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


def get_transforms(image_size=224):
    """Standard inference transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args = checkpoint['args']
    dimension_names = checkpoint['dimension_names']
    
    model = MultiOutputRegressionModel(
        num_outputs=len(dimension_names),
        backbone=args['backbone'],
        pretrained=False,
        dropout=args['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, dimension_names, args


def predict_single_image(image_path, model, transform, device):
    """Predict ratings for a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    return output.cpu().numpy()[0]


def predict_batch(image_paths, model, transform, device, batch_size=32):
    """Predict ratings for multiple images."""
    all_predictions = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        all_predictions.append(outputs.cpu().numpy())
    
    return np.vstack(all_predictions)


def main(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, dimension_names, train_args = load_model(args.checkpoint, device)
    print(f"Model loaded successfully")
    print(f"Predicting {len(dimension_names)} dimensions: {dimension_names}")
    
    # Load target scaler if available
    scaler = None
    scaler_path = Path(args.checkpoint).parent / 'target_scaler.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded target scaler for denormalization")
    
    # Get transforms
    image_size = train_args.get('image_size', 224)
    transform = get_transforms(image_size)
    
    # Get image paths
    if args.image_dir:
        image_paths = list(Path(args.image_dir).glob('*.jpg'))
        image_paths += list(Path(args.image_dir).glob('*.png'))
        image_paths += list(Path(args.image_dir).glob('*.jpeg'))
        image_paths = [str(p) for p in sorted(image_paths)]
        print(f"Found {len(image_paths)} images in {args.image_dir}")
    elif args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(image_paths)} image paths from {args.image_list}")
    else:
        raise ValueError("Must provide either --image_dir or --image_list")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Predict
    print("Starting prediction...")
    predictions = predict_batch(image_paths, model, transform, device, args.batch_size)
    
    # Denormalize if scaler exists
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        print("Predictions denormalized")
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions, columns=dimension_names)
    results_df.insert(0, 'image_path', image_paths[:len(predictions)])
    results_df.insert(1, 'image_name', [Path(p).name for p in image_paths[:len(predictions)]])
    
    # Save results
    output_path = args.output if args.output else 'predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    for dim in dimension_names:
        mean_val = results_df[dim].mean()
        std_val = results_df[dim].std()
        min_val = results_df[dim].min()
        max_val = results_df[dim].max()
        print(f"{dim:15s}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict psychological ratings for images')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--image_dir', type=str,
                        help='Directory containing images to predict')
    parser.add_argument('--image_list', type=str,
                        help='Text file with list of image paths (one per line)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    if not args.image_dir and not args.image_list:
        parser.error("Must provide either --image_dir or --image_list")
    
    main(args)
