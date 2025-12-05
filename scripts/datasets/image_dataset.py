"""
Image dataset for psychological rating prediction.
"""
import io
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageRegressionDataset(Dataset):
    """
    Dataset for loading images and their psychological ratings.
    
    Supports both:
    - Original 14-dimensional ratings (columns ending with _mean)
    - PCA-transformed ratings (columns like PC1, PC2, ...)
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform=None,
        use_pca: bool = False,
        n_components: int = 4,
        return_original: bool = False,
        original_csv: Optional[str] = None
    ):
        """
        Args:
            csv_file: Path to CSV with ratings (PCA or original)
            image_dir: Directory containing images
            transform: torchvision transforms to apply
            use_pca: If True, expect PCA columns (PC1, PC2, ...)
            n_components: Number of PCA components (if use_pca=True)
            return_original: Also return original 14-dim ratings (for aux loss)
            original_csv: Path to original ratings CSV (if return_original=True)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.use_pca = use_pca
        self.return_original = return_original
        
        # Load main CSV
        self.df = self._load_csv(csv_file)
        
        # Determine target columns
        if use_pca:
            self.target_cols = [f'PC{i+1}' for i in range(n_components)]
        else:
            self.target_cols = [col for col in self.df.columns if col.endswith('_mean')]
        
        # Validate columns exist
        missing = [c for c in self.target_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # Load original ratings if needed
        self.original_df = None
        self.original_cols = None
        if return_original and original_csv:
            self.original_df = self._load_csv(original_csv)
            self.original_cols = [col for col in self.original_df.columns if col.endswith('_mean')]
        
        # Build valid samples list
        self.samples = self._build_samples()
        print(f"Dataset: {len(self.samples)} samples, {len(self.target_cols)} targets")
    
    def _load_csv(self, csv_file: str) -> pd.DataFrame:
        """Load CSV with quirky header handling."""
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Fix header with extra quotes
        header = lines[0].strip().replace('""', '"').replace('"', '')
        lines[0] = header + '\n'
        
        return pd.read_csv(io.StringIO(''.join(lines)))
    
    def _build_samples(self) -> List[dict]:
        """Build list of valid samples with existing images."""
        samples = []
        
        for idx, row in self.df.iterrows():
            img_name = Path(row['image']).name
            img_path = self.image_dir / img_name
            
            if img_path.exists():
                sample = {
                    'image_path': str(img_path),
                    'image_name': row['image'],
                    'targets': np.array([row[c] for c in self.target_cols], dtype=np.float32)
                }
                
                # Add original targets if available
                if self.original_df is not None:
                    orig_row = self.original_df[self.original_df['image'] == row['image']]
                    if len(orig_row) > 0:
                        sample['original_targets'] = np.array(
                            [orig_row.iloc[0][c] for c in self.original_cols], 
                            dtype=np.float32
                        )
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'target': torch.from_numpy(sample['targets']),
            'image_name': sample['image_name']
        }
        
        if 'original_targets' in sample:
            result['original_target'] = torch.from_numpy(sample['original_targets'])
        
        return result
