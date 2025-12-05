"""
Augmentation strategies for psychological rating prediction.
"""
import torch
from torchvision import transforms
from typing import Literal


def get_augmentation(
    mode: Literal['conservative', 'moderate', 'strong'] = 'moderate',
    img_size: int = 224
) -> transforms.Compose:
    """Get augmentation pipeline based on intensity level."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'conservative':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif mode == 'moderate':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    
    elif mode == 'strong':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
    
    raise ValueError(f"Unknown augmentation mode: {mode}")


def get_validation_transform(img_size: int = 224) -> transforms.Compose:
    """Standard validation/test transform - no augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MixUpCutMix:
    """MixUp augmentation for regression tasks."""
    
    def __init__(self, mixup_alpha: float = 0.2, mixup_prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
    
    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        if torch.rand(1).item() > self.mixup_prob:
            return images, targets
        
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_images, mixed_targets
