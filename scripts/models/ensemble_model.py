"""
Ensemble model combining ViT and ResNet backbones.
Rationale: ViT captures global semantics, ResNet captures local texture/color patterns.
"""
import torch
import torch.nn as nn
from torchvision import models


class EnsembleRegressor(nn.Module):
    """
    Combines ViT-B/16 and ResNet50 features for multi-dimensional regression.
    
    Architecture:
        Image → [ViT features (768) | ResNet features (2048)] → MLP → outputs
    """
    
    def __init__(
        self, 
        n_outputs: int = 4,
        freeze_backbones: bool = True,
        unfreeze_layers: int = 0,
        dropout: float = 0.3,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.n_outputs = n_outputs
        
        # Load pretrained backbones
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Get feature dimensions
        self.vit_dim = self.vit.heads.head.in_features  # 768
        self.resnet_dim = self.resnet.fc.in_features     # 2048
        
        # Remove classification heads
        self.vit.heads.head = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # Freeze backbones
        if freeze_backbones:
            self._freeze_backbones(unfreeze_layers)
        
        # Combined regression head
        combined_dim = self.vit_dim + self.resnet_dim  # 2816
        
        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs)
        )
    
    def _freeze_backbones(self, unfreeze_layers: int = 0):
        """Freeze backbone parameters, optionally leaving last N layers unfrozen."""
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        if unfreeze_layers > 0:
            for block in self.vit.encoder.layers[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            resnet_layers = [self.resnet.layer4, self.resnet.layer3, 
                           self.resnet.layer2, self.resnet.layer1]
            for layer in resnet_layers[:unfreeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_features = self.vit(x)
        resnet_features = self.resnet(x)
        combined = torch.cat([vit_features, resnet_features], dim=1)
        return self.head(combined)


class EnsembleWithDecoder(nn.Module):
    """
    Ensemble model with PCA decoder for 14-dim reconstruction.
    """
    
    def __init__(
        self,
        n_pca_components: int = 4,
        n_original_dims: int = 14,
        pca_components: torch.Tensor = None,
        pca_mean: torch.Tensor = None,
        freeze_decoder: bool = True,
        **ensemble_kwargs
    ):
        super().__init__()
        
        self.encoder = EnsembleRegressor(n_outputs=n_pca_components, **ensemble_kwargs)
        self.decoder = nn.Linear(n_pca_components, n_original_dims, bias=True)
        
        if pca_components is not None:
            with torch.no_grad():
                self.decoder.weight.copy_(pca_components.T)
        
        if pca_mean is not None:
            with torch.no_grad():
                self.decoder.bias.copy_(pca_mean)
        
        if freeze_decoder:
            self.decoder.weight.requires_grad = False
            self.decoder.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> dict:
        pc_scores = self.encoder(x)
        reconstructed = self.decoder(pc_scores)
        return {'pc_scores': pc_scores, 'reconstructed': reconstructed}
