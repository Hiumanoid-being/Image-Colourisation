import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CNN Encoder
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # H/2, W/2
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # H/4, W/4
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, feature_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Input x is L channel in [-1, 1] range
        return self.encoder(x)  # [B, feature_dim, H/4, W/4]


# ============================================================
# Transformer Encoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, patch_size=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        
        # Calculate projection dimension (patch_size^2 * feature_dim)
        patch_dim = patch_size * patch_size * feature_dim
        
        # Create projection layer as a module (will be trained)
        self.patch_projection = nn.Linear(patch_dim, feature_dim)
        
        # Positional embedding (will be adjusted dynamically if needed)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, feature_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape

        # Split feature map into patches
        x = x.unfold(2, self.patch_size, self.patch_size)\
             .unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).flatten(2)  # [B, N_patches, C * patch_area]

        # Project patches to feature dimension
        x = self.patch_projection(x)  # [B, N_patches, feature_dim]
        
        # Add positional embedding
        num_patches = x.size(1)
        if num_patches <= self.pos_embed.size(1):
            pos_emb = self.pos_embed[:, :num_patches, :]
        else:
            # Interpolate if we have more patches than expected
            pos_emb = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=num_patches,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        
        x = x + pos_emb

        # Transformer processing
        x = self.transformer(x)
        return x, (H, W)  # Return original spatial dims


# ============================================================
# Decoder (Upsampling back to color space)
# ============================================================
class ColorDecoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Only 2 upsampling layers to match encoder's 4x downsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 128, 4, stride=2, padding=1),  # 2x
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4x total
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 2, 3, stride=1, padding=1),
            nn.Tanh()  # final a,b in [-1,1]
        )

    def forward(self, x, orig_H, orig_W):
        """
        Args:
            x: [B, N_patches, feature_dim]
            orig_H, orig_W: Original input image dimensions
        """
        B, N, C = x.shape
        side = int(N ** 0.5)
        
        # Reshape patches back to spatial grid
        x = x.permute(0, 2, 1).view(B, C, side, side)  # [B, feature_dim, H', W']

        # Apply decoder upsampling
        x = self.decoder(x)  # [B, 2, H/4*4, W/4*4] = [B, 2, H, W] approximately
        
        # Final resize to exact target dimensions (if needed)
        if x.shape[2] != orig_H or x.shape[3] != orig_W:
            x = F.interpolate(x, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
        
        return x


# ============================================================
# Full Model
# ============================================================
class CNNTransformerColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.transformer = TransformerEncoder()
        self.decoder = ColorDecoder()

    def forward(self, x):
        orig_H, orig_W = x.shape[2], x.shape[3]
        
        # Encode
        feat = self.encoder(x)  # [B, 256, H/4, W/4]
        
        # Transform
        trans, (enc_H, enc_W) = self.transformer(feat)  # [B, N_patches, 256]
        
        # Decode
        ab = self.decoder(trans, orig_H, orig_W)  # [B, 2, H, W]
        
        return ab