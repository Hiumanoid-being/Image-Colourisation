import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """CNN encoder to extract hierarchical features from grayscale images."""
    
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, feature_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [B, 64, H, W]
        x2 = self.conv2(x1) # [B, 128, H/2, W/2]
        x3 = self.conv3(x2) # [B, 256, H/4, W/4]
        x4 = self.conv4(x3) # [B, 256, H/4, W/4]
        return x4, (x1, x2, x3)


class TransformerEncoder(nn.Module):
    """Transformer encoder for capturing long-range dependencies."""
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, patch_size=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        
        patch_dim = patch_size * patch_size * feature_dim
        self.patch_projection = nn.Linear(patch_dim, feature_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, feature_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape

        # Split feature map into patches
        x = x.unfold(2, self.patch_size, self.patch_size)\
             .unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).flatten(2)

        # Project patches
        x = self.patch_projection(x)
        
        # Add positional embedding
        num_patches = x.size(1)
        if num_patches <= self.pos_embed.size(1):
            pos_emb = self.pos_embed[:, :num_patches, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=num_patches,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        
        x = x + pos_emb
        x = self.transformer(x)
        return x, (H, W)


class ColourDecoder(nn.Module):
    """Decoder with skip connections to predict AB colour channels."""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, orig_H, orig_W, skip_features=None):
        B, N, C = x.shape
        side = int(N ** 0.5)
        
        x = x.permute(0, 2, 1).view(B, C, side, side)
        x = self.up1(x)
        
        if skip_features is not None:
            skip3 = skip_features[2]
            if skip3.shape[2:] != x.shape[2:]:
                skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip3], dim=1)
        
        x = self.up2(x)
        
        if skip_features is not None:
            skip2 = skip_features[1]
            if skip2.shape[2:] != x.shape[2:]:
                skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip2], dim=1)
        
        x = self.up3(x)
        
        if skip_features is not None:
            skip1 = skip_features[0]
            if skip1.shape[2:] != x.shape[2:]:
                skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip1], dim=1)
        
        x = self.final(x)
        
        if x.shape[2] != orig_H or x.shape[3] != orig_W:
            x = F.interpolate(x, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
        
        return x


class CNNTransformerColourizer(nn.Module):
    """
    Complete colourization model combining CNN encoder, Transformer, and decoder.
    
    Input: Grayscale image (L channel) [B, 1, H, W]
    Output: AB colour channels [B, 2, H, W]
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.transformer = TransformerEncoder(dropout=0.1)
        self.decoder = ColourDecoder()

    def forward(self, x):
        orig_H, orig_W = x.shape[2], x.shape[3]
        feat, skip_features = self.encoder(x)
        trans, (enc_H, enc_W) = self.transformer(feat)
        ab = self.decoder(trans, orig_H, orig_W, skip_features)
        return ab