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

        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        # remove one downsampling here if needed
        nn.Conv2d(256, feature_dim, 3, stride=1, padding=1),
        nn.BatchNorm2d(feature_dim),
        nn.ReLU(),
    )

    def forward(self, x):
        return self.encoder(x)  # [B, feature_dim, H/8, W/8]


# ============================================================
# Transformer Encoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, patch_size=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
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
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).flatten(2)  # [B, N_patches, C * patch_area]

        # Reduce dimension for transformer input
        proj = nn.Linear(x.size(-1), self.feature_dim).to(x.device)
        x = proj(x) + self.pos_embed[:, :x.size(1), :]

        # Transformer processing
        x = self.transformer(x)
        return x  # [B, N_patches, feature_dim]


# ============================================================
# Decoder (Upsampling back to color space)
# ============================================================
class ColorDecoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),
            nn.Tanh()  # final a,b in [-1,1]
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        side = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, side, side)

        # Resize to match input image size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.decoder(x)



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
        feat = self.encoder(x)
        trans = self.transformer(feat)
        ab = self.decoder(trans, x.shape[2], x.shape[3])
        return ab
