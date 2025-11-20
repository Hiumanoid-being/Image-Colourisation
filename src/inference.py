import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F
from skimage import color
from cnn_transformer_colorizer_ver1 import CNNTransformerColorizer

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/cnn_transformer_colorizer.pth"
INPUT_IMAGE = "../data/processed/test/grayscale/Places365_val_00000065.jpg"  # path to grayscale input
OUTPUT_IMAGE = "../data/results/0001_colorized.png"

IMAGE_SIZE = (256, 256)  # size to resize input

# -----------------------------
# Load Model
# -----------------------------
model = CNNTransformerColorizer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# -----------------------------
# Load & preprocess grayscale image
# -----------------------------
img = Image.open(INPUT_IMAGE).convert("L").resize(IMAGE_SIZE)

# Convert to [-1, 1] range (matching training pipeline)
L = np.array(img)  # [0, 255]
L_norm = (L / 127.5) - 1.0  # [-1, 1]

# Create tensor [1, 1, H, W]
L_tensor = torch.tensor(L_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

print(f"Input L shape: {L_tensor.shape}")
print(f"Input L range: [{L_tensor.min().item():.4f}, {L_tensor.max().item():.4f}]")

# -----------------------------
# Predict AB channels
# -----------------------------
with torch.no_grad():
    pred_AB = model(L_tensor)  # [1, 2, H, W]

print(f"Output AB shape: {pred_AB.shape}")
print(f"Output AB range: [{pred_AB.min().item():.4f}, {pred_AB.max().item():.4f}]")

# Resize to match L size if needed
if pred_AB.shape[2:] != L_tensor.shape[2:]:
    pred_AB = F.interpolate(pred_AB, size=L_tensor.shape[2:], mode='bilinear', align_corners=False)
    print(f"Resized AB to: {pred_AB.shape}")

# Convert to numpy [H, W, 2] in [-1, 1] range
pred_AB_np = pred_AB.squeeze(0).permute(1, 2, 0).cpu().numpy()

# -----------------------------
# Reconstruct RGB image using skimage
# -----------------------------
# Convert from [-1, 1] back to CIELAB range
L_lab = (L_norm + 1.0) * 50.0  # [-1, 1] -> [0, 100]
AB_lab = pred_AB_np * 110.0     # [-1, 1] -> [-110, 110]

# Construct LAB image
LAB = np.zeros((L_lab.shape[0], L_lab.shape[1], 3))
LAB[:, :, 0] = L_lab
LAB[:, :, 1:] = AB_lab

# Convert LAB to RGB using skimage (more accurate than PIL)
rgb_array = color.lab2rgb(LAB)  # Returns [0, 1] range
rgb_array = np.clip(rgb_array, 0, 1)  # Ensure valid range

# Convert to [0, 255] for saving
rgb_img = Image.fromarray((rgb_array * 255).astype(np.uint8))

# -----------------------------
# Save and display
# -----------------------------
Path(OUTPUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
rgb_img.save(OUTPUT_IMAGE)
print(f"✅ Colorized image saved to {OUTPUT_IMAGE}")

# Optional: Also save a comparison image
try:
    # Create side-by-side comparison
    comparison = Image.new('RGB', (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1]))
    gray_rgb = Image.open(INPUT_IMAGE).convert("L").resize(IMAGE_SIZE).convert("RGB")
    comparison.paste(gray_rgb, (0, 0))
    comparison.paste(rgb_img, (IMAGE_SIZE[0], 0))
    
    comparison_path = Path(OUTPUT_IMAGE).parent / f"{Path(OUTPUT_IMAGE).stem}_comparison.png"
    comparison.save(comparison_path)
    print(f"✅ Comparison image saved to {comparison_path}")
except Exception as e:
    print(f"⚠️ Could not create comparison image: {e}")

rgb_img.show()