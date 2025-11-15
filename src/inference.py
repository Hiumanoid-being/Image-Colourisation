import torch
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F
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
L = np.array(img) / 255.0
L_tensor = torch.tensor(L, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

# -----------------------------
# Predict AB channels
# -----------------------------
with torch.no_grad():
    pred_AB = model(L_tensor)  # [1,2,H_out,W_out]

# Resize to match L size
pred_AB = F.interpolate(pred_AB, size=L_tensor.shape[2:], mode='bilinear', align_corners=False)
pred_AB = pred_AB.squeeze(0).permute(1,2,0).cpu().numpy()  # [H,W,2]

# -----------------------------
# Reconstruct RGB image
# -----------------------------
LAB = np.zeros((L.shape[0], L.shape[1], 3))
LAB[:,:,0] = L * 255
LAB[:,:,1:] = pred_AB * 128 + 128  # denormalize AB
rgb_img = Image.fromarray(LAB.astype(np.uint8), mode="LAB").convert("RGB")

# -----------------------------
# Save and display
# -----------------------------
Path(OUTPUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
rgb_img.save(OUTPUT_IMAGE)
print(f"✅ Colorized image saved to {OUTPUT_IMAGE}")
rgb_img.show()
