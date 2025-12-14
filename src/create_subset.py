from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage import color

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGE_SIZE = (256, 256)
splits = ["train", "test"]

# -----------------------------
# Ensure output folders exist
# -----------------------------
for split in splits:
    for sub in ["grayscale", "color"]:
        (PROCESSED_DIR / split / sub).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img_path, split):
    """
    Convert RGB image to CIELAB and normalize to [-1, 1] range.
    Uses skimage for accurate CIELAB conversion.
    """
    # Open and resize
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    
    # Convert to numpy array [0, 255] -> [0, 1]
    img_rgb = np.array(img) / 255.0
    
    # Convert RGB to CIELAB using skimage
    # L: [0, 100], a: [-127, 127], b: [-127, 127]
    img_lab = color.rgb2lab(img_rgb)
    
    # Split channels
    L = img_lab[:, :, 0]      # L channel [0, 100]
    AB = img_lab[:, :, 1:]    # a, b channels [-127, 127]
    
    # Normalize to [-1, 1] range
    L_norm = (L / 50.0) - 1.0        # [0, 100] -> [-1, 1]
    AB_norm = AB / 110.0              # [-127, 127] -> approximately [-1.15, 1.15]
    
    # Clip to ensure strict [-1, 1] range
    L_norm = np.clip(L_norm, -1.0, 1.0)
    AB_norm = np.clip(AB_norm, -1.0, 1.0)
    
    # Output paths
    gray_out = PROCESSED_DIR / split / "grayscale" / img_path.name
    color_out = PROCESSED_DIR / split / "color" / img_path.with_suffix(".npy").name
    
    # Save L channel as grayscale image (convert back to [0, 255] for visualization)
    L_save = ((L_norm + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(L_save).save(gray_out)
    
    # Save AB channels as numpy array (keep in [-1, 1] range)
    np.save(color_out, AB_norm.astype(np.float32))

# -----------------------------
# Process all images in each split
# -----------------------------
for split in splits:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"Skipping missing split folder: {split_dir}")
        continue
    
    print(f"Processing {split} images...")
    image_list = list(split_dir.rglob("*.jpg"))
    
    if not image_list:
        print(f"No images found in {split_dir}")
        continue
    
    for img_path in tqdm(image_list):
        try:
            preprocess_image(img_path, split)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("✅ Preprocessing complete!")
print(f"Processed images stored in: {PROCESSED_DIR.resolve()}")
