from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"            # source images: train/test, (note the use of sample subset)
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"   # destination for processed images
IMAGE_SIZE = (128, 128)                                                         # resize images
splits = ["train", "test"]                                                      # subfolders in sample/


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
    # Open and resize
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    
    # Convert to LAB color space
    img_lab = img.convert("LAB")
    img_np = np.array(img_lab)
    
    # Split channels
    L = img_np[:, :, 0]           # grayscale
    AB = img_np[:, :, 1:]         # color channels
    
    # Normalize
    L = L / 255.0
    AB = (AB - 128) / 128.0       # scale [-1,1]
    
    # Output paths
    gray_out = PROCESSED_DIR / split / "grayscale" / img_path.name
    color_out = PROCESSED_DIR / split / "color" / img_path.with_suffix(".npy").name
    
    # Save files
    Image.fromarray((L * 255).astype(np.uint8)).save(gray_out)
    np.save(color_out, AB)

# -----------------------------
# Process all images in each split
# -----------------------------
for split in splits:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"Skipping missing split folder: {split_dir}")
        continue
    
    print(f"Processing {split} images...")
    for img_path in tqdm(list(split_dir.rglob("*.jpg"))):
        preprocess_image(img_path, split)

print("✅ Preprocessing complete!")
print(f"Processed images stored in: {PROCESSED_DIR.resolve()}")
