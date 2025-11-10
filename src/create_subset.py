# ===============================================================
# Image Colorization Project — Dataset Subsampling Script
# ===============================================================

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
# Update the path definitions
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SUBSET_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
IMAGES_PER_CLASS = 100           # per class for training
TEST_RATIO = 0.2                 # 20% of train subset size

# Ensure subset folders exist
for split in ["train", "val"]:
    (SUBSET_DIR / split).mkdir(parents=True, exist_ok=True)

# Helper to safely copy an image
def safe_copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

# 1️⃣ Subset from TRAIN
train_dir = RAW_DIR / "train"
print("📦 Creating train subset...")

for class_dir in tqdm(list(train_dir.iterdir())):
    if not class_dir.is_dir():
        continue
    all_images = list(class_dir.glob("*.jpg"))
    if len(all_images) == 0:
        continue

    # Randomly sample N images
    sample_size = min(IMAGES_PER_CLASS, len(all_images))
    subset_imgs = random.sample(all_images, sample_size)

    # Copy to subset/train/<class_name>/
    for img_path in subset_imgs:
        dest = SUBSET_DIR / "train" / class_dir.name / img_path.name
        safe_copy(img_path, dest)

print("✅ Train subset complete.")

# 2️⃣ Subset from TEST (proportional ratio)
test_dir = RAW_DIR / "val"  # or "test" depending on your dataset
print("📦 Creating test subset...")

for class_dir in tqdm(list(test_dir.iterdir())):
    if not class_dir.is_dir():
        continue
    all_images = list(class_dir.glob("*.jpg"))
    if len(all_images) == 0:
        continue

    sample_size = min(int(IMAGES_PER_CLASS * TEST_RATIO), len(all_images))
    subset_imgs = random.sample(all_images, sample_size)

    # Copy to subset/test/<class_name>/
    for img_path in subset_imgs:
        dest = SUBSET_DIR / "test" / class_dir.name / img_path.name
        safe_copy(img_path, dest)

print("✅ Test subset complete.")
print(f"Subset dataset created at: {SUBSET_DIR.resolve()}")
