import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img_path, out_dir, split, image_size):
    img = Image.open(img_path).convert("RGB").resize(image_size)

    # Convert to LAB
    img_lab = img.convert("LAB")
    img_np = np.array(img_lab)

    # Split channels
    L = img_np[:, :, 0]
    AB = img_np[:, :, 1:]

    # Normalize
    L = L / 255.0
    AB = (AB - 128) / 128.0

    # Output paths
    gray_out = out_dir / split / "grayscale" / img_path.name
    color_out = out_dir / split / "color" / img_path.with_suffix(".npy").name

    # Save grayscale
    Image.fromarray((L * 255).astype(np.uint8)).save(gray_out)

    # Save AB
    np.save(color_out, AB)


# -----------------------------
# Main processing logic
# -----------------------------
def process_dataset(raw_dir, out_dir, splits, fraction, image_size):
    # Ensure output folders
    for split in splits:
        for sub in ["grayscale", "color"]:
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_dir = raw_dir / split

        if not split_dir.exists():
            print(f"Skipping missing split folder: {split_dir}")
            continue

        all_images = list(split_dir.rglob("*.jpg"))

        if not all_images:
            print(f"No JPG images found in: {split_dir}")
            continue

        # Subset size
        sample_size = max(1, int(len(all_images) * fraction))

        print(f"\n📌 {split}: Found {len(all_images)} images")
        print(f"➡ Processing {sample_size} images ({fraction * 100:.1f}%)")

        # Random selection
        sampled_images = random.sample(all_images, sample_size)

        # Process subset
        for img_path in tqdm(sampled_images):
            preprocess_image(img_path, out_dir, split, image_size)

    print("\n✅ Subset preprocessing complete!")
    print(f"Processed subset stored in: {out_dir.resolve()}")


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Preprocessing Script (Subset Version)")

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "sample"),
        help="Path to the raw dataset (containing train/test folders)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "processed_subset"),
        help="Directory to save processed subset"
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of dataset to process (default 0.10)"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Output image size as: --image_size 64 64"
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    fraction = args.fraction
    image_size = tuple(args.image_size)
    splits = ["train", "test"]

    process_dataset(raw_dir, out_dir, splits, fraction, image_size)


"""
1. Use defaults (10% subset, 64×64):
python preprocess_subset.py

2. Process 20% of dataset
python preprocess_subset.py --fraction 0.20

3. Change output image size
python preprocess_subset.py --image_size 128 128

4. Specify custom raw/processed directories
python preprocess_subset.py \
    --raw_dir "/path/to/raw/data" \
    --out_dir "/path/to/save/output" \
    --fraction 0.15

5. Current use
python preprocess_subset.py \
    --image_size 128 128 \
    --fraction 0.10
"""