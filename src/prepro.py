from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import skimage.color as color

# Use this version!!!!!

RAW_DIR = Path("data/raw")
OUT_COLOR = Path("data/processed/color")
OUT_GRAY = Path("data/processed/grayscale")

# set None for full train, or an integer limit
# Note that 36.5k validation images are automatically processed in full
MAX_TRAIN_IMAGES = 120000



def get_image_paths(directory: Path, limit=None):
    """Return list of image paths, optionally capped by limit."""
    paths = sorted(list(directory.rglob("*.jpg")))
    if limit is not None:
        return paths[:limit]
    return paths


def process_image(img_path: Path):
    """Convert image to LAB, save .npy (AB only) and grayscale .jpg."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Convert RGB to LAB using scikit-image
    lab = color.rgb2lab(img_np).astype("float32")

    # Extract L (grayscale) channel - range [0, 100]
    L_channel = lab[..., 0]
    
    # Extract AB channels only - range approximately [-128, 127]
    AB_channels = lab[..., 1:]  # Shape: (H, W, 2)
    
    # Normalize AB channels to [-1, 1] for training
    # AB channels in LAB space typically range from -128 to 127
    # We'll use -110 to 110 as a safe normalization range (covers ~99% of values)
    AB_normalized = AB_channels / 110.0  # Now in approximately [-1.16, 1.16]
    AB_normalized = np.clip(AB_normalized, -1.0, 1.0)  # Clip to [-1, 1]
    
    # Convert L channel to grayscale image (0-255 range for JPEG)
    gray_img = Image.fromarray(np.uint8((L_channel / 100.0) * 255))

    # Save outputs
    rel_path = img_path.relative_to(RAW_DIR)
    color_path = OUT_COLOR / rel_path.with_suffix(".npy")
    gray_path = OUT_GRAY / rel_path.with_suffix(".jpg")

    color_path.parent.mkdir(parents=True, exist_ok=True)
    gray_path.parent.mkdir(parents=True, exist_ok=True)

    # Save AB channels only as .npy (shape: H, W, 2)
    np.save(color_path, AB_normalized)
    
    # Save grayscale L channel as JPEG
    gray_img.save(gray_path)


def verify_preprocessing(num_samples=5):
    """Verify that preprocessing worked correctly by checking some samples."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Find some processed files
    color_files = list(OUT_COLOR.rglob("*.npy"))[:num_samples]
    
    if not color_files:
        print("❌ No processed files found to verify!")
        return
    
    print(f"\nChecking {len(color_files)} processed files...\n")
    
    all_correct = True
    
    for i, color_file in enumerate(color_files, 1):
        try:
            AB = np.load(color_file)
            
            print(f"File {i}: {color_file.name}")
            print(f"  Shape: {AB.shape}")
            print(f"  Dtype: {AB.dtype}")
            print(f"  Min/Max: {AB.min():.4f} / {AB.max():.4f}")
            print(f"  Mean: {AB.mean():.4f}")
            
            # Check shape
            if len(AB.shape) == 3 and AB.shape[2] == 2:
                print(f"  ✅ Correct: 2 channels (AB)")
            else:
                print(f"  ❌ ERROR: Expected shape (H, W, 2), got {AB.shape}")
                all_correct = False
            
            # Check value range
            if AB.min() >= -1.0 and AB.max() <= 1.0:
                print(f"  ✅ Correct: Values in [-1, 1] range")
            else:
                print(f"  ⚠️  WARNING: Values outside [-1, 1] range")
                all_correct = False
            
            print()
            
        except Exception as e:
            print(f"❌ Error checking {color_file.name}: {e}\n")
            all_correct = False
    
    if all_correct:
        print("✅ All verification checks passed!")
    else:
        print("⚠️  Some issues found. Please review above.")
    
    print("=" * 60)


def main():
    train_dir = RAW_DIR / "train"
    val_dir = RAW_DIR / "val"

    # Check if raw directories exist
    if not train_dir.exists() and not val_dir.exists():
        print(f"❌ ERROR: Neither {train_dir} nor {val_dir} found!")
        print("Please ensure your raw data is in the correct location.")
        return

    # ---- TRAIN ----
    if train_dir.exists():
        train_paths = get_image_paths(train_dir, limit=MAX_TRAIN_IMAGES)
        print(f"Processing TRAIN: {len(train_paths)} images")
        for p in tqdm(train_paths, desc="Train"):
            try:
                process_image(p)
            except Exception as e:
                print(f"\n❌ Error processing {p}: {e}")
    else:
        print(f"⚠️  Skipping TRAIN (directory not found: {train_dir})")

    # ---- VAL ----
    if val_dir.exists():
        val_paths = get_image_paths(val_dir, limit=None)  # unlimited
        print(f"\nProcessing VAL: {len(val_paths)} images")
        for p in tqdm(val_paths, desc="Val"):
            try:
                process_image(p)
            except Exception as e:
                print(f"\n❌ Error processing {p}: {e}")
    else:
        print(f"⚠️  Skipping VAL (directory not found: {val_dir})")

    print("\n✅ DONE.")
    
    # Run verification
    verify_preprocessing(num_samples=5)


if __name__ == "__main__":
    main()