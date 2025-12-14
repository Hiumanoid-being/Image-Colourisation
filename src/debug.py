from pathlib import Path
from PIL import Image
import numpy as np
from skimage import color
import random

# -----------------------------
# Configuration
# -----------------------------
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data/processed"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = "train"  # or "test"

# -----------------------------
# Get random image
# -----------------------------
gray_dir = PROCESSED_DIR / SPLIT / "grayscale"
color_dir = PROCESSED_DIR / SPLIT / "color"

gray_files = list(gray_dir.glob("*.jpg"))

if not gray_files:
    print(f"❌ No images found in {gray_dir}")
    exit()

# Pick a random image
random_gray = random.choice(gray_files)
random_color = color_dir / random_gray.with_suffix(".npy").name

print(f"Selected: {random_gray.name}")
print(f"Grayscale: {random_gray}")
print(f"Color data: {random_color}")

# -----------------------------
# Load preprocessed data
# -----------------------------
# Load L channel (saved as grayscale image [0, 255])
L_img = np.array(Image.open(random_gray))
print(f"\nL channel (image) range: [{L_img.min()}, {L_img.max()}]")

# Load AB channels (saved as .npy in [-1, 1] range)
AB_npy = np.load(random_color)
print(f"AB channels shape: {AB_npy.shape}")
print(f"AB channels range: [{AB_npy.min():.4f}, {AB_npy.max():.4f}]")

# -----------------------------
# Convert back to CIELAB color space
# -----------------------------
# Convert L from [0, 255] to [-1, 1] (as done in dataset)
L_norm = (L_img / 127.5) - 1.0

# Now convert both L and AB from [-1, 1] back to CIELAB range
L_lab = (L_norm + 1.0) * 50.0     # [-1, 1] -> [0, 100]
AB_lab = AB_npy * 110.0            # [-1, 1] -> [-110, 110]

print(f"\nCIELAB conversion:")
print(f"L range: [{L_lab.min():.2f}, {L_lab.max():.2f}] (expected: [0, 100])")
print(f"A range: [{AB_lab[:,:,0].min():.2f}, {AB_lab[:,:,0].max():.2f}] (expected: ~[-127, 127])")
print(f"B range: [{AB_lab[:,:,1].min():.2f}, {AB_lab[:,:,1].max():.2f}] (expected: ~[-127, 127])")

# -----------------------------
# Reconstruct full LAB image
# -----------------------------
LAB = np.zeros((L_lab.shape[0], L_lab.shape[1], 3))
LAB[:, :, 0] = L_lab
LAB[:, :, 1:] = AB_lab

# -----------------------------
# Convert LAB to RGB
# -----------------------------
rgb = color.lab2rgb(LAB)
rgb = np.clip(rgb, 0, 1)

print(f"\nRGB conversion:")
print(f"RGB range: [{rgb.min():.4f}, {rgb.max():.4f}] (expected: [0, 1])")

# -----------------------------
# Save reconstructed image
# -----------------------------
# Convert to [0, 255] and save
rgb_uint8 = (rgb * 255).astype(np.uint8)
output_path = OUTPUT_DIR / f"reconstructed_{random_gray.name}"

Image.fromarray(rgb_uint8).save(output_path)
print(f"\n✅ Reconstructed image saved to: {output_path}")

# -----------------------------
# Optional: Create comparison with original
# -----------------------------
original_path = Path("../data/sample") / SPLIT / random_gray.name

if original_path.exists():
    print(f"\n📊 Creating comparison with original image...")
    
    # Load original
    original_img = Image.open(original_path).convert("RGB")
    
    # Resize original to match processed size
    original_resized = original_img.resize((rgb_uint8.shape[1], rgb_uint8.shape[0]))
    
    # Create side-by-side comparison
    comparison = Image.new('RGB', (rgb_uint8.shape[1] * 3, rgb_uint8.shape[0]))
    
    # Paste images
    comparison.paste(original_resized, (0, 0))
    comparison.paste(Image.fromarray(L_img).convert("RGB"), (rgb_uint8.shape[1], 0))
    comparison.paste(Image.fromarray(rgb_uint8), (rgb_uint8.shape[1] * 2, 0))
    
    # Add labels (optional - requires PIL ImageDraw)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        font_size = max(12, rgb_uint8.shape[0] // 20)
        
        # Simple text without custom font
        draw.text((10, 10), "Original", fill=(255, 255, 255))
        draw.text((rgb_uint8.shape[1] + 10, 10), "Grayscale (L)", fill=(255, 255, 255))
        draw.text((rgb_uint8.shape[1] * 2 + 10, 10), "Reconstructed", fill=(255, 255, 255))
    except:
        pass  # Skip labels if ImageDraw fails
    
    comparison_path = OUTPUT_DIR / f"comparison_{random_gray.name}"
    comparison.save(comparison_path)
    print(f"✅ Comparison saved to: {comparison_path}")
    
    # Calculate difference
    original_array = np.array(original_resized) / 255.0
    diff = np.abs(original_array - rgb)
    mean_diff = diff.mean()
    print(f"\n📈 Mean absolute difference: {mean_diff:.6f}")
    print(f"   (Lower is better. Should be < 0.05 for good reconstruction)")
    
else:
    print(f"\n⚠️ Original image not found at {original_path}")
    print("   Skipping comparison...")

print(f"\n✅ Done! Check the output directory: {OUTPUT_DIR.resolve()}")