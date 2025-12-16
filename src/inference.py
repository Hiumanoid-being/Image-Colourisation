import torch
import numpy as np
from PIL import Image
from pathlib import Path
import skimage.color as color
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from CNN_Transformer_Model import CNNTransformerColourizer


def load_model(checkpoint_path, device='cuda'):
    """Load trained colorization model."""
    model = CNNTransformerColourizer()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   Loss: {checkpoint['loss']:.4f}")
    
    return model


def check_image_compatibility(image_path, patch_size=4):
    """
    Check if image dimensions are compatible with the model.
    
    The model uses patch_size=4 in the transformer, so after CNN encoding
    (which downsamples by 4x), dimensions should be divisible by patch_size.
    
    Args:
        image_path: Path to image
        patch_size: Transformer patch size (default: 4)
    
    Returns:
        is_compatible: Boolean indicating if image can be processed as-is
        recommended_size: Recommended resize dimensions
    """
    img = Image.open(image_path)
    h, w = img.height, img.width
    
    # After CNN encoding (stride 4), dimensions are h/4 and w/4
    # These need to be divisible by patch_size
    encoded_h, encoded_w = h // 4, w // 4
    
    is_compatible = (encoded_h % patch_size == 0) and (encoded_w % patch_size == 0)
    
    if not is_compatible:
        # Find nearest compatible size (multiple of 16)
        recommended_h = ((h + 15) // 16) * 16
        recommended_w = ((w + 15) // 16) * 16
        recommended_size = (recommended_h, recommended_w)
    else:
        recommended_size = (h, w)
    
    return is_compatible, recommended_size, (h, w)


def print_image_info(image_path):
    """Print information about image compatibility."""
    is_compatible, recommended, original = check_image_compatibility(image_path)
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"  Original size: {original[0]}x{original[1]}")
    
    if is_compatible:
        print(f"  ✅ Compatible - can process at original size")
    else:
        print(f"  ⚠️  Not compatible - dimensions not divisible by 16")
        print(f"  💡 Recommended size: {recommended[0]}x{recommended[1]}")
    
    return is_compatible, recommended


def preprocess_grayscale(image_path, target_size=None):
    """
    Load and preprocess grayscale image for inference.
    
    Args:
        image_path: Path to grayscale .jpg image
        target_size: Optional (H, W) to resize image. If None, uses original size.
                    For best results, use multiples of 16 (e.g., 256, 512, 1024)
    
    Returns:
        L_tensor: Normalized L channel tensor [1, 1, H, W]
        L_original: Original L channel for reconstruction [H, W]
        original_size: Original image size (H, W) before any resizing
    """
    # Load grayscale image
    gray_img = Image.open(image_path).convert('L')
    original_size = (gray_img.height, gray_img.width)
    
    # Optionally resize
    if target_size is not None:
        gray_img = gray_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    gray_np = np.array(gray_img).astype('float32')
    
    # Convert from [0, 255] to [0, 100] (LAB L channel range)
    L_channel = (gray_np / 255.0) * 100.0
    
    # Normalize to [0, 1] for model input
    L_normalized = L_channel / 100.0
    
    # Convert to tensor [1, 1, H, W]
    L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).unsqueeze(0)
    
    return L_tensor, L_channel, original_size


def postprocess_to_rgb(L_channel, AB_pred):
    """
    Convert L channel and predicted AB channels back to RGB.
    
    Args:
        L_channel: Original L channel [H, W] in range [0, 100]
        AB_pred: Predicted AB channels [H, W, 2] in range [-1, 1]
    
    Returns:
        rgb_image: PIL RGB image
    """
    # Denormalize AB channels from [-1, 1] to approximately [-110, 110]
    # This reverses the normalization done in prepro.py
    AB_denormalized = AB_pred * 110.0
    
    # Clip to safe LAB range
    AB_denormalized = np.clip(AB_denormalized, -128, 127)
    
    # Combine L and AB channels
    lab_image = np.zeros((L_channel.shape[0], L_channel.shape[1], 3), dtype='float32')
    lab_image[..., 0] = L_channel  # L channel [0, 100]
    lab_image[..., 1:] = AB_denormalized  # AB channels [-128, 127]
    
    # Convert LAB to RGB using scikit-image
    rgb_image = color.lab2rgb(lab_image)
    
    # Convert to uint8 [0, 255]
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(rgb_image)


def colorize_image(model, image_path, device='cuda', target_size=None, resize_output=True):
    """
    Colorize a single grayscale image.
    
    Args:
        model: Trained CNNTransformerColourizer model
        image_path: Path to grayscale image
        device: Device to run inference on
        target_size: Optional (H, W) to resize input. Recommended: multiples of 16.
                    If None, uses original size (may fail if not divisible by 16)
        resize_output: If True and target_size was used, resize back to original size
    
    Returns:
        rgb_image: Colorized PIL RGB image
    """
    # Preprocess
    L_tensor, L_channel, original_size = preprocess_grayscale(image_path, target_size)
    L_tensor = L_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        AB_pred = model(L_tensor)  # [1, 2, H, W]
    
    # Convert to numpy
    AB_pred = AB_pred.cpu().squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 2]
    
    # Postprocess to RGB
    rgb_image = postprocess_to_rgb(L_channel, AB_pred)
    
    # Optionally resize back to original size
    if resize_output and target_size is not None and original_size != target_size:
        rgb_image = rgb_image.resize((original_size[1], original_size[0]), Image.LANCZOS)
    
    return rgb_image


def colorize_batch(model, image_paths, output_dir, device='cuda', target_size=None):
    """
    Colorize a batch of grayscale images and save results.
    
    Args:
        model: Trained model
        image_paths: List of paths to grayscale images
        output_dir: Directory to save colorized images
        device: Device to run inference on
        target_size: Optional (H, W) to resize inputs. Recommended: multiples of 16.
                    Output will be resized back to original dimensions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Colorizing {len(image_paths)} images...")
    if target_size:
        print(f"   Processing at {target_size[0]}x{target_size[1]}, then resizing to original")
    
    for i, img_path in enumerate(image_paths, 1):
        try:
            # Colorize
            rgb_image = colorize_image(model, img_path, device, target_size=target_size)
            
            # Save with same filename
            output_path = output_dir / Path(img_path).name
            rgb_image.save(output_path, quality=95)
            
            print(f"  [{i}/{len(image_paths)}] ✅ {img_path.name} -> {output_path.name}")
            
        except Exception as e:
            print(f"  [{i}/{len(image_paths)}] ❌ Error processing {img_path.name}: {e}")
    
    print(f"\n✅ Done! Results saved to {output_dir}")


def compare_side_by_side(original_gray_path, colorized_rgb_path, output_path):
    """
    Create a side-by-side comparison image.
    
    Args:
        original_gray_path: Path to original grayscale image
        colorized_rgb_path: Path to colorized RGB image
        output_path: Path to save comparison image
    """
    gray_img = Image.open(original_gray_path).convert('RGB')
    color_img = Image.open(colorized_rgb_path)
    
    # Create side-by-side image
    width, height = gray_img.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(gray_img, (0, 0))
    comparison.paste(color_img, (width, 0))
    
    comparison.save(output_path, quality=95)
    print(f"✅ Comparison saved to {output_path}")


def main():
    """Example usage of the inference pipeline."""
    
    # Configuration
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    INPUT_DIR = Path("data/processed/grayscale/val")
    OUTPUT_DIR = Path("results/colorized")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For non-standard image sizes, set to multiple of 16 (e.g., 256, 512, 1024)
    # Set to None to use original image size (works if images are already proper size)
    TARGET_SIZE = (256, 256)  # or None
    
    print(f"Device: {DEVICE}")
    print(f"Target processing size: {TARGET_SIZE}")
    
    # Load model
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Get test images (limit to first 10 for demo)
    image_paths = sorted(list(INPUT_DIR.rglob("*.jpg")))[:10]
    
    if not image_paths:
        print(f"❌ No images found in {INPUT_DIR}")
        return
    
    # Colorize batch
    colorize_batch(model, image_paths, OUTPUT_DIR, device=DEVICE, target_size=TARGET_SIZE)
    
    # Optional: Create comparison for first image
    if image_paths:
        comparison_path = OUTPUT_DIR / "comparison_example.jpg"
        compare_side_by_side(
            image_paths[0],
            OUTPUT_DIR / image_paths[0].name,
            comparison_path
        )


if __name__ == "__main__":
    main()