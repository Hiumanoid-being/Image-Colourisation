import torch
import numpy as np
import argparse
from PIL import Image
import cv2
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import model architecture
from CNN_Transformer_Model import CNNTransformerColourizer


def load_model(checkpoint_path, device='cuda'):
    """Load the trained model from checkpoint."""
    # Auto-detect device if cuda requested but not available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model = CNNTransformerColourizer()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def preprocess_image(image_path, target_size=256):
    """Load and preprocess image for the model."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # Convert RGB to LAB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # Resize
    lab = cv2.resize(lab, (target_size, target_size))
    
    # Extract L channel and normalize to [-1, 1]
    L = lab[:, :, 0]
    L = (L / 50.0) - 1.0  # Normalize L channel
    
    # Convert to tensor
    L_tensor = torch.from_numpy(L).float().unsqueeze(0).unsqueeze(0)
    
    return L_tensor, lab, img_np.shape[:2]


def postprocess_output(L, ab_pred, original_size):
    """Convert model output back to RGB image."""
    # Convert tensors to numpy
    L_np = L.squeeze().cpu().numpy()
    ab_np = ab_pred.squeeze().cpu().numpy()
    
    # Denormalize L
    L_np = (L_np + 1.0) * 50.0
    
    # Denormalize AB (assuming tanh output, so already in [-1, 1])
    ab_np = ab_np * 110.0  # Scale to LAB range
    
    # Combine L and AB channels
    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3))
    lab[:, :, 0] = L_np
    lab[:, :, 1] = ab_np[0]
    lab[:, :, 2] = ab_np[1]
    
    # Convert LAB to RGB
    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize back to original size if needed
    if rgb.shape[:2] != original_size:
        rgb = cv2.resize(rgb, (original_size[1], original_size[0]))
    
    return rgb


def colourize_image(model, image_path, device='cuda', output_path=None):
    """Colourize a single image."""
    # Preprocess
    L_tensor, lab, original_size = preprocess_image(image_path)
    L_tensor = L_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        ab_pred = model(L_tensor)
    
    # Postprocess
    colourized = postprocess_output(L_tensor, ab_pred, original_size)
    
    # Save or return
    if output_path:
        Image.fromarray(colourized).save(output_path)
        print(f"Colourized image saved to {output_path}")
    
    return colourized


def batch_colourize(model, input_dir, output_dir, device='cuda'):
    """Colourize all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))
    
    print(f"Found {len(image_files)} images to colourize")
    
    for img_file in image_files:
        output_file = output_path / f"colourized_{img_file.name}"
        print(f"Processing {img_file.name}...")
        try:
            colourize_image(model, str(img_file), device, str(output_file))
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Colourize grayscale images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='colourized.png',
                        help='Path to save colourized image (or output directory for batch mode)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in input directory')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for inference (auto will use CUDA if available)')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size for processing')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            device = 'cpu'
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Colourize
    if args.batch:
        batch_colourize(model, args.input, args.output, device)
    else:
        colourize_image(model, args.input, device, args.output)


if __name__ == '__main__':
    main()