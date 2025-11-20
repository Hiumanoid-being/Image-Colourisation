# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from cnn_transformer_colorizer_v1 import CNNTransformerColorizer 

# ============================================================
# Dataset
# ============================================================
class ColorizationDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.gray_dir = Path(data_dir) / split / "grayscale"
        self.color_dir = Path(data_dir) / split / "color"
        self.gray_files = list(self.gray_dir.glob("*.jpg"))
        
        print(f"📁 Dataset: {len(self.gray_files)} images found in {self.gray_dir}")

    def __len__(self):
        return len(self.gray_files)

    def __getitem__(self, idx):
        gray_path = self.gray_files[idx]
        color_path = self.color_dir / gray_path.with_suffix(".npy").name

        # Load L and AB (both already normalized to [-1, 1])
        L_img = np.array(Image.open(gray_path))
        L = (L_img / 127.5) - 1.0  # [0, 255] -> [-1, 1]
        AB = np.load(color_path)   # Already [-1, 1]
        
        # Sanity check: Both L and AB should be in [-1, 1]
        assert L.min() >= -1.0 and L.max() <= 1.0, \
            f"L channel out of range: [{L.min():.3f}, {L.max():.3f}]. Expected [-1, 1]"
        assert AB.min() >= -1.0 and AB.max() <= 1.0, \
            f"AB channels out of range: [{AB.min():.3f}, {AB.max():.3f}]. Expected [-1, 1]"

        # Convert to tensors
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        AB = torch.tensor(AB, dtype=torch.float32).permute(2, 0, 1)  # [2, H, W]
        return L, AB


# ============================================================
# Enhanced Training Function with Multi-GPU Support
# ============================================================
def train_model_multi_gpu(model, train_loader, criterion, optimizer, device, epochs=5):
    """
    Enhanced training function with GPU monitoring and multi-GPU support
    """
    model.train()
    
    # GPU monitoring function
    def print_gpu_usage():
        if torch.cuda.is_available():
            print("GPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: {alloc:.2f}GB / {cached:.2f}GB")

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (L, AB) in pbar:
            L, AB = L.to(device, non_blocking=True), AB.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            output = model(L)

            # --- Sanity check (only for first batch of first epoch)
            if epoch == 0 and i == 0:
                print("\n=== Sanity check for shapes ===")
                print(f"Input L shape: {L.shape}")
                print(f"Output (predicted AB) shape: {output.shape}")
                print(f"Target AB shape: {AB.shape}")
                print(f"Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
                print(f"Target min/max: {AB.min().item():.4f} / {AB.max().item():.4f}")

            # --- Resize for safety on every batch
            if output.shape != AB.shape:
                output = torch.nn.functional.interpolate(output, size=(AB.shape[2], AB.shape[3]), mode='bilinear', align_corners=False)

            # Compute loss
            loss = criterion(output, AB)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            # Print GPU usage first batch of first epoch
            if epoch == 0 and i == 0:
                print_gpu_usage()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.6f}")

    print("Training completed successfully!")
    return model


# ============================================================
# Visualization Function
# ============================================================
def visualize_results(model, dataset, device, num_samples=3):
    """
    Visualize model predictions by converting LAB back to RGB.
    All values are in [-1, 1] range.
    """
    from skimage import color
    
    model.eval()
    idxs = np.random.choice(len(dataset), num_samples, replace=False)
    plt.figure(figsize=(9, 3 * num_samples))
    
    for i, idx in enumerate(idxs):
        L, AB = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            pred_AB = model(L.unsqueeze(0).to(device)).cpu().squeeze(0)
        
        # Convert from tensors to numpy
        L_np = L.squeeze().numpy()  # [-1, 1]
        pred_AB_np = pred_AB.permute(1, 2, 0).numpy()  # [-1, 1]
        AB_gt_np = AB.permute(1, 2, 0).numpy()  # [-1, 1]
        
        # Convert back to CIELAB range
        L_lab = (L_np + 1.0) * 50.0  # [-1, 1] -> [0, 100]
        pred_AB_lab = pred_AB_np * 110.0  # [-1, 1] -> [-110, 110]
        AB_gt_lab = AB_gt_np * 110.0  # [-1, 1] -> [-110, 110]
        
        # Reconstruct LAB images
        pred_lab = np.zeros((L_lab.shape[0], L_lab.shape[1], 3))
        pred_lab[:, :, 0] = L_lab
        pred_lab[:, :, 1:] = pred_AB_lab
        
        gt_lab = np.zeros_like(pred_lab)
        gt_lab[:, :, 0] = L_lab
        gt_lab[:, :, 1:] = AB_gt_lab
        
        # Convert LAB to RGB using skimage
        pred_rgb = color.lab2rgb(pred_lab)
        gt_rgb = color.lab2rgb(gt_lab)
        
        # Plot
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(L_np, cmap="gray", vmin=-1, vmax=1)
        plt.title("Input L")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(np.clip(pred_rgb, 0, 1))
        plt.title("Predicted Color")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(np.clip(gt_rgb, 0, 1))
        plt.title("Ground Truth")
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()


# ============================================================
# Auto-detect Dataset Path
# ============================================================
def find_dataset_path():
    """Automatically find the dataset path in Kaggle"""
    input_dir = Path("/kaggle/input")
    
    if not input_dir.exists():
        print("/kaggle/input directory not found")
        return None
    
    available_datasets = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Available datasets: {[d.name for d in available_datasets]}")
    
    # Look for processed data
    for dataset in available_datasets:
        potential_path = dataset / "processed"
        if potential_path.exists():
            print(f"Found processed data at: {potential_path}")
            return str(potential_path)
    
    # Look for train/grayscale structure directly
    for dataset in available_datasets:
        potential_train = dataset / "train" / "grayscale"
        if potential_train.exists():
            print(f"Found data at: {dataset}")
            return str(dataset)
    
    # Use the first dataset as fallback
    if available_datasets:
        fallback = available_datasets[0]
        print(f"Using fallback dataset: {fallback}")
        return str(fallback)
    
    print("No datasets found")
    return None


# ============================================================
# Enhanced Main Function with Multi-GPU Support
# ============================================================
def main():
    # GPU configuration
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    
    # Use all available GPUs
    if num_gpus > 1:
        device = torch.device("cuda:0")  # Use first GPU as main
        print(f"Using {num_gpus} GPUs with DataParallel")
        
        # Model with DataParallel
        model = CNNTransformerColorizer()
        model = DataParallel(model)
        model = model.to(device)
        
        # Larger batch size for multiple GPUs
        batch_size = 8 * num_gpus
        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNTransformerColorizer().to(device)
        batch_size = 8
        print(f"Using device: {device}")

    # Auto-detect dataset path for Kaggle
    data_dir = find_dataset_path()
    if data_dir is None:
        print("Could not find dataset. Please check your Kaggle dataset.")
        return
    
    print(f"Using data directory: {data_dir}")
    
    try:
        train_dataset = ColorizationDataset(data_dir, split="train")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True  # Faster data transfer to GPU
        )
        print(f"Batch size: {batch_size} (adjusted for {num_gpus} GPU(s))")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Available directories:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files) - 5} more files')
        return

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Starting training with multi-GPU support...")
    model = train_model_multi_gpu(model, train_loader, criterion, optimizer, device, epochs=100)

    # Save model (handle DataParallel wrapping)
    if num_gpus > 1:
        model_to_save = model.module  # Get the original model from DataParallel
    else:
        model_to_save = model
        
    model_save_path = "/kaggle/working/cnn_transformer_colorizer_multi_gpu.pth"
    torch.save(model_to_save.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Visualize some predictions
    print("Generating visualizations...")
    visualize_results(model_to_save, train_dataset, device)


if __name__ == "__main__":
    main()