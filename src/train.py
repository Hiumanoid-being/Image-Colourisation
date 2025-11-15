import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn_transformer_colorizer_v1 import CNNTransformerColorizer

# ============================================================
# Dataset
# ============================================================
class ColorizationDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.gray_dir = Path(data_dir).resolve() / split / "grayscale"
        self.color_dir = Path(data_dir).resolve() / split / "color"
        self.gray_files = list(self.gray_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.gray_files)

    def __getitem__(self, idx):
        gray_path = self.gray_files[idx]
        color_path = self.color_dir / gray_path.with_suffix(".npy").name

        # Load L and AB
        L = np.array(Image.open(gray_path)) / 255.0
        AB = np.load(color_path)
        
        # Sanity check: AB should be in [-1, 1] from preprocessing
        assert AB.min() >= -1.0 and AB.max() <= 1.0, \
            f"AB channels out of range: [{AB.min():.3f}, {AB.max():.3f}]. Expected [-1, 1]"

        # Convert to tensors
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        AB = torch.tensor(AB, dtype=torch.float32).permute(2, 0, 1)  # [2, H, W]
        return L, AB


# ============================================================
# Training Function
# ============================================================
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """
    Train the CNN-Transformer model for image colourisation.
    Args:
        model: The model instance (CNN + Transformer).
        train_loader: DataLoader for training set.
        criterion: Loss function (e.g. L1Loss).
        optimizer: Optimizer (e.g. Adam).
        device: 'cpu' or 'cuda'.
        epochs: Number of epochs.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (L, AB) in pbar:
            L, AB = L.to(device), AB.to(device)

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

                # Resize output to match target if necessary
                if output.shape != AB.shape:
                    print(f"⚠️ Shape mismatch detected. Resizing output from {output.shape} → {AB.shape}")
                    output = torch.nn.functional.interpolate(output, size=(AB.shape[2], AB.shape[3]), mode='bilinear', align_corners=False)

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

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.6f}\n")

    print("🎯 Training completed successfully!")
    return model


# ============================================================
# Visualization Function
# ============================================================
def visualize_results(model, dataset, device, num_samples=3):
    model.eval()
    idxs = np.random.choice(len(dataset), num_samples, replace=False)
    plt.figure(figsize=(9, 3 * num_samples))
    for i, idx in enumerate(idxs):
        L, AB = dataset[idx]
        with torch.no_grad():
            pred_AB = model(L.unsqueeze(0).to(device)).cpu().squeeze(0)
        pred_AB = pred_AB.permute(1, 2, 0).numpy()

        # Reconstruct LAB → RGB
        L_np = L.squeeze().numpy() * 255
        AB_np = pred_AB * 128 + 128
        LAB = np.zeros((L_np.shape[0], L_np.shape[1], 3))
        LAB[:, :, 0] = L_np
        LAB[:, :, 1:] = AB_np
        rgb_img = Image.fromarray(LAB.astype(np.uint8), mode="LAB").convert("RGB")

        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(L.squeeze(), cmap="gray")
        plt.title("Input L")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(rgb_img)
        plt.title("Predicted Color")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 3)
        AB_gt = AB.permute(1, 2, 0).numpy()
        LAB_gt = np.zeros_like(LAB)
        LAB_gt[:, :, 0] = L_np
        LAB_gt[:, :, 1:] = AB_gt * 128 + 128
        rgb_gt = Image.fromarray(LAB_gt.astype(np.uint8), mode="LAB").convert("RGB")
        plt.imshow(rgb_gt)
        plt.title("Ground Truth")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "../data/processed"
    train_dataset = ColorizationDataset(data_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0) # set num_workers=0 for Windows

    model = CNNTransformerColorizer().to(device)
    criterion = nn.L1Loss()  # Changed from MSELoss to L1Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Increased learning rate

    model = train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # Save model
    Path("../models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "../models/cnn_transformer_colorizer.pth")

    # Visualize some predictions
    visualize_results(model, train_dataset, device)


if __name__ == "__main__":
    main()