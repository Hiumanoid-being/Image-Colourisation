import gradio as gr
import torch
import numpy as np
from PIL import Image
from skimage import color
from pathlib import Path
import sys

# Setup paths
base_dir = Path(__file__).resolve().parent
repo_root = base_dir.parent
sys.path.insert(0, str(repo_root))

from CNN_Transformer_Model import CNNTransformerColourizer

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformerColourizer().to(device)

MODEL_PATH = repo_root / "models" / "final.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

def colorize_image(image, saturation_boost):
    """
    Colorize a grayscale image following inference.ipynb exactly.
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale if RGB
    if image.mode == 'RGB':
        image = image.convert('L')
    
    # Convert to numpy and normalize to [-1, 1]
    L = np.array(image).astype("float32")
    L = (L / 128) - 1.0
    
    # Add batch dimension: [H, W] -> [1, H, W]
    L = torch.tensor(L).unsqueeze(0)
    
    # L already has shape [1, H, W], so just add channel dimension and move to device
    L_tensor = L.unsqueeze(0).to(device)  # Now shape is [1, 1, H, W]
    
    with torch.no_grad():
        pred_AB = model(L_tensor)
    
    # Apply saturation boost (in normalized tanh [-1,1] space) and clamp
    pred_AB_boosted = torch.clamp(pred_AB * saturation_boost, -1.0, 1.0)
    
    # Convert to numpy and reshape: (2, H, W) -> (H, W, 2)
    L_np = L.squeeze().cpu().numpy()  # Remove batch dim from L
    pred_AB_boosted_np = np.transpose(pred_AB_boosted.squeeze(0).cpu().numpy(), (1, 2, 0))
    
    # Convert back to CIELAB ranges
    # L in [-1,1] -> normalize to [0,100]
    L_lab = ((L_np + 1.0) / 2.0) * 100.0
    
    # AB predicted in [-1,1] -> approximate a/b scale
    AB_SCALE = 128.0
    pred_AB_boosted_lab = pred_AB_boosted_np * AB_SCALE
    
    # Clip AB to reasonable CIELAB-like range to avoid extreme values
    pred_AB_boosted_lab = np.clip(pred_AB_boosted_lab, -127.0, 127.0)
    
    # Reconstruct LAB image (use float dtype)
    lab_boosted = np.empty((L_lab.shape[0], L_lab.shape[1], 3), dtype=np.float64)
    lab_boosted[:, :, 0] = L_lab
    lab_boosted[:, :, 1:] = pred_AB_boosted_lab
    
    # Convert LAB → RGB, clamp to [0,1] then convert to uint8 safely
    rgb_boosted_f = color.lab2rgb(lab_boosted)
    rgb_boosted_f = np.clip(rgb_boosted_f, 0.0, 1.0)
    rgb_boosted = (rgb_boosted_f * 255.0).round().astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(rgb_boosted)

# Create Gradio interface
with gr.Blocks(title="Image Colourisation") as demo:
    gr.Markdown("# Image Colourisation with CNN-Transformer")
    gr.Markdown("Upload a grayscale image and adjust the saturation boost slider to colorize it. By Edison Chan")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                scale=1
            )
            saturation_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=1.2,
                step=0.1,
                label="Saturation Boost",
                info="1.0 = no boost, >1.0 = more saturated"
            )
            colorize_btn = gr.Button("Colourise!!!!", scale=1, variant="primary")
        
        with gr.Column():
            image_output = gr.Image(
                label="Colorized Output",
                type="pil"
            )
    
    # Connect button click to function
    colorize_btn.click(
        fn=colorize_image,
        inputs=[image_input, saturation_slider],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(share=True)