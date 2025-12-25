import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageEnhance
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
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if RGB (take mean of channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2)
    
    # Normalize to [-1, 1]
    L = image.astype("float32")
    L = (L / 128) - 1.0
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        pred_AB = model(L_tensor)
    
    # Apply saturation boost
    pred_AB_boosted = torch.clamp(pred_AB * saturation_boost, -1.0, 1.0)
    
    # Convert to numpy
    L_np = L_tensor.squeeze().cpu().numpy()
    pred_AB_boosted_np = np.transpose(pred_AB_boosted.squeeze().cpu().numpy(), (1, 2, 0))
    
    # Convert back to CIELAB ranges
    L_lab = L_np * 100.0
    AB_SCALE = 128.0
    pred_AB_lab = pred_AB_boosted_np * AB_SCALE
    pred_AB_lab = np.clip(pred_AB_lab, -127.0, 127.0)
    
    # Reconstruct LAB image
    lab_out = np.empty((L_lab.shape[0], L_lab.shape[1], 3), dtype=np.float64)
    lab_out[:, :, 0] = L_lab
    lab_out[:, :, 1:] = pred_AB_lab
    
    # Convert LAB → RGB
    rgb_out_f = color.lab2rgb(lab_out)
    rgb_out_f = np.clip(rgb_out_f, 0.0, 1.0)
    rgb_out = (rgb_out_f * 255.0).round().astype(np.uint8)
    
    # Convert to PIL Image for additional saturation enhancement
    img_colorized = Image.fromarray(rgb_out)
    enhancer = ImageEnhance.Color(img_colorized)
    img_final = enhancer.enhance(1.0) 
    
    return img_final

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
                value=1.5,
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
