import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNet
from tqdm import tqdm

parser = argparse.ArgumentParser("UNet")
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of UNet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image that used to predict the model")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store prediction results")
args = parser.parse_args()


# Configuration
image_folder = args.test_image_path
save_folder = args.save_path
model_path = args.checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create save folder if not exists
os.makedirs(save_folder, exist_ok=True)

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transformation (corrected comma + structure)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Inference
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

with torch.no_grad():
    for filename in tqdm(image_files, desc="Predicting"):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

        output = model(input_tensor)               # raw logits
        output = torch.sigmoid(output)             # [1, 1, H, W]
        output = output[:, 0, :, :]                # [1, H, W]

        #prob_map = output.squeeze().cpu().numpy()  # values in range [0, 1]
        #print(f"{filename} — Min: {prob_map.min():.4f}, Max: {prob_map.max():.4f}, Mean: {prob_map.mean():.4f}")

        binary_mask = (output > 0.6).float()       # threshold to binary

        # Convert tensor to image
        mask = binary_mask.squeeze().cpu().numpy() * 255  # [H, W]
        mask = mask.astype(np.uint8)
        mask_image = Image.fromarray(mask, mode='L')

        # Save prediction
        save_path = os.path.join(save_folder, filename)
        mask_image.save(save_path)

print(f"Predictions saved to '{save_folder}'")
