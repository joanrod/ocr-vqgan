# File: compute_ocr_perceptual_loss.py
# Created by Juan A. Rodriguez on 18/06/2022
# Goal: Script to compute OCR perceptual loss from a pair of images (input and reconstruction)

import argparse
from taming.modules.losses.lpips import OCR_CRAFT_LPIPS
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="Path to input image")
parser.add_argument("--recons_path", type=str, required=True, help="Path to reconstructed image")
args = parser.parse_args()

def get_image_tensor(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)
    return torch.unsqueeze(T.ToTensor()(image), 0)

if __name__ == "__main__":
    # Load image and reconstruction to tensors
    input_path = args.input_path
    recons_path = args.recons_path

    input_tensor = get_image_tensor(input_path).cuda()
    rec_tensor = get_image_tensor(recons_path).cuda()

    OCR_perceptual_loss = OCR_CRAFT_LPIPS().eval()
    OCR_perceptual_loss.cuda()

    ocr_sim = OCR_perceptual_loss(input_tensor, rec_tensor)
