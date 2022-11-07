# File: plot_ocr_features.py
# Created by Juan A. Rodriguez on 27/6/2022
# Goal: Util script to plot OCR features from images. 
# It is itended to input a pair of images, (original and reconstruction/decoded), and will store ocr deep features for qualitative analysis

import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
from taming.modules.losses.craft import CRAFT
from taming.modules.util import copyStateDict
import argparse

# Args for the 4 models that we evaluate
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="Path to input image")
parser.add_argument("--recons_path", type=str, required=True, help="Path to reconstructed image")
parser.add_argument("--out", type=str, default='output', help="output path")
args = parser.parse_args()

def get_image_tensor(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)

    return torch.unsqueeze(T.ToTensor()(image), 0)

def save_features(tensor, path):
    # get some feature maps
    feats = tensor[2][4][:,:12].squeeze(0)
    
    feats = feats.detach().cpu()
    feats = torch.clamp(feats, -1., 1.) 
    feats = (feats+1.0)/2.0 # -1,1 -> 0,1; c,h,w
    feats = feats.reshape(-1, 3, 192, 192)
    feats = feats.transpose(1, 2).transpose(2, 3)
    feats = feats.numpy()
    feats = (feats*255).astype(np.uint8)

    for k in range(feats.shape[0]):
        filename = f"feature_{k}.png"
        im = feats[k]
        Image.fromarray(im).save(os.path.join(path, filename))

if __name__ == '__main__':
    # Load image and reconstruction to tensors
    input_path = args.input_path
    recons_path = args.recons_path

    input_tensor = get_image_tensor(input_path)
    rec_tensor = get_image_tensor(recons_path)
    model_path = ''
    craft = CRAFT(pretrained=True, freeze=True, amp=False)
    param = torch.load(model_path)
    print("Loading craft model from {}".format(model_path))
    craft.load_state_dict(copyStateDict(param))

    in_feats = craft(input_tensor)
    out_feats = craft(rec_tensor)
    path_store_in = ''
    path_store_rec = ''

    save_features(in_feats, path_store_in)
    save_features(out_feats, path_store_rec)