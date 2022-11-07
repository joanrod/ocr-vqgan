# File: prepare_eval_samples.py
# Created by Juan A. Rodriguez on 23/06/2022
# Goal: Util script to process the samples in the test set and prepare them for evaluation

from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import argparse
from taming.data.custom import CustomTest
import torch

# Args for the 4 models that we evaluate
parser = argparse.ArgumentParser()
parser.add_argument("--image_txt_path", type=str, required=True, help="Path to directory containing images")
parser.add_argument("--store_path", type=str, required=True, help="Path to directory containing images")
parser.add_argument("--size", default=384, help="Image size")
args = parser.parse_args()

if __name__ == "__main__":
    
    # Create folder for output
    dataset = os.path.split(args.image_txt_path)[-1].split(".")[0]
    outpath = os.path.join(args.store_path, dataset)
    os.makedirs(outpath, exist_ok=True)
   
    ds = CustomTest(384, args.image_txt_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for i, sample in enumerate(dl):
        image = sample['image']
        if len(image.shape) == 3:
            image = image[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        image = image.detach().cpu()
        image = torch.clamp(image, -1., 1.) 
        image = (image+1.0)/2.0 # -1,1 -> 0,1; c,h,w
        image = image.transpose(1, 2).transpose(2, 3)
        image = image.numpy()
        image = (image*255).astype(np.uint8)

        filename = f"input_{i}.png"
        path = os.path.join(outpath, filename)
        Image.fromarray(image[0]).save(path)
