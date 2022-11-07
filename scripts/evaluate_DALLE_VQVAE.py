# File: compute_ssim.py
# Created by Juan A. Rodriguez on 18/06/2022
# Goal: Evaluate VQVAE model in the text-within-image reconstruction task. 
# Two datasets are used, Paper2Fig100k and ICDAR13

# Note: This script is no longer working, because DALLE-Pytorch is not compatble with torch==1.11.0
# TODO: Find a workaround to run VQVAEs in the project

from dalle_pytorch import OpenAIDiscreteVAE # pip install dalle-pytorch
from torch.utils.data import DataLoader, Dataset
from taming.modules.losses.lpips import LPIPS, OCR_CRAFT_LPIPS
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging import version

from PIL import Image
import numpy as np
import os
import torch
from tqdm import tqdm
import argparse

# Args for the 4 models that we evaluate
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing images")
parser.add_argument("--store_path", type=str, required=True, help="Path to directory containing images")
args = parser.parse_args()

# Add args, and add that it is all stored in VQGAN logs
class ImageDataset(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = False #random_crop
        
        with open(paths, "r") as f:
            self.data = f.read().splitlines()

        self.image_transform = A.Compose([
            A.SmallestMaxSize(max_size = self.size),
            # A.RandomCrop(height=image_size,width=image_size),
            A.CenterCrop(height=self.size,width=self.size),
            ToTensorV2()
        ])
        self._length = len(self.data)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.image_transform(image=image)["image"]
        return image

    def preprocess(self, img_path):
        img = Image.open(img_path)
        s = min(img.size)
        r = self.size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.size])
        img = T.ToTensor()(img)
        return img

    def __getitem__(self, i):
        sample = self.preprocess(self.data[i])
        return sample

if __name__ == "__main__":

    datasets = ["Paper2Fig100k", "ICDAR2013"]

    save_dir = args.store_path
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    B = 1
    IMAGE_SIZE = 384
    IMAGE_MODE = 'RGB'

    for d in datasets:
        name_experiment = 'openAI_VAE_' + d
        path_results = os.path.join(save_dir, name_experiment)
        if not os.path.exists(path_results): os.makedirs(path_results)

        if d == 'ICDAR2013':
            IMAGE_PATH = args.data_path + '/ICDAR2013/Challenge2_Test_Task12_Images/ICDAR_2013_img_test.txt'
        else:
            IMAGE_PATH = args.data_path + '/Paper2Fig100k/paper2fig1_img_test.txt'

        ds = ImageDataset(IMAGE_PATH, IMAGE_SIZE)
        dl = DataLoader(ds, batch_size = B, num_workers=1, shuffle=False)

        # Losses
        perceptual_loss = LPIPS().eval()
        perceptual_loss.cuda()
        ocr_perceptual_loss = OCR_CRAFT_LPIPS().eval().cuda()
        ocr_perceptual_loss.cuda()
        vae = OpenAIDiscreteVAE().cuda() 
        vae.eval()      
        print(f"dataset with {ds.__len__()} images")
        LPIPS_list = []
        OCR_list = []
        for i, images in tqdm(enumerate((dl))):
            images = images.cuda()
            image_tokens = vae.get_codebook_indices(images)
            rec_images = vae.decode(image_tokens)

            # Compute LPIPS
            LPIPS_list.append(perceptual_loss(images.contiguous(), rec_images.contiguous()).item())
            # Compute OCR SIM
            OCR_list.append(ocr_perceptual_loss(images.contiguous(), rec_images.contiguous()).item())

            # Store samples
            for k in range(rec_images.shape[0]):
                filename = f"reconstruction_batch_{i}_id_{k}.png"
                path = os.path.join(path_results, filename)
                x_rec = T.ToPILImage(mode='RGB')(rec_images[k]).save(path)
        
    print(f"LPIPS loss: {np.mean(LPIPS_list)}, OCR loss: {np.mean(OCR_list)}")
