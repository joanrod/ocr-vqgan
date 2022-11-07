# File: parse_paper2fig1_img_to_VQGAN.py
# Created by Juan A. Rodriguez on 12/06/2022
# Goal: This script is intended to access the json files corresponding to the paper2fig dataset (train, val)
# and convert them to the format required by the VQ-GAN, that is, a txt file containing the image path,

import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Path to dataset root, containing json files and figures directory")
args = parser.parse_args()

if __name__ == '__main__':
    path = args.path
    splits = ['train', 'test']
    count = 0
    for split in splits:
        with open(os.path.join(path, f'paper2fig_{split}.json')) as f:
            data = json.load(f)
        for item in tqdm(data):
            path_img = os.path.join(path, 'figures', f'{item["figure_id"]}.png')
            # append to txt file
            with open(path + '/paper2fig1_img_'+split+'.txt', 'a') as f:
                f.write(path_img + '\n')
            count += 1
    print(f"Stored {count} images in paper2fig1_img_{split}.txt")
