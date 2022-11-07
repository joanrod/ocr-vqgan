# File: parse_ICDAR2013_img_to_VQGAN.py
# Created by Juan A. Rodriguez on 12/06/2022
# Goal: This script is intended to access the json files corresponding to the ICDAR13 dataset (train, val)
# and convert them to the format required by the VQGAN, that is, a txt file containing the image path,

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Path to dataset root, containing image directories (train and test)")
args = parser.parse_args()

if __name__ == '__main__':
    splits = ['train', 'test']
    count = 0
    for split in splits:
        if split == "train":
            split_dir_name = "Challenge2_Training_Task12_Images"
        else:
            split_dir_name = "Challenge2_Test_Task12_Images"
        path = os.path.join(args.path, split_dir_name)
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                path_img = os.path.join(path, filename)
                # append to txt file
                with open(os.path.join(path, 'ICDAR_2013_img_'+split+'.txt'), 'a') as f:
                    f.write(path_img + '\n')
                count += 1
    print(f"Stored {count} images in paper2fig1_img_{split}.txt")
