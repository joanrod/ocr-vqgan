# File: compute_ssim.py
# Created by Juan A. Rodriguez on 18/06/2022
# Goal: Util script to compute Structural Similarity Index (SSIM) 
# from two sets of images (original and reconstructuted) 
# You must pass both directory paths in input1 and input2

import argparse
import os
from PIL import Image
from SSIM_PIL import compare_ssim
from tqdm import tqdm
import multiprocessing as mp

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input1', type=str, required = True, help='path to directory 1 ')
parser.add_argument('--input2', type=str, required = True, help='path to directory 2')
args = parser.parse_args()

if __name__ == "__main__":
    MAX_WORKERS = mp.cpu_count()
    CHUNK_SIZE = 50

    print(f'Working with :{MAX_WORKERS} CPUs on Multiprocessing')

    files_input1 = os.listdir(args.input1)
    files_input2 = os.listdir(args.input2)

    pair_image_tuples = zip(files_input1, files_input2)

    def compute_ssim(pair):
        im_1 = Image.open(os.path.join(args.input1, pair[0]))
        im_2 = Image.open(os.path.join(args.input2, pair[1]))
        return compare_ssim(im_1, im_2)

    with mp.Pool(processes=MAX_WORKERS) as p:
        ssim_list = list(
            tqdm(p.imap(compute_ssim, list(pair_image_tuples), CHUNK_SIZE), total=len(files_input2)))

    ssim_score = sum(ssim_list) / len(ssim_list)

    print(f'SSIM score: {ssim_score}')
