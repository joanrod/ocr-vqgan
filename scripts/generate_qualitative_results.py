# File: generate_qualitative_results.py
# Created by Juan A. Rodriguez on 18/06/2022
# Goal: Access generated images from different models 
# and randomly pick samples and analize qualitative results.

import argparse
import os
import random 
import datetime
import shutil

# Args for the 4 models that we evaluate
parser = argparse.ArgumentParser()
parser.add_argument("--num_images", type=int, default=1, help="Number of random samples to extract")
parser.add_argument("--DALLE", type=str, default = None, help="Path to directory containing DALLE generated images (using pretrained DALLE VQVAE)")
parser.add_argument("--VQGAN_pretrained", default = None, type=str,help="Path to directory containing VQGAN_pretrained generated samples (using pretrained VQGAN on imagenet)")
parser.add_argument("--VQGAN_finetuned", default = None, type=str, help="Path to directory containing VQGAN finetuned with Paper2Fig100k generated images")
parser.add_argument("--OCR_VQGAN", type=str, default = None, help="Path to directory containing OCR-VQGAN finetuned with Paper2Fig100k generated images")
parser.add_argument("--test_dataset", type=str, required=True, help="Path to directory containing images of test (original input images)")
args = parser.parse_args()

test_dataset = args.test_dataset

# Note: This script must be executed one the evaluate script has been executed. Images should be in the "evaluate directory"
models_to_evaluate = {
    "DALLE": args.DALLE if args.DALLE else None ,
    "VQGAN_pretrained": args.VQGAN_pretrained if args.VQGAN_pretrained else None ,
    "VQGAN_finetuned": args.VQGAN_finetuned if args.VQGAN_finetuned else None ,
    "OCR_VQGAN":args.OCR_VQGAN if args.OCR_VQGAN else None,
    "Input":test_dataset
}

# Obtain number total images
total_images = len(os.listdir(test_dataset)) 
 
# Extract num_images random generated samples
rand_indeces = random.sample(range(0, total_images), k=args.num_images)

# Create new directory to store comparison of samples
out_dir = os.path.join('output', f'model_comparison_{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}')
if not os.path.exists(out_dir): os.makedirs(out_dir)

for key in models_to_evaluate:
    if models_to_evaluate[key]:
        model_name = key
        path_to_reconstructed_images = models_to_evaluate[key]
        list_paths_samples = os.listdir(path_to_reconstructed_images)
    
        # index those samples and store them in a folder
        out_dir_model = os.path.join(out_dir, model_name)
        if not os.path.exists(out_dir_model): os.makedirs(out_dir_model)
        count = 0
        for sample in rand_indeces:
            format_image = list_paths_samples[sample].split('.')[1]
            path_sample = os.path.join(path_to_reconstructed_images, list_paths_samples[sample])
            # Parse values and mat
            path_out = os.path.join(out_dir_model, f"sample_{count}_id_{sample}.{format_image}")
            shutil.copy(path_sample, path_out)
            count+=1



        