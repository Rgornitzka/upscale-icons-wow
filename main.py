from torchvision.transforms.functional import rgb_to_grayscale
import types
import sys

functional_tensor_mod = types.ModuleType("functional_tensor")
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault(
    "torchvision.transforms.functional_tensor", functional_tensor_mod
)

import os
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import multiprocessing as mp
from functools import partial
import torch
from typing import List, Tuple
import time


def init_upsampler(model_path: str):
    """Initialize a new upsampler instance"""
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    return upsampler


def process_image_chunk(
    chunk_data: Tuple[List[Tuple[str, str]], str, int, int, tuple],
) -> None:
    """Process a chunk of images with its own upsampler instance"""
    image_pairs, model_path, crop_amount, border_width, border_color = chunk_data

    # Initialize upsampler for this process
    upsampler = init_upsampler(model_path)

    for input_path, output_path in image_pairs:
        try:
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: Could not load image {input_path}")
                continue

            # Determine if image has alpha channel
            has_alpha = len(img.shape) == 3 and img.shape[2] == 4

            # Crop if specified
            if crop_amount > 0:
                img = img[crop_amount:-crop_amount, crop_amount:-crop_amount]

            # Upscale image
            output, _ = upsampler.enhance(img, outscale=4)

            # Add border if specified
            if border_width > 0:
                if has_alpha:
                    border_color_val = (*border_color[::-1], 255)
                else:
                    border_color_val = border_color[::-1]

                output = cv2.copyMakeBorder(
                    output,
                    border_width,
                    border_width,
                    border_width,
                    border_width,
                    cv2.BORDER_CONSTANT,
                    value=border_color_val,
                )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save upscaled image
            cv2.imwrite(output_path, output)
            print(f"Processed: {input_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")


def upscale_anime_images(
    input_folder: str,
    output_folder: str,
    model_path: str = None,
    crop_amount: int = 0,
    border_width: int = 0,
    border_color: tuple = (255, 255, 255),
    num_processes: int = None,
):
    """
    Upscale anime-style images using RealESRGAN_x4plus_anime_6B model with multiprocessing.
    """
    start_time = time.time()

    # Set number of processes if not specified
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max

    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True

    # Set up model path and download if needed
    if model_path is None:
        model_path = os.path.join("weights", "RealESRGAN_x4plus_anime_6B.pth")
        if not os.path.isfile(model_path):
            os.makedirs("weights", exist_ok=True)
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            model_path = load_file_from_url(
                url=model_url, model_dir="weights", progress=True
            )

    # Get all images from input folder
    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]

    # Prepare input/output pairs
    io_pairs = []
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        basename, ext = os.path.splitext(filename)

        # Check if output should be PNG
        output_ext = ".png" if ext.lower() not in [".jpg", ".jpeg"] else ext
        output_path = os.path.join(output_folder, f"{basename}{output_ext}")

        io_pairs.append((input_path, output_path))

    # Split work into chunks for each process
    total_images = len(io_pairs)
    chunk_size = (total_images + num_processes - 1) // num_processes
    chunks = [io_pairs[i : i + chunk_size] for i in range(0, total_images, chunk_size)]

    # Prepare chunks with all necessary data
    chunk_data = [
        (chunk, model_path, crop_amount, border_width, border_color) for chunk in chunks
    ]

    print(f"Processing {total_images} images using {num_processes} processes...")

    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        pool.map(process_image_chunk, chunk_data)

    end_time = time.time()
    print(f"Processed {total_images} images in {end_time - start_time:.2f} seconds")
    print(
        f"Average time per image: {(end_time - start_time) / total_images:.2f} seconds"
    )


if __name__ == "__main__":
    # Example usage
    upscale_anime_images(
        input_folder="input",
        output_folder="output",
        crop_amount=5,
        border_width=5,
        border_color=(0, 0, 0),
        num_processes=os.cpu_count(),  # Adjust based on your CPU cores
    )
