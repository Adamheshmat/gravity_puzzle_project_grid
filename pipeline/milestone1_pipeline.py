# pipeline/milestone1_pipeline.py

import os
import cv2
import numpy as np

from config import DATASET_ROOT, OUTPUTS_ROOT, PUZZLE_FOLDERS
from io_utils.file_utils import ensure_dir, list_images
from io_utils.save_utils import (
    save_intermediate_images,
    prepare_piece_dirs,
    save_crop
)
from preprocessing.denoise import denoise_image
from preprocessing.enhancement import enhance_image
from preprocessing.thresholding import binarize_and_clean
from segmentation.splitter import split_grid


def run_milestone1():
    ensure_dir(OUTPUTS_ROOT)

    for folder in PUZZLE_FOLDERS:
        in_folder = os.path.join(DATASET_ROOT, folder)
        out_folder = os.path.join(OUTPUTS_ROOT, "Gravity_Falls", folder)
        ensure_dir(out_folder)

        print(f"[INFO] Processing folder: {in_folder}")
        images = list_images(in_folder)

        if not images:
            print(f"[WARN] No images found in {in_folder}")
            continue

        if "2x2" in folder:
            grid = 2
        elif "4x4" in folder:
            grid = 4
        elif "8x8" in folder:
            grid = 8
        else:
            raise ValueError(f"Unknown folder {folder}")

        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[INFO] Processing image: {img_name}")

            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"[ERROR] Cannot read image: {img_path}")
                continue

            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            den = denoise_image(gray)
            enh = enhance_image(den)
            binary = binarize_and_clean(enh)

            save_intermediate_images(out_folder, img_name, gray, den, enh, binary)

            tiles = split_grid(bgr, grid)
            crop_dir = prepare_piece_dirs(out_folder, img_name)

            for piece_id, crop in enumerate(tiles):
                crop = np.ascontiguousarray(crop, dtype=np.uint8)
                save_crop(crop_dir, piece_id, crop)

    print("[DONE] Simplified Milestone 1 pipeline finished.")
