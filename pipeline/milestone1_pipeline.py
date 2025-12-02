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

        # determine grid size
        if "2x2" in folder:
            grid = 2
        elif "4x4" in folder:
            grid = 4
        elif "8x8" in folder:
            grid = 8
        else:
            raise ValueError(f"Unknown folder name: {folder}")

        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[INFO] Processing image: {img_name}")

            # -------------------
            # Load image
            # -------------------
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"[ERROR] Cannot read image: {img_path}")
                continue

            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # -------------------
            # Preprocessing
            # -------------------
            den = denoise_image(gray)
            enh = enhance_image(den)
            binary = binarize_and_clean(enh)

            # save intermediate images
            save_intermediate_images(out_folder, img_name, gray, den, enh, binary)

            # -------------------
            # Split pieces (ORIGINAL)
            # -------------------
            tiles_orig = split_grid(bgr, grid)

            # split ENHANCED image same way
            tiles_enh = split_grid(enh, grid)

            # -------------------
            # Prepare directories
            # -------------------
            dirs = prepare_piece_dirs(out_folder, img_name)

            # -------------------
            # Save crops ORIGINAL + ENHANCED
            # -------------------
            for piece_id in range(len(tiles_orig)):
                tile_orig = np.ascontiguousarray(tiles_orig[piece_id], dtype=np.uint8)
                tile_enh = np.ascontiguousarray(tiles_enh[piece_id], dtype=np.uint8)

                save_crop(dirs["original"], piece_id, tile_orig)
                save_crop(dirs["enhanced"], piece_id, tile_enh)

    print("[DONE] Simplified Milestone 1 pipeline finished.")