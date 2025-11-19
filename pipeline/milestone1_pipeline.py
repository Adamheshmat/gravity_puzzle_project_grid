# pipeline/milestone1_pipeline.py

import os
import cv2
import numpy as np

from config import DATASET_ROOT, OUTPUTS_ROOT, PUZZLE_FOLDERS
from io_utils.file_utils import ensure_dir, list_images
from io_utils.save_utils import (
    save_intermediate_images,
    prepare_piece_dirs,
    save_piece_outputs,
    save_descriptor_json,
)
from preprocessing.denoise import denoise_image
from preprocessing.enhancement import enhance_image
from preprocessing.thresholding import binarize_and_clean
from segmentation.splitter import split_grid
from descriptors.fourier import fourier_descriptor


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

        # choose grid size based on folder
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

            # convert to contiguous array for Pylance OpenCV compatibility
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            den = denoise_image(gray)
            enh = enhance_image(den)
            binary = binarize_and_clean(enh)

            save_intermediate_images(out_folder, img_name, gray, den, enh, binary)


            tiles = split_grid(bgr, grid)
            dirs = prepare_piece_dirs(out_folder, img_name)

            for piece_id, crop in enumerate(tiles):
                crop = np.ascontiguousarray(crop, dtype=np.uint8)

                h, w = crop.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8) * 255

                contour_points = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ], dtype=np.int32)

                crop_path, mask_path, cnt_path = save_piece_outputs(
                    dirs, piece_id, crop, mask, contour_points
                )

                fd = fourier_descriptor(contour_points, n=32)

                data = {
                    "image": img_name,
                    "piece_id": piece_id,
                    "descriptor": fd.tolist(),
                    "contour": contour_points.tolist(),
                    "crop_file": os.path.basename(crop_path),
                    "mask_file": os.path.basename(mask_path)
                }

                save_descriptor_json(dirs, piece_id, data)

    print("[DONE] Milestone 1 pipeline finished.")
