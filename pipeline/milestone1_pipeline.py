# pipeline/milestone1_pipeline_FIXED.py

import os
import cv2
import numpy as np
from pathlib import Path

# Assuming these are defined in your config
from config import DATASET_ROOT, OUTPUTS_ROOT, PUZZLE_FOLDERS
from io_utils.file_utils import ensure_dir
from io_utils.save_utils import save_intermediate_images, prepare_piece_dirs, save_crop
from preprocessing.denoise import denoise_image
from preprocessing.enhancement import enhance_image
from preprocessing.thresholding import binarize_and_clean
from segmentation.splitter import split_grid

# Point to the source of truth
GROUND_TRUTH_DIR = Path(DATASET_ROOT) / "correct"


def run_milestone1():
    ensure_dir(OUTPUTS_ROOT)

    # Validate Ground Truth exists
    if not GROUND_TRUTH_DIR.exists():
        print(f"[ERROR] Ground Truth directory not found: {GROUND_TRUTH_DIR}")
        return

    # List valid ground truth images (0.png, 1.png, etc.)
    valid_exts = {".png", ".jpg", ".jpeg"}
    gt_images = sorted([
        f for f in GROUND_TRUTH_DIR.iterdir()
        if f.suffix.lower() in valid_exts and f.stem.isdigit()
    ], key=lambda x: int(x.stem))

    print(f"[INFO] Found {len(gt_images)} Ground Truth images. Generating puzzles...")

    for folder in PUZZLE_FOLDERS:
        # Define output path
        out_folder = os.path.join(OUTPUTS_ROOT, "Gravity_Falls", folder)
        ensure_dir(out_folder)

        # Determine grid size from folder name
        if "2x2" in folder:
            grid = 2
        elif "4x4" in folder:
            grid = 4
        elif "8x8" in folder:
            grid = 8
        else:
            print(f"[WARN] Skipping unknown folder pattern: {folder}")
            continue

        print(f"\n[INFO] Generating {folder} (Grid: {grid}x{grid}) from Ground Truth...")

        for img_path in gt_images:
            # Use the ID from the Ground Truth filename (e.g., "3" from "3.png")
            img_name = img_path.stem

            # -------------------
            # Load Ground Truth
            # -------------------
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[ERROR] Cannot read image: {img_path}")
                continue

            # OPTIONAL: Handle Aspect Ratio Here
            # If you want square puzzles from rectangular images, crop the center now.
            h, w = bgr.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            bgr = bgr[start_y:start_y + min_dim, start_x:start_x + min_dim]

            # Ensure contiguous array
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # -------------------
            # Preprocessing
            # -------------------
            den = denoise_image(gray)
            enh = enhance_image(den)
            binary = binarize_and_clean(enh)

            # Save intermediate images (optional)
            save_intermediate_images(out_folder, img_name, gray, den, enh, binary)

            # -------------------
            # Split Grid
            # -------------------
            tiles_orig = split_grid(bgr, grid)
            tiles_enh = split_grid(enh, grid)

            # -------------------
            # Save Pieces
            # -------------------
            dirs = prepare_piece_dirs(out_folder, img_name)

            for piece_id in range(len(tiles_orig)):
                tile_orig = np.ascontiguousarray(tiles_orig[piece_id], dtype=np.uint8)
                tile_enh = np.ascontiguousarray(tiles_enh[piece_id], dtype=np.uint8)

                save_crop(dirs["original"], piece_id, tile_orig)
                save_crop(dirs["enhanced"], piece_id, tile_enh)

            print(f"   -> Processed Puzzle {img_name} successfully.")

    print("\n[DONE] Milestone 1 Pipeline finished with CORRECT IDs.")