# io_utils/save_utils.py

import os
import cv2
from .file_utils import ensure_dir


def save_intermediate_images(out_root, img_name, gray, den, enh, binary):
    inter_dir = os.path.join(out_root, "intermediate")
    ensure_dir(inter_dir)

    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_denoised.png"), den)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_enhanced.png"), enh)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_binary.png"), binary)


def prepare_piece_dirs(out_root, img_name):
    """
    Creates:
        pieces/<img_name>/original/
        pieces/<img_name>/enhanced/
    Returns dictionary with both folder paths.
    """

    base_dir = os.path.join(out_root, "pieces", img_name)
    orig_dir = os.path.join(base_dir, "original")
    enh_dir = os.path.join(base_dir, "enhanced")

    ensure_dir(orig_dir)
    ensure_dir(enh_dir)

    return {
        "original": orig_dir,
        "enhanced": enh_dir
    }


def save_crop(dir_path, piece_id, crop):
    filename = f"piece_{piece_id:03}.png"
    full_path = os.path.join(dir_path, filename)
    cv2.imwrite(full_path, crop)
    return full_path