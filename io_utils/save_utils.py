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
    crop_dir = os.path.join(out_root, "pieces", img_name, "crops")
    ensure_dir(crop_dir)
    return crop_dir


def save_crop(crop_dir, piece_id, crop):
    crop_path = os.path.join(crop_dir, f"piece_{piece_id:03}.png")
    cv2.imwrite(crop_path, crop)
    return crop_path
