import os
import cv2
import json
import numpy as np
from .file_utils import ensure_dir


def save_intermediate_images(out_root, img_name, gray, den, enh, binary):
    inter_dir = os.path.join(out_root, "intermediate")
    ensure_dir(inter_dir)

    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_denoised.png"), den)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_enhanced.png"), enh)
    cv2.imwrite(os.path.join(inter_dir, f"{img_name}_binary.png"), binary)


def prepare_piece_dirs(out, img):
    root = os.path.join(out, "pieces", img)

    crops = os.path.join(root, "crops")
    masks = os.path.join(root, "masks")
    contours = os.path.join(root, "contours")
    descriptors = os.path.join(root, "descriptors")

    for d in (crops, masks, contours, descriptors):
        ensure_dir(d)

    return {
        "crops": crops,
        "masks": masks,
        "contours": contours,
        "descriptors": descriptors
    }


def save_piece_outputs(dirs, piece_id, crop, mask, contour_points):
    crop_path = os.path.join(dirs["crops"], f"piece_{piece_id:03}.png")
    mask_path = os.path.join(dirs["masks"], f"piece_{piece_id:03}.png")
    contour_path = os.path.join(dirs["contours"], f"piece_{piece_id:03}.npy")

    cv2.imwrite(crop_path, crop)
    cv2.imwrite(mask_path, mask)
    np.save(contour_path, contour_points)

    return crop_path, mask_path, contour_path


def save_descriptor_json(dirs, piece_id, data):
    json_path = os.path.join(dirs["descriptors"], f"piece_{piece_id:03}.json")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return json_path
