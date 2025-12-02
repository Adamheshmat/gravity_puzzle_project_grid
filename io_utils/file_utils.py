import os
import glob

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder, exts=('.png', '.jpg', '.jpeg', '.bmp')):
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(folder, '*' + e)))
    return sorted(imgs)