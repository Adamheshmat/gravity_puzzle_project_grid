import os
import cv2

OUTPUTS_ROOT = "outputs/Gravity_Falls"

def count_files(path, ext=None):
    if not os.path.exists(path):
        return 0
    if ext:
        return sum(1 for f in os.listdir(path) if f.endswith(ext))
    return len(os.listdir(path))


def verify_folder(folder, grid):
    print(f"\n[CHECK] {folder} (grid = {grid})")

    a_path = os.path.join(OUTPUTS_ROOT, folder)
    intermediate = os.path.join(a_path, "intermediate")
    pieces_root = os.path.join(a_path, "pieces")

    # Check intermediate images
    if not os.path.exists(intermediate):
        print("  [ERROR] Intermediate folder missing!")
        return

    interm_count = count_files(intermediate, ".png")
    print(f"  Intermediate images: {interm_count}")

    # Check tiles inside each image folder
    for img_folder in sorted(os.listdir(pieces_root)):
        crops_path = os.path.join(pieces_root, img_folder, "crops")

        if not os.path.exists(crops_path):
            print(f"  [ERROR] Missing crops: {img_folder}")
            continue

        crop_count = count_files(crops_path, ".png")

        expected = grid * grid
        if crop_count != expected:
            print(f"  [ERROR] {img_folder}: found {crop_count}, expected {expected}")
        else:
            print(f"  [OK] {img_folder}: {crop_count} tiles")


if __name__ == "__main__":
    verify_folder("puzzle_2x2", 2)
    verify_folder("puzzle_4x4", 4)
    verify_folder("puzzle_8x8", 8)

    print("\n[DONE] Output verification completed.")
