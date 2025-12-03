import os

OUTPUTS_ROOT = "outputs/Gravity_Falls"


def count_png(path):
    """Count PNG files in a folder."""
    if not os.path.exists(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.lower().endswith(".png"))


def verify_folder(folder, grid):
    print(f"\n[CHECK] {folder} (grid = {grid})")

    base_path = os.path.join(OUTPUTS_ROOT, folder)
    intermediate = os.path.join(base_path, "intermediate")
    pieces_root = os.path.join(base_path, "pieces")

    # -----------------------------
    # Check intermediate images
    # -----------------------------
    if not os.path.exists(intermediate):
        print("  [ERROR] Intermediate folder missing!")
    else:
        interm_count = count_png(intermediate)
        print(f"  Intermediate images: {interm_count}")

    # -----------------------------
    # Check pieces
    # -----------------------------
    if not os.path.exists(pieces_root):
        print("  [ERROR] Pieces folder missing!")
        return

    expected_tiles = grid * grid

    for img_folder in sorted(os.listdir(pieces_root)):
        img_path = os.path.join(pieces_root, img_folder)

        orig_dir = os.path.join(img_path, "original")
        enh_dir = os.path.join(img_path, "enhanced")

        # Validate original crops
        if not os.path.exists(orig_dir):
            print(f"  [ERROR] {img_folder}: missing original/")
        else:
            orig_count = count_png(orig_dir)
            status = "[OK]" if orig_count == expected_tiles else "[ERROR]"
            print(f"  {status} {img_folder} original: {orig_count} tiles")

        # Validate enhanced crops
        if not os.path.exists(enh_dir):
            print(f"  [ERROR] {img_folder}: missing enhanced/")
        else:
            enh_count = count_png(enh_dir)
            status = "[OK]" if enh_count == expected_tiles else "[ERROR]"
            print(f"  {status} {img_folder} enhanced: {enh_count} tiles")


if __name__ == "__main__":
    verify_folder("puzzle_2x2", 2)
    verify_folder("puzzle_4x4", 4)
    verify_folder("puzzle_8x8", 8)

    print("\n[DONE] Output verification completed.")