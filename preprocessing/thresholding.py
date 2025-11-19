import cv2
import numpy as np

def binarize_and_clean(gray):
    # Apply Otsu thresholding
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure puzzle tiles appear white, not black
    if (binary == 255).mean() < 0.5:
        binary = 255 - binary

    return binary
