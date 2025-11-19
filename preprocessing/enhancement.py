import cv2
import numpy as np

def enhance_image(gray):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Ensure the result is a contiguous uint8 array (good for OpenCV)
    eq = np.ascontiguousarray(eq, dtype=np.uint8)

    return eq
