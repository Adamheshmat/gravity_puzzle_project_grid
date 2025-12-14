# 🧩 Gravity Puzzle Reconstruction – Milestone 1 (Fall 2025)
### Image Preprocessing, Enhancement, and Grid-Based Segmentation (Option B)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)
![Status](https://img.shields.io/badge/Milestone-1%20Completed-brightgreen.svg)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey.svg)
![License](https://img.shields.io/badge/License-Academic%20Project-orange.svg)

This repository contains the complete implementation of **Milestone 1** for the Gravity Puzzle project.  
The objective of this milestone is to **preprocess puzzle images**, **enhance them**, and **perform grid-based segmentation** using known puzzle sizes (2×2, 4×4, 8×8).


---

## 📌 Overview of Milestone 1

Each puzzle image is passed through a simple but complete preprocessing pipeline:

1. Convert to **grayscale**  
2. Apply **noise reduction** (Gaussian blur)  
3. Apply **contrast enhancement** (CLAHE)  
4. Apply **Otsu binarization**  
5. Save all intermediate steps  
6. Perform **grid-based segmentation**  
7. Save:
   - Crops from the **original** image  
   - Crops from the **enhanced** image  

---

## 📁 Project Structure

```
gravity_puzzle_project/
│
├── datasets/
│   └── Gravity Falls/
│        ├── puzzle_2x2/
│        ├── puzzle_4x4/
│        └── puzzle_8x8/
│
├── pipeline/
│   └── milestone1_pipeline.py
│
├── preprocessing/
│   ├── denoise.py
│   ├── enhancement.py
│   └── thresholding.py
│
├── segmentation/
│   └── splitter.py
│
├── io_utils/
│   ├── file_utils.py
│   └── save_utils.py
│
├── outputs/
│   └── Gravity_Falls/
│        ├── puzzle_2x2/
│        ├── puzzle_4x4/
│        └── puzzle_8x8/
│            ├── intermediate/
│            └── pieces/
│                 └── <image_name>/
│                       ├── original/
│                       └── enhanced/
│
├── verify_outputs.py
└── main.py
```

---

## 🔧 How to Run

Run Milestone 1:

```
python3 main.py
```

Verify the number of output tiles:

```
python3 verify_outputs.py
```

---

## 🖼 Pipeline Diagram

```
┌────────────────────────────┐
│        Load Image          │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Convert to Grayscale       │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Noise Reduction            │
│ (Gaussian Blur)            │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Contrast Enhancement       │
│ (CLAHE)                    │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Binarization               │
│ (Otsu Threshold)           │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Save Intermediate Images   │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Grid Segmentation          │
│ (2×2 / 4×4 / 8×8)          │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Save Original Crops        │
│ Save Enhanced Crops        │
└────────────────────────────┘
```

---

## 📤 Output Description

### Intermediate images:
```
outputs/.../intermediate/
    <image>_gray.png
    <image>_denoised.png
    <image>_enhanced.png
    <image>_binary.png
```

### Cropped puzzle pieces:
```
outputs/.../pieces/<image_name>/
    ├── original/
    │      piece_000.png
    │      piece_001.png
    │      ...
    └── enhanced/
           piece_000.png
           piece_001.png
           ...
```

Tile counts:
- 2×2 → 4 tiles  
- 4×4 → 16 tiles  
- 8×8 → 64 tiles  

---

## 📌 Overview of Milestone 2

Using the segmented puzzle pieces generated in Milestone 1, the solver performs:

1. Load puzzle pieces and convert them to **LAB color space**
2. Compute **pairwise horizontal and vertical edge matching costs**
3. Apply **best-buddies (mutual nearest-neighbor) constraints**
4. Rank candidate **seed pieces**
5. Grow the puzzle using **priority-based placement**
6. Minimize **global seam energy**
7. Perform **post-assembly refinement** using swap-based optimization
8. Normalize the final **rotation**
9. Reconstruct the final image
10. Compute **quantitative accuracy metrics**

No machine learning or deep learning methods are used.

---

## 🧠 Core Ideas Used

- LAB color space for perceptual robustness  
- Edge seam comparison using color + texture gradients  
- Best-buddies constraint to suppress false matches  
- Priority queue growth instead of greedy placement  
- Multiple seed attempts to avoid local minima  
- Global seam energy as an optimization objective  
- Non-ground-truth refinement using swap-based hill climbing  

---

## 📁 Additional Structure for Milestone 2

```
pipeline/
└── solver.py        # Non-ground-truth puzzle solver (Milestone 2)
```

Solved outputs are written to:

```
outputs/Gravity_Falls/<puzzle_size>/pieces/<puzzle_id>/solved.png
```

---

## 🔧 How to Run Milestone 2

After running Milestone 1 and generating puzzle pieces:

```bash
python3 pipeline/solver.py
```

The solver will:
- Assemble each puzzle
- Save the reconstructed image
- Print per-puzzle and per-category accuracy statistics

---

## 🧩 Solver Pipeline (Milestone 2)

```
┌────────────────────────────┐
│ Load Puzzle Pieces         │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Convert to LAB Color Space │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Pairwise Edge Cost         │
│ (Color + Texture)          │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Best-Buddies Filtering     │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Seed Ranking               │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Priority-Based Growth      │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Global Refinement          │
│ (Swap Optimization)        │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Rotation Normalization     │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Image Reconstruction       │
│ + Accuracy Evaluation      │
└────────────────────────────┘
```

---

## 📊 Evaluation Method (Milestone 2)

Accuracy is computed by comparing predicted piece indices with their expected positions:

- **Accuracy (%) = Correct Placements / Total Pieces × 100**
- A puzzle is considered **perfectly solved** if accuracy ≥ **99.9%**

Reported metrics include:
- Per-puzzle accuracy
- Average accuracy per puzzle size
- Number of perfect reconstructions

---

## 📝 License

Academic use only — part of **CSE381 / Computer Vision – Fall 2025**.
