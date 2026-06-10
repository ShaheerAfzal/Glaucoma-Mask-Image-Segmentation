# Glaucoma Optic Nerve Head Segmentation

**Automated segmentation of the Optic Cup and Optic Disc in retinal fundus images using classical Digital Image Processing - no deep learning required.**

Evaluated on the [Drishti-GS benchmark dataset](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) using Dice Similarity Coefficients across both Training and Test phases.

---

## Table of Contents

- [Background](#background)
- [How It Works](#how-it-works)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Background

Glaucoma is a leading cause of irreversible blindness, primarily caused by increased intraocular pressure that damages the **Optic Nerve Head (ONH)**. Clinical screening relies on measuring the **Cup-to-Disc Ratio (CDR)** - the ratio of the Optic Cup diameter to the Optic Disc diameter. A high CDR is a strong indicator of glaucomatous damage.

This project builds an end-to-end image processing pipeline to automatically segment both structures from retinal fundus photographs, enabling automated CDR estimation without any neural networks or pre-trained models.

---

## How It Works

The pipeline has three main stages:

### 1. Image Enhancement

Raw fundus images are challenging due to uneven illumination and low contrast around the optic nerve head. The enhancement stage:

- **Extracts the green channel** - the green channel carries the highest contrast for retinal structures compared to red or blue.
- **Applies a circular FOV mask** - a circular mask (48% of the image's shorter dimension) isolates the actual retinal field-of-view and discards the dark border artifacts.
- **Performs percentile-based contrast stretching** - instead of using the absolute min/max pixel values (which are easily skewed by outliers), the 2nd and 98th percentiles are used as the stretch boundaries, mapping the useful intensity range to [0, 255].
- **Applies histogram equalization** - further redistributes pixel intensities for maximum contrast.
- **Applies Gaussian blur** (11×11 kernel) - smooths local noise before thresholding.

### 2. Optic Disc Segmentation

- A **90th percentile threshold** is applied to the enhanced image. The Optic Disc is the brightest region in the fundus, so taking the top 10% of pixel intensities isolates it well.
- A custom **two-pass 8-connected Component Labeling (CCL)** algorithm groups connected bright regions into labelled blobs - implemented from scratch without `cv2.connectedComponents`.
- The **largest blob** is selected as the Optic Disc candidate.
- **Contour filling** (`cv2.drawContours` with `cv2.FILLED`) closes any holes inside the disc mask, producing a clean, solid region.

### 3. Optic Cup Segmentation

- The cup is searched **strictly within the disc mask**, which constrains the search space and avoids false detections elsewhere in the image.
- A **60th percentile threshold** is applied to disc-region pixels only - the cup is the brightest sub-region inside the disc, and the lower threshold captures it reliably.
- CCL and largest-blob selection are applied again within this constrained region.

---

## Pipeline Overview

```
Fundus Image (.png)
        │
        ▼
 ┌─────────────────────────────────────────┐
 │  1. Enhancement                         │
 │     • Extract green channel             │
 │     • Circular FOV mask                 │
 │     • Percentile contrast stretch (2/98)│
 │     • Histogram equalization            │
 │     • Gaussian blur (11×11)             │
 └───────────────────┬─────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 ┌─────────────┐           ┌─────────────┐
 │  2. Disc    │           │  3. Cup     │
 │  90th pct   │──mask──▶  │  60th pct   │
 │  threshold  │           │  (inside    │
 │  + CCL      │           │   disc only)│
 │  + fill     │           │  + CCL      │
 └─────────────┘           └─────────────┘
        │                         │
        └────────────┬────────────┘
                     ▼
          Dice Score vs. Ground Truth
          Saved masks + comparison plots
```

---

## Project Structure

```
Glaucoma-Mask-Image-Segmentation/
│
├── dhrishti-code.py          # Main pipeline script
│
├── Drishti-GS/               # Dataset directory (not included - see setup below)
│   ├── Training/
│   │   └── Training/
│   │       ├── Images/
│   │       └── Training_GT/
│   └── Test/
│       └── Test/
│           ├── Images/
│           └── Test_GT/
│
├── Results/                  # Auto-generated outputs
│   ├── Training/
│   │   ├── Masks/            # Predicted disc & cup binary masks
│   │   └── Comparisons/      # 4-panel comparison plots
│   ├── Test/
│   │   ├── Masks/
│   │   └── Comparisons/
│   ├── Training_scores.csv   # Per-image Dice scores (Training)
│   └── Test_scores.csv       # Per-image Dice scores (Test)
│
└── README.md
```

---

## Dataset Setup

This project uses the **Drishti-GS dataset**, a publicly available benchmark for glaucoma screening.

1. Download the dataset from the [official source](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) or [IEEE DataPort](https://ieee-dataport.org/open-access/drishti-gs-retinal-image-dataset-optic-nerve-head-segmentation).
2. Extract it so the folder structure matches `Drishti-GS/Training/Training/Images/` and `Drishti-GS/Test/Test/Images/` as shown above.
3. Place the `Drishti-GS/` folder in the same directory as `dhrishti-code.py`.

The ground-truth soft maps should follow the naming convention:
```
Drishti-GS/<phase>/<phase>/<phase>_GT/<image_name>/SoftMap/<image_name>_cupsegSoftmap.png
Drishti-GS/<phase>/<phase>/<phase>_GT/<image_name>/SoftMap/<image_name>_ODsegSoftmap.png
```

---

## Installation

Requires Python 3.7+. Install dependencies with:

```bash
pip install opencv-python numpy pandas matplotlib
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python
numpy
pandas
matplotlib
```

No GPU or deep learning frameworks required.

---

## Usage

Make sure the `Drishti-GS/` dataset folder is in the same directory as the script, then run:

```bash
python dhrishti-code.py
```

The script automatically runs both **Training** and **Test** phases in sequence, printing per-image Dice scores to the console and saving all results to the `Results/` folder.

**Expected console output:**
```
========================================
--- Starting Training Phase ---
========================================
Starting run for 50 images in Training...

Image: drishtiImg001  | Disc Dice: 0.8921 | Cup Dice: 0.7234
Image: drishtiImg002  | Disc Dice: 0.9102 | Cup Dice: 0.7561
...
------------------------------
Avg Disc Dice (Training): 0.XXXX
Avg Cup Dice (Training):  0.XXXX
------------------------------

========================================
--- Starting Test Phase ---
========================================
...
Finished! Check the Results folder for your Training and Test data.
```

---

## Output

For each image, the pipeline produces:

**Binary Masks** (saved to `Results/<phase>/Masks/`):
- `<image_name>_disc.png` - predicted Optic Disc mask
- `<image_name>_cup.png` - predicted Optic Cup mask

**Comparison Plots** (saved to `Results/<phase>/Comparisons/`):
- `<image_name>_plot.png` - a 4-panel figure showing:
  1. Original fundus image
  2. Ground truth Optic Disc mask
  3. Predicted Optic Disc mask
  4. Predicted Optic Cup mask

**Score CSVs** (saved to `Results/`):
- `Training_scores.csv` - columns: `Name`, `Disc_Dice`, `Cup_Dice`
- `Test_scores.csv` - same structure for the test split

---

## Results

Performance is measured using the **Dice Similarity Coefficient (DSC)**:

$$\text{DSC} = \frac{2 \cdot |P \cap G|}{|P| + |G|}$$

where *P* is the predicted mask and *G* is the ground truth. A DSC of 1.0 is a perfect overlap; 0.0 is no overlap.

> Results will vary depending on dataset version and preprocessing. Run the script to generate your own scores, which will be saved to `Results/Training_scores.csv` and `Results/Test_scores.csv`.

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| Green channel extraction | Provides highest contrast for the ONH vs. surrounding retinal tissue |
| Percentile stretch (2nd / 98th) | Robust to outlier pixels at image borders; avoids washing out the useful range |
| 90th percentile for Disc | The Optic Disc is the brightest region; top 10% captures it without noise |
| 60th percentile for Cup | The Cup is only slightly brighter than the surrounding disc tissue; a lower threshold is needed to capture it |
| Custom CCL (8-connected) | Implemented from scratch to demonstrate the algorithm; avoids `cv2.connectedComponents` |
| Contour fill on Disc | The disc often has small dark vessel shadows inside; filling the outer contour produces a clean mask |
| Cup constrained to Disc region | Prevents false cup detections in other bright regions of the image |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.5 | Image I/O, masking, contour operations |
| `numpy` | ≥ 1.21 | Array operations, CCL implementation |
| `pandas` | ≥ 1.3 | Dice score aggregation and CSV export |
| `matplotlib` | ≥ 3.4 | Comparison plot generation |

---

## Author

**Muhammad Shaheer Afzal**  
B.E. Computer Engineering, NUST EME  
[GitHub](https://github.com/ShaheerAfzal) · [LinkedIn](https://www.linkedin.com/in/muhammad-shaheer-afzal-939aa220a)
