# Dataset Specifications and Preprocessing Guide

## Target Datasets

### 1. FER2013 (Facial Expression Recognition 2013)
- **Origin**: Kaggle FER2013 Challenge / ICML 2013 Workshop.
- **Image Characteristics**: 48x48 pixel grayscale facial crops.
- **Samples**: 35,887 images in total.
  - Training split: 28,709 images
  - Public validation split: 3,589 images
  - Private test split: 3,589 images
- **Classes**: 7 emotions (`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`).

### Class Distribution Challenges
- Highly imbalanced distribution: `happy` and `neutral` contain significantly more samples, while `disgust` contains fewer than 600 samples.
- Inherent label noise from automated web scraping in original benchmark dataset.

---

## Preprocessing Pipeline

```
Raw Frame / Image
      │
      ▼
Face Detection & Bounding Box Extraction
      │
      ▼
Grayscale Conversion (if 3-channel input)
      │
      ▼
Spatial Resizing (48 x 48 px)
      │
      ▼
Pixel Value Normalization (Scaling [0, 255] -> [0.0, 1.0] or Z-score Standardization)
      │
      ▼
Tensor Formatting (Batch, 48, 48, 1)
```

### Data Augmentation Strategy (Training only)
- Random horizontal flip ($p=0.5$).
- Random rotation ($\pm 10^\circ$).
- Random translation / shift ($\pm 10\%$).
- Random zoom range ($\pm 10\%$).
- Random shear range ($\pm 5\%$).

---

## Directory Organization

```
data/
├── raw/
│   ├── fer2013.csv          # Optional raw CSV format
│   ├── train/               # Train subdirectories organized by class
│   └── test/                # Test subdirectories organized by class
└── processed/
    ├── train_arrays.npz     # Preprocessed numpy tensors
    └── val_arrays.npz
```
*(All large data files are excluded from git tracking via `.gitignore`)*
