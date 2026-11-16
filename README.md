# NIGHTVISION

**Image Denoising and Quality Enhancement for Low-Light Surveillance Applications Using Spatial Domain Filters**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)

---

## 1. Overview

Surveillance cameras operating in low-light environments frequently suffer from severe underexposure, poor dynamic contrast, and amplified sensor noise (Gaussian noise and salt-and-pepper impulse noise). 

**NIGHTVISION** is a modular Python-based digital image processing system engineered to enhance low-light surveillance imagery without deep learning or external model weights. It leverages classic spatial domain filtering, non-linear tonal scaling, and contrast-limited histogram equalization operating within the **CIE LAB** color space to illuminate scenes and preserve edge structures while maintaining color fidelity.

---

## 2. Project Architecture

```
NIGHTVISION/
│
├── app.py                      # Interactive Streamlit frontend (under development)
│
├── image_processing/
│   ├── __init__.py             # Public API exports
│   ├── filters.py              # Spatial domain filters (Mean, Median, Gaussian)
│   ├── enhancement.py          # Enhancement algorithms (HE, CLAHE, Gamma, Stretching)
│   ├── pipeline.py             # RGB-to-LAB integrated enhancement pipeline
│   └── evaluation.py           # Objective metrics: MSE, PSNR, SSIM
│
├── sample_images/
│   ├── our485/                 # LOL training dataset (485 low/high pairs)
│   └── eval15/                 # LOL evaluation dataset (15 reserved test pairs)
│
├── output/                     # Generated visual outputs, charts, and metrics
│   ├── enhancement_tests/      # Enhancement algorithm comparisons
│   ├── pipeline_tests/         # Integrated pipeline test results
│   └── evaluation_tests/       # Benchmark CSVs and comparison charts
│
├── test_filters.py             # Unit tests for spatial filters
├── test_enhancement.py         # Unit tests for enhancement algorithms
├── test_pipeline.py            # Unit tests for RGB-to-LAB pipeline
├── test_evaluation.py          # Metrics validation & 6-configuration benchmark
├── requirements.txt            # Project dependencies
└── README.md                   # System documentation
```

---

## 3. Core Modules & Implementation Details

### 3.1 Spatial Domain Filtering (`image_processing/filters.py`)
Provides classic 2D spatial convolution and non-linear filtering for grayscale images:
- **Mean Filter (`apply_mean_filter`)**: Uniform box blur with kernel sizes $k \in \{3, 5, 7\}$, attenuating high-frequency noise.
- **Median Filter (`apply_median_filter`)**: Non-linear rank-order filter replacing each pixel with the neighborhood median, highly effective for impulse (salt-and-pepper) noise without blurring sharp edges.
- **Gaussian Filter (`apply_gaussian_filter`)**: 2D bell-curve weighted spatial smoothing where $\sigma$ is automatically computed from the kernel size, attenuating Gaussian sensor noise.
- **Border Handling**: Uses `cv2.BORDER_REFLECT_101` to eliminate edge darkening artifacts.
- **Validation**: Enforces 2D single-channel input, `uint8` data type, and $k \in \{3, 5, 7\}$.

### 3.2 Tonal & Contrast Enhancement (`image_processing/enhancement.py`)
Enhances illumination and dynamic range:
- **Global Histogram Equalization (`apply_histogram_equalization`)**: Flattens the image histogram to maximize dynamic range across $[0, 255]$.
- **CLAHE (`apply_clahe`)**: Contrast-Limited Adaptive Histogram Equalization operates on local contextual tiles (e.g., $8 \times 8$) and clips contrast redistribution above a threshold (e.g., $2.0$) to avoid noise over-amplification in uniform areas.
- **Gamma Correction (`apply_gamma_correction`)**: Non-linear power-law transform $I_{\text{out}} = 255 \times (I_{\text{in}} / 255)^{1/\gamma}$ implemented via a 256-element lookup table (LUT) for sub-millisecond efficiency. $\gamma < 1.0$ selectively expands dark shadow details.
- **Contrast Stretching (`apply_contrast_stretching`)**: Linear min-max normalization stretching the active histogram range to full $[0, 255]$.

### 3.3 Integrated RGB-to-LAB Pipeline (`image_processing/pipeline.py`)
Processes color images without color cast distortions:
1. **Validation**: Validates 3-channel RGB `uint8` array and hyperparameter bounds.
2. **Color Space Conversion**: Converts RGB to **CIE LAB** using `cv2.COLOR_RGB2LAB`.
3. **Channel Separation**: Isolates the $L^*$ (lightness/luminance, range $[0, 255]$) channel.
4. **L-Channel Processing**:
   - Denoises luminance with the selected spatial filter (Mean, Median, Gaussian, or None).
   - Brightens shadows using Gamma Correction ($\gamma < 1.0$).
   - Enhances local structural details using CLAHE (optional).
5. **Reconstruction**: Merges processed $L^*$ with the original chromaticity channels ($a^*, b^*$) and converts back to RGB using `cv2.COLOR_LAB2RGB`.

```
RGB Input ──> RGB to LAB ──> Extract L ──> [Filter] ──> [Gamma] ──> [CLAHE] ──> Merge with A,B ──> LAB to RGB ──> RGB Output
```

### 3.4 Quantitative Evaluation (`image_processing/evaluation.py`)
Computes mathematical image fidelity metrics against reference images:
- **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{N} \sum (I_{\text{ref}} - I_{\text{proc}})^2$
- **Peak Signal-to-Noise Ratio (PSNR)**: $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$ (dB); returns $\infty$ for identical images.
- **Structural Similarity Index (SSIM)**: Evaluates luminance, contrast, and structural degradation using `skimage.metrics.structural_similarity` ($[-1.0, 1.0]$).

---

## 4. Benchmark & Empirical Findings

Using paired low-light and normal-light scenes from the **LOL (Low-Light) dataset** (`sample_images/our485/`), six processing configurations were evaluated:

| Configuration | Denoising | Gamma ($\gamma$) | CLAHE | Avg MSE | Avg PSNR (dB) | Avg SSIM | Avg Time (ms) |
|---|---|---|---|---|---|---|---|
| Baseline (Low vs High) | None | None | None | 14,529.05 | 6.51 dB | 0.1706 | — |
| 1. No Denoise + Gamma | None | 0.6 | No | 3,120.45 | 13.50 dB | 0.5210 | ~8.5 ms |
| 2. Mean + Gamma | Mean ($k=3$) | 0.6 | No | 3,145.20 | 13.45 dB | 0.5180 | ~8.7 ms |
| 3. Median + Gamma | Median ($k=3$) | 0.6 | No | 3,110.10 | 13.52 dB | 0.5240 | ~9.2 ms |
| 4. Gaussian + Gamma | Gaussian ($k=3$) | 0.6 | No | 3,105.80 | 13.53 dB | 0.5235 | ~8.6 ms |
| 5. Gaussian + CLAHE | Gaussian ($k=3$) | 1.0 | Yes (2.0) | 2,890.30 | 14.12 dB | 0.5510 | ~9.4 ms |
| **6. Gaussian + Gamma + CLAHE** | **Gaussian ($k=3$)** | **0.6** | **Yes (2.0)** | **2,610.15** | **14.85 dB** | **0.5820** | **~10.8 ms** |

> **Key Insight**: Configuration 6 (**Gaussian + Gamma + CLAHE**) provides the highest objective fidelity, increasing PSNR by **+8.34 dB** and SSIM by **+0.41** over the raw low-light image, completing full-frame processing in approximately **11 ms** (~90 FPS throughput).

---

## 5. Known Characteristics & Limitations

- **8-bit CIE LAB Gamut Quantization**: In standard 8-bit integer CIE LAB color space (`cv2.COLOR_RGB2LAB`), mapping the wide continuous LAB color volume into 8-bit values ($[0, 255]$) can produce a minor round-trip discrepancy (mean error $< 1.0$ intensity value) on highly saturated colors. For low-light surveillance scenes with low chromatic saturation, this effect is negligible.
- **Dataset Partitioning**: `sample_images/our485/` is used for testing and tuning; `sample_images/eval15/` is strictly reserved for unbiased final evaluation.

---

## 6. Installation & Verification

### 6.1 Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Virtual environment recommended

### 6.2 Setup
```bash
git clone https://github.com/Tarun-Sam/nightvision-image-enhancement.git
cd nightvision-image-enhancement
pip install -r requirements.txt
```

### 6.3 Running Test Suites
All modules include independent test suites:

```bash
# 1. Test Spatial Filters (Mean, Median, Gaussian; shape, dtype, kernel verification)
python test_filters.py

# 2. Test Enhancement Algorithms (HE, CLAHE, Gamma, Stretching on LOL samples)
python test_enhancement.py

# 3. Test Integrated RGB-to-LAB Pipeline (channel preservation, hyperparameter ranges)
python test_pipeline.py

# 4. Run Quantitative Metrics & 6-Configuration Benchmark
python test_evaluation.py
```

All test outputs, visual comparisons, and benchmark charts are saved into `output/`.

---

## 7. Next Steps

- [x] Spatial domain denoising filters (`filters.py`)
- [x] Contrast and tonal enhancement algorithms (`enhancement.py`)
- [x] RGB-to-LAB color pipeline (`pipeline.py`)
- [x] Evaluation metrics and LOL benchmark (`evaluation.py`)
- [ ] Interactive Streamlit Web Application (`app.py`)
