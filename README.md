# NIGHTVISION

**Image Denoising and Quality Enhancement for Low-Light Surveillance Applications Using Spatial Domain Filters**

## Overview
NIGHTVISION is an image processing application designed to enhance low-light surveillance imagery by reducing sensor noise and improving visual clarity using spatial domain filtering techniques and quality enhancement algorithms.

## Project Structure
```
NIGHTVISION/
│
├── app.py                      # Streamlit frontend application (to be implemented)
│
├── image_processing/
│   ├── __init__.py             # Module exports
│   ├── filters.py              # Spatial domain filtering implementations
│   ├── enhancement.py          # Contrast and brightness enhancement (future)
│   ├── pipeline.py             # Image processing pipeline (future)
│   └── evaluation.py           # Quantitative metrics: PSNR, SSIM, etc. (future)
│
├── sample_images/              # Input surveillance / low-light images
├── output/                     # Filtered and enhanced output images
├── test_filters.py             # Verification and test script for spatial filters
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup & Running Tests
1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the spatial filters test suite:
   ```bash
   python test_filters.py
   ```
