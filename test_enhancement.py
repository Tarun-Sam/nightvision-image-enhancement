"""
Test script for NIGHTVISION enhancement module (image_processing/enhancement.py).

This script:
1. Loads sample images from the LOL (Low-Light) dataset (sample_images/eval15/low/).
2. Tests all four enhancement methods individually:
   - Global Histogram Equalization
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gamma Correction (Power-Law)
   - Linear Contrast Stretching (Min-Max)
3. Verifies that output dimensions and data types are strictly preserved.
4. Validates edge cases and error handling:
   - Invalid gamma values (<= 0, non-numeric, booleans)
   - Invalid CLAHE parameters (clip_limit <= 0, invalid tile_grid_size)
   - Empty image arrays
   - Non-2D images (3-channel arrays) and non-numpy inputs
5. Generates side-by-side comparison images with visual labels and saves all artifacts
   in output/enhancement_tests/.
"""

import os
import cv2
import numpy as np

from image_processing.enhancement import (
    apply_histogram_equalization,
    apply_clahe,
    apply_gamma_correction,
    apply_contrast_stretching,
)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "enhancement_tests")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images", "eval15", "low")


def add_label_header(image: np.ndarray, label: str) -> np.ndarray:
    """
    Adds a clean labeled header bar on top of an image panel for side-by-side display.
    """
    h, w = image.shape
    header_height = 40
    labeled = np.zeros((h + header_height, w), dtype=np.uint8)
    # Header background
    labeled[0:header_height, :] = 25
    # White label text
    cv2.putText(
        labeled,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        255,
        2,
        cv2.LINE_AA,
    )
    # Border line separating header from image
    labeled[header_height - 1 : header_height, :] = 100
    # Copy image
    labeled[header_height:, :] = image
    return labeled


def test_enhancement_module():
    """
    Main test runner for the enhancement module.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" NIGHTVISION: Enhancement Module Verification & Testing")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Discover & Load Sample Images from the LOL dataset
    # -------------------------------------------------------------
    print("\n[Step 1] Loading sample images from LOL dataset...")
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg"))]
    if not sample_files:
        raise FileNotFoundError(f"No sample images found in {SAMPLE_DIR}")

    sample_files.sort()
    # Select primary test sample (e.g., '1.png' and optionally '22.png')
    primary_filename = sample_files[0]
    primary_path = os.path.join(SAMPLE_DIR, primary_filename)
    low_light_img = cv2.imread(primary_path, cv2.IMREAD_GRAYSCALE)

    if low_light_img is None:
        raise ValueError(f"Failed to load image from {primary_path}")

    print(f" -> Selected sample: {primary_filename}")
    print(f" -> Dimensions: {low_light_img.shape} (Height: {low_light_img.shape[0]}, Width: {low_light_img.shape[1]})")
    print(f" -> Data Type: {low_light_img.dtype}")
    print(f" -> Intensity Range: [{low_light_img.min()}, {low_light_img.max()}] (Mean: {low_light_img.mean():.2f})")

    # Save original grayscale image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"01_original_{primary_filename}"), low_light_img)

    # -------------------------------------------------------------
    # Step 2: Test All Four Enhancement Methods Individually
    # -------------------------------------------------------------
    print("\n[Step 2] Testing individual enhancement algorithms...")

    enhancement_methods = {
        "histogram_equalization": {
            "name": "Histogram Equalization (Global HE)",
            "func": lambda img: apply_histogram_equalization(img),
        },
        "clahe": {
            "name": "CLAHE (clip=2.0, grid=8x8)",
            "func": lambda img: apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)),
        },
        "gamma_correction": {
            "name": "Gamma Correction (gamma=0.5)",
            "func": lambda img: apply_gamma_correction(img, gamma=0.5),
        },
        "contrast_stretching": {
            "name": "Contrast Stretching (Min-Max)",
            "func": lambda img: apply_contrast_stretching(img),
        },
    }

    results = {}
    all_tests_passed = True
    check_count = 0

    for key, info in enhancement_methods.items():
        check_count += 1
        enhanced = info["func"](low_light_img)
        results[key] = enhanced

        # Check dimension preservation
        shape_ok = enhanced.shape == low_light_img.shape
        # Check data type preservation
        dtype_ok = enhanced.dtype == low_light_img.dtype

        status = "PASS" if (shape_ok and dtype_ok) else "FAIL"
        if not (shape_ok and dtype_ok):
            all_tests_passed = False

        # Save individual result image
        out_filename = f"02_enhanced_{key}_{primary_filename}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_filename), enhanced)

        print(
            f"  [{status}] {info['name']:<35} -> "
            f"Shape: {enhanced.shape}, Dtype: {enhanced.dtype}, "
            f"Range: [{enhanced.min()}, {enhanced.max()}], Mean: {enhanced.mean():.2f}"
        )

    # -------------------------------------------------------------
    # Step 3: Input Validation and Error Handling Tests
    # -------------------------------------------------------------
    print("\n[Step 3] Verifying input validation and edge cases...")

    empty_2d_img = np.zeros((0, 0), dtype=np.uint8)
    empty_1d_img = np.array([], dtype=np.uint8)
    three_channel_img = np.zeros((100, 100, 3), dtype=np.uint8)
    float_img = np.zeros((100, 100), dtype=np.float32)

    negative_test_cases = [
        # Empty images
        ("Empty 2D array on HE", lambda: apply_histogram_equalization(empty_2d_img), ValueError),
        ("Empty 1D array on CLAHE", lambda: apply_clahe(empty_1d_img), ValueError),
        ("Empty array on Gamma", lambda: apply_gamma_correction(empty_2d_img, 0.5), ValueError),
        ("Empty array on Contrast Stretch", lambda: apply_contrast_stretching(empty_2d_img), ValueError),

        # Invalid gamma values
        ("Gamma <= 0 (gamma = 0)", lambda: apply_gamma_correction(low_light_img, 0), ValueError),
        ("Gamma negative (gamma = -0.5)", lambda: apply_gamma_correction(low_light_img, -0.5), ValueError),
        ("Gamma non-numeric (string)", lambda: apply_gamma_correction(low_light_img, "0.5"), TypeError),
        ("Gamma boolean (gamma = True)", lambda: apply_gamma_correction(low_light_img, True), TypeError),

        # Invalid CLAHE parameters
        ("CLAHE clip_limit = 0", lambda: apply_clahe(low_light_img, clip_limit=0), ValueError),
        ("CLAHE negative clip_limit (-1)", lambda: apply_clahe(low_light_img, clip_limit=-1), ValueError),
        ("CLAHE boolean clip_limit", lambda: apply_clahe(low_light_img, clip_limit=True), TypeError),
        ("CLAHE string clip_limit", lambda: apply_clahe(low_light_img, clip_limit="2.0"), TypeError),
        ("CLAHE invalid grid size tuple (8,)", lambda: apply_clahe(low_light_img, tile_grid_size=(8,)), ValueError),
        ("CLAHE negative grid size (-8, 8)", lambda: apply_clahe(low_light_img, tile_grid_size=(-8, 8)), ValueError),
        ("CLAHE zero grid size (0, 8)", lambda: apply_clahe(low_light_img, tile_grid_size=(0, 8)), ValueError),

        # Non-2D / Wrong types
        ("3-channel image on HE", lambda: apply_histogram_equalization(three_channel_img), ValueError),
        ("Float32 image on CLAHE", lambda: apply_clahe(float_img), TypeError),
        ("Non-numpy input on Stretch", lambda: apply_contrast_stretching([[10, 20], [30, 40]]), TypeError),
    ]

    for label, test_call, expected_error in negative_test_cases:
        check_count += 1
        try:
            test_call()
            print(f"  [FAIL] {label}: Expected {expected_error.__name__}, but no exception was raised.")
            all_tests_passed = False
        except expected_error as err:
            print(f"  [PASS] {label}: Successfully caught {expected_error.__name__}")
        except Exception as unexpected:
            print(f"  [FAIL] {label}: Caught unexpected exception {type(unexpected).__name__}: {unexpected}")
            all_tests_passed = False

    # -------------------------------------------------------------
    # Step 4: Generate Side-by-Side Visual Comparisons
    # -------------------------------------------------------------
    print("\n[Step 4] Generating side-by-side comparison images...")

    labeled_original = add_label_header(low_light_img, "Original (Low-Light)")

    # 1. Individual pairs: Original vs Each Method
    for key, info in enhancement_methods.items():
        labeled_enhanced = add_label_header(results[key], info["name"])
        side_by_side = np.hstack([labeled_original, labeled_enhanced])
        pair_filename = f"comparison_original_vs_{key}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, pair_filename), side_by_side)
        print(f"  -> Saved pairwise comparison: {pair_filename}")

    # 2. Comprehensive 5-panel montage:
    # Resize slightly for clean multi-column viewing (scale by 0.5 if image is large)
    scale = 0.6
    orig_resized = cv2.resize(low_light_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    panel_orig = add_label_header(orig_resized, "Original Low-Light")
    panel_he = add_label_header(
        cv2.resize(results["histogram_equalization"], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        "Global Hist Eq",
    )
    panel_clahe = add_label_header(
        cv2.resize(results["clahe"], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        "CLAHE (k=2.0)",
    )
    panel_gamma = add_label_header(
        cv2.resize(results["gamma_correction"], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        "Gamma (0.5)",
    )
    panel_stretch = add_label_header(
        cv2.resize(results["contrast_stretching"], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        "Min-Max Stretch",
    )

    full_montage = np.hstack([panel_orig, panel_he, panel_clahe, panel_gamma, panel_stretch])
    montage_path = os.path.join(OUTPUT_DIR, "comparison_all_methods_montage.png")
    cv2.imwrite(montage_path, full_montage)
    print(f"  -> Saved full 5-panel montage: comparison_all_methods_montage.png")

    print("\n" + "=" * 70)
    if all_tests_passed:
        print(f" ALL {check_count} ENHANCEMENT TESTS & VALIDATIONS PASSED!")
        print(f" Artifacts saved to: {os.path.abspath(OUTPUT_DIR)}")
    else:
        print(" SOME CHECKS FAILED. Review errors above.")
    print("=" * 70)

    return all_tests_passed


if __name__ == "__main__":
    success = test_enhancement_module()
    exit(0 if success else 1)
