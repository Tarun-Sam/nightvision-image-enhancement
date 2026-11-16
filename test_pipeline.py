"""
Test suite for the NIGHTVISION integrated colour-image processing pipeline.

This script:
1. Loads 5 paired low-light and normal-light images from the LOL dataset (eval15).
2. Converts RGB colour images to CIE LAB, isolates and enhances the L channel,
   and reconstructs the output in RGB colour space.
3. Verifies output shapes, uint8 data types, and valid 3-channel colour dimensions.
4. Saves enhanced output images and side-by-side comparisons in output/pipeline_tests/.
5. Tests error handling for invalid dimensions, wrong data types, unsupported denoise methods,
   invalid kernel sizes, and invalid gamma values.
"""

import os
import cv2
import numpy as np

from image_processing.pipeline import enhance_image, ALLOWED_DENOISE_METHODS

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "pipeline_tests")
LOW_DIR = os.path.join(BASE_DIR, "sample_images", "eval15", "low")
HIGH_DIR = os.path.join(BASE_DIR, "sample_images", "eval15", "high")


def create_labeled_rgb_panel(rgb_image: np.ndarray, label: str) -> np.ndarray:
    """
    Prepends a dark header banner with clean white text to an RGB image panel.
    """
    h, w, _ = rgb_image.shape
    header_height = 42
    panel = np.zeros((h + header_height, w, 3), dtype=np.uint8)
    # Dark header background
    panel[0:header_height, :, :] = 28
    # Separator rule line
    panel[header_height - 1 : header_height, :, :] = 110
    # Copy RGB image
    panel[header_height:, :, :] = rgb_image
    # Render label text
    cv2.putText(
        panel,
        label,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def test_pipeline_integration():
    """
    Executes the integration and validation tests for the processing pipeline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print(" NIGHTVISION: Integrated Colour Processing Pipeline Verification")
    print("=" * 72)

    total_checks = 0
    all_passed = True

    # -------------------------------------------------------------
    # Step 1 & 2: Test 5 Paired Images from LOL Dataset
    # -------------------------------------------------------------
    print("\n[Step 1 & 2] Processing 5 paired images from LOL dataset (eval15)...")

    # Select 5 distinct evaluation samples
    test_filenames = ["1.png", "22.png", "111.png", "146.png", "179.png"]

    # Parameter variations across the 5 test images to verify flexibility
    configurations = [
        {"denoise_method": "gaussian", "kernel_size": 3, "gamma": 0.6, "use_clahe": True, "clip_limit": 2.0},
        {"denoise_method": "median", "kernel_size": 3, "gamma": 0.5, "use_clahe": True, "clip_limit": 2.5},
        {"denoise_method": "mean", "kernel_size": 3, "gamma": 0.6, "use_clahe": True, "clip_limit": 2.0},
        {"denoise_method": "gaussian", "kernel_size": 5, "gamma": 0.55, "use_clahe": True, "clip_limit": 3.0},
        {"denoise_method": "none", "kernel_size": 3, "gamma": 0.5, "use_clahe": True, "clip_limit": 2.0},
    ]

    for idx, filename in enumerate(test_filenames):
        total_checks += 1
        low_path = os.path.join(LOW_DIR, filename)
        high_path = os.path.join(HIGH_DIR, filename)

        if not os.path.exists(low_path):
            raise FileNotFoundError(f"Missing low-light sample: {low_path}")

        # Load low-light image in BGR and convert to RGB
        bgr_low = cv2.imread(low_path)
        if bgr_low is None:
            raise ValueError(f"Failed to load image from {low_path}")
        rgb_low = cv2.cvtColor(bgr_low, cv2.COLOR_BGR2RGB)

        # Retrieve test configuration
        config = configurations[idx % len(configurations)]

        # Execute the pipeline
        enhanced_rgb = enhance_image(
            rgb_low,
            denoise_method=config["denoise_method"],
            kernel_size=config["kernel_size"],
            gamma=config["gamma"],
            use_clahe=config["use_clahe"],
            clip_limit=config["clip_limit"],
            tile_grid_size=(8, 8),
        )

        # -------------------------------------------------------------
        # Step 3 & 4: Shape, Dtype, and Channel Verification
        # -------------------------------------------------------------
        shape_matches = (enhanced_rgb.shape == rgb_low.shape)
        dtype_matches = (enhanced_rgb.dtype == np.uint8)
        channels_valid = (enhanced_rgb.ndim == 3 and enhanced_rgb.shape[2] == 3)
        range_valid = (enhanced_rgb.min() >= 0 and enhanced_rgb.max() <= 255)

        status = "PASS" if (shape_matches and dtype_matches and channels_valid and range_valid) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(
            f"  [{status}] Sample {filename:<8} (Filter: {config['denoise_method']:<8}, Gamma: {config['gamma']}) -> "
            f"Shape: {enhanced_rgb.shape}, Dtype: {enhanced_rgb.dtype}, "
            f"Mean Luma: {enhanced_rgb.mean():.1f} (Orig: {rgb_low.mean():.1f})"
        )

        # -------------------------------------------------------------
        # Step 5: Save Enhanced Result
        # -------------------------------------------------------------
        enhanced_out_path = os.path.join(OUTPUT_DIR, f"enhanced_{filename}")
        # Convert RGB back to BGR for saving with OpenCV
        cv2.imwrite(enhanced_out_path, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))

        # -------------------------------------------------------------
        # Step 6: Generate Side-by-Side Comparisons
        # -------------------------------------------------------------
        panel_low = create_labeled_rgb_panel(rgb_low, "Original (Low-Light)")
        panel_enh = create_labeled_rgb_panel(enhanced_rgb, f"Enhanced ({config['denoise_method']}, g={config['gamma']})")

        # 2-panel comparison
        side_by_side = np.hstack([panel_low, panel_enh])
        cmp_path = os.path.join(OUTPUT_DIR, f"comparison_pair_{filename}")
        cv2.imwrite(cmp_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        # 3-panel comparison including normal-light reference (if available)
        if os.path.exists(high_path):
            bgr_high = cv2.imread(high_path)
            if bgr_high is not None:
                rgb_high = cv2.cvtColor(bgr_high, cv2.COLOR_BGR2RGB)
                panel_high = create_labeled_rgb_panel(rgb_high, "Ground Truth (Normal)")
                trio = np.hstack([panel_low, panel_enh, panel_high])
                trio_path = os.path.join(OUTPUT_DIR, f"comparison_trio_{filename}")
                cv2.imwrite(trio_path, cv2.cvtColor(trio, cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # Step 7: Negative Validation Tests
    # -------------------------------------------------------------
    print("\n[Step 7] Testing parameter validation and negative test cases...")

    valid_rgb_sample = np.zeros((100, 100, 3), dtype=np.uint8)

    negative_cases = [
        # Invalid dimensions
        ("2D Grayscale image array", lambda: enhance_image(np.zeros((100, 100), dtype=np.uint8)), ValueError),
        ("4D image array (1, 100, 100, 3)", lambda: enhance_image(np.zeros((1, 100, 100, 3), dtype=np.uint8)), ValueError),
        ("4-channel image array (BGRA)", lambda: enhance_image(np.zeros((100, 100, 4), dtype=np.uint8)), ValueError),
        ("Empty image array (0, 0, 3)", lambda: enhance_image(np.zeros((0, 0, 3), dtype=np.uint8)), ValueError),

        # Incorrect data types
        ("Float32 image array", lambda: enhance_image(np.zeros((100, 100, 3), dtype=np.float32)), TypeError),
        ("Int32 image array", lambda: enhance_image(np.zeros((100, 100, 3), dtype=np.int32)), TypeError),
        ("Non-numpy input (list)", lambda: enhance_image([[[10, 20, 30]]]), TypeError),

        # Unsupported denoise methods
        ("Unsupported method 'bilateral'", lambda: enhance_image(valid_rgb_sample, denoise_method="bilateral"), ValueError),
        ("Unsupported method 'wavelet'", lambda: enhance_image(valid_rgb_sample, denoise_method="wavelet"), ValueError),
        ("Non-string denoise_method (123)", lambda: enhance_image(valid_rgb_sample, denoise_method=123), TypeError),

        # Invalid kernel sizes
        ("Even kernel size (4)", lambda: enhance_image(valid_rgb_sample, denoise_method="gaussian", kernel_size=4), ValueError),
        ("Out of bounds kernel size (9)", lambda: enhance_image(valid_rgb_sample, denoise_method="mean", kernel_size=9), ValueError),
        ("Float kernel size (3.0)", lambda: enhance_image(valid_rgb_sample, denoise_method="median", kernel_size=3.0), TypeError),
        ("Boolean kernel size (True)", lambda: enhance_image(valid_rgb_sample, denoise_method="gaussian", kernel_size=True), TypeError),

        # Invalid gamma values
        ("Gamma = 0", lambda: enhance_image(valid_rgb_sample, gamma=0), ValueError),
        ("Gamma < 0 (-0.6)", lambda: enhance_image(valid_rgb_sample, gamma=-0.6), ValueError),
        ("String gamma ('0.6')", lambda: enhance_image(valid_rgb_sample, gamma="0.6"), TypeError),
        ("Boolean gamma (True)", lambda: enhance_image(valid_rgb_sample, gamma=True), TypeError),

        # Invalid CLAHE parameters
        ("Non-boolean use_clahe ('True')", lambda: enhance_image(valid_rgb_sample, use_clahe="True"), TypeError),
        ("CLAHE clip_limit <= 0 (0)", lambda: enhance_image(valid_rgb_sample, use_clahe=True, clip_limit=0), ValueError),
        ("CLAHE invalid tile_grid_size (8,)", lambda: enhance_image(valid_rgb_sample, use_clahe=True, tile_grid_size=(8,)), ValueError),
        ("CLAHE negative tile_grid_size (-8, 8)", lambda: enhance_image(valid_rgb_sample, use_clahe=True, tile_grid_size=(-8, 8)), ValueError),
    ]

    for label, test_call, expected_error in negative_cases:
        total_checks += 1
        try:
            test_call()
            print(f"  [FAIL] {label}: Expected {expected_error.__name__}, but no exception was raised.")
            all_passed = False
        except expected_error as err:
            print(f"  [PASS] {label}: Successfully caught {expected_error.__name__}")
        except Exception as unexpected:
            print(f"  [FAIL] {label}: Unexpected exception {type(unexpected).__name__}: {unexpected}")
            all_passed = False

    print("\n" + "=" * 72)
    if all_passed:
        print(f" ALL {total_checks} PIPELINE CHECKS & NEGATIVE TESTS PASSED!")
        print(f" Saved pipeline artifacts to: {os.path.abspath(OUTPUT_DIR)}")
    else:
        print(" SOME CHECKS FAILED. Review output above.")
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    success = test_pipeline_integration()
    exit(0 if success else 1)
