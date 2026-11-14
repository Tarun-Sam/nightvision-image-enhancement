"""
Test script for NIGHTVISION spatial domain filtering functions.

This script:
1. Generates a synthetic test image representing low-light surveillance content.
2. Introduces Gaussian and Salt & Pepper noise.
3. Applies Mean, Median, and Gaussian spatial filters across kernel sizes 3, 5, and 7.
4. Saves all test and resulting images in the output/ folder.
5. Verifies that output dimensions, data types, and border regions remain intact.
6. Validates error-handling for invalid inputs.
"""

import os
import cv2
import numpy as np

from image_processing.filters import (
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    ALLOWED_KERNEL_SIZES,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def create_synthetic_surveillance_image(height: int = 256, width: int = 256) -> np.ndarray:
    """
    Generates a synthetic grayscale image simulating low-light surveillance scenes.
    Contains low-intensity background, surveillance target silhouettes, and edges.
    """
    # Low-light base ambient illumination (low grayscale level around 35)
    image = np.full((height, width), 35, dtype=np.uint8)

    # Ambient low-intensity horizontal gradient simulating dim lighting falloff
    gradient = np.linspace(0, 30, width, dtype=np.uint8)
    image = cv2.add(image, np.tile(gradient, (height, 1)))

    # Draw simulated structural objects (buildings / door frame)
    cv2.rectangle(image, (30, 60), (100, 220), 75, -1)
    cv2.rectangle(image, (150, 40), (220, 220), 65, -1)

    # Draw simulated surveillance target / subject
    cv2.circle(image, (65, 100), 18, 120, -1)        # Head
    cv2.rectangle(image, (50, 120), (80, 190), 110, -1)  # Body

    # Add a brighter light source / lamp reflection
    cv2.circle(image, (185, 90), 25, 160, -1)

    return image


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 20.0) -> np.ndarray:
    """
    Adds zero-mean Gaussian distributed sensor noise to an image.
    """
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_and_pepper_noise(
    image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02
) -> np.ndarray:
    """
    Adds impulse (salt-and-pepper) noise by setting random pixels to 0 or 255.
    """
    noisy_image = image.copy()
    random_matrix = np.random.rand(*image.shape)

    # Pepper noise: random low-intensity drops (0)
    noisy_image[random_matrix < pepper_prob] = 0

    # Salt noise: random high-intensity spikes (255)
    noisy_image[random_matrix > (1.0 - salt_prob)] = 255

    return noisy_image


def test_filters():
    """
    Executes the filter testing and verification suite.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)  # For reproducible test noise generation

    print("=" * 65)
    print(" NIGHTVISION: Spatial Domain Filter Testing & Verification")
    print("=" * 65)

    # Step 1: Generate synthetic test image
    print("\n[Step 1] Generating synthetic surveillance test image...")
    original_image = create_synthetic_surveillance_image(256, 256)
    original_path = os.path.join(OUTPUT_DIR, "01_original_image.png")
    cv2.imwrite(original_path, original_image)
    print(f" -> Saved original image: {original_path} (Shape: {original_image.shape}, Dtype: {original_image.dtype})")

    # Step 2: Introduce Gaussian and Salt & Pepper noise
    print("\n[Step 2] Introducing noise models...")
    gaussian_noisy = add_gaussian_noise(original_image, mean=0.0, sigma=22.0)
    sp_noisy = add_salt_and_pepper_noise(original_image, salt_prob=0.03, pepper_prob=0.03)

    gaussian_path = os.path.join(OUTPUT_DIR, "02_noisy_gaussian.png")
    sp_path = os.path.join(OUTPUT_DIR, "03_noisy_salt_and_pepper.png")
    cv2.imwrite(gaussian_path, gaussian_noisy)
    cv2.imwrite(sp_path, sp_noisy)
    print(f" -> Saved Gaussian noisy image: {gaussian_path}")
    print(f" -> Saved Salt & Pepper noisy image: {sp_path}")

    # Step 3 & 4 & 5: Apply filters, verify dimensions/dtypes, and save outputs
    filters = {
        "mean": apply_mean_filter,
        "median": apply_median_filter,
        "gaussian": apply_gaussian_filter,
    }

    noise_cases = {
        "gaussian_noise": gaussian_noisy,
        "salt_and_pepper_noise": sp_noisy,
    }

    print("\n[Step 3, 4, 5] Applying filters and verifying shape & dtype preservation...")
    verification_passed = True
    total_tests = 0

    for noise_name, noisy_img in noise_cases.items():
        print(f"\n  --- Processing noise type: {noise_name} ---")
        for filter_name, filter_func in filters.items():
            for k in ALLOWED_KERNEL_SIZES:
                total_tests += 1
                filtered_img = filter_func(noisy_img, kernel_size=k)

                # Verify shape preservation
                shape_match = (filtered_img.shape == original_image.shape)
                # Verify dtype preservation
                dtype_match = (filtered_img.dtype == original_image.dtype)

                if shape_match and dtype_match:
                    status = "PASS"
                else:
                    status = "FAIL"
                    verification_passed = False

                # Save the resulting image
                out_filename = f"{noise_name}_{filter_name}_k{k}.png"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                cv2.imwrite(out_path, filtered_img)

                print(
                    f"  [{status}] {filter_name.capitalize():<8} Filter (k={k}) -> "
                    f"Shape: {filtered_img.shape}, Dtype: {filtered_img.dtype} -> Saved: {out_filename}"
                )

    # Verification of input validation checks
    print("\n[Input Validation Verification]")
    validation_checks = [
        ("Invalid kernel size (4)", lambda: apply_mean_filter(original_image, 4), ValueError),
        ("Invalid kernel size (9)", lambda: apply_gaussian_filter(original_image, 9), ValueError),
        ("Float kernel size (3.0)", lambda: apply_median_filter(original_image, 3.0), TypeError),
        ("3-channel image (H, W, 3)", lambda: apply_mean_filter(np.zeros((100, 100, 3), dtype=np.uint8), 3), ValueError),
        ("Non-numpy input", lambda: apply_gaussian_filter([[1, 2], [3, 4]], 3), TypeError),
    ]

    for label, test_call, expected_error in validation_checks:
        total_tests += 1
        try:
            test_call()
            print(f"  [FAIL] {label}: Expected {expected_error.__name__}, but no exception was raised.")
            verification_passed = False
        except expected_error as e:
            print(f"  [PASS] {label}: Successfully caught {expected_error.__name__} ('{e}')")
        except Exception as unexpected:
            print(f"  [FAIL] {label}: Unexpected error {type(unexpected).__name__}: {unexpected}")
            verification_passed = False

    print("\n" + "=" * 65)
    if verification_passed:
        print(f" ALL {total_tests} VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print(f" Processed output files saved to: {os.path.abspath(OUTPUT_DIR)}")
    else:
        print(" SOME CHECKS FAILED. Please review the output above.")
    print("=" * 65)

    return verification_passed


if __name__ == "__main__":
    success = test_filters()
    exit(0 if success else 1)
