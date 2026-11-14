"""
Image Quality Enhancement for Low-Light Surveillance Applications.

This module provides spatial domain enhancement techniques:
- Histogram Equalization (Global HE): Maximizes global contrast across the entire dynamic range.
- CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast while preventing noise over-amplification.
- Gamma Correction: Non-linear luminance scaling to brighten low-light shadows.
- Contrast Stretching: Linear min-max normalization stretching intensity to full [0, 255] range.
"""

import cv2
import numpy as np


def _validate_image_input(image: np.ndarray) -> None:
    """
    Validates that the input image is a valid 2D grayscale numpy array.

    Parameters:
        image: Input image array.

    Raises:
        TypeError: If image is not a numpy array.
        ValueError: If image is not 2D grayscale or is empty.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected 'image' to be a numpy.ndarray, got {type(image).__name__}."
        )

    if image.ndim != 2:
        raise ValueError(
            f"Input image must be a single-channel grayscale image (2D array). "
            f"Received shape {image.shape} with {image.ndim} dimensions."
        )

    if image.size == 0:
        raise ValueError("Input image array is empty.")

    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected image data type to be uint8, got {image.dtype}. "
            f"Convert image to uint8 before enhancement."
        )


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies Global Histogram Equalization to expand dynamic contrast.

    Computes the cumulative distribution function (CDF) of pixel intensities
    and maps them uniformly across the range [0, 255].

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image (uint8).

    Returns:
        np.ndarray: Contrast-enhanced grayscale image (uint8).
    """
    _validate_image_input(image)

    # cv2.equalizeHist redistributes pixel values across full 0-255 spectrum
    enhanced_image = cv2.equalizeHist(image)

    return enhanced_image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Computes local histograms across discrete contextual tiles and limits
    contrast amplification below a clip limit to prevent noise amplification
    in dark surveillance backgrounds. Bilinear interpolation blends tile boundaries.

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image (uint8).
        clip_limit (float): Threshold for contrast limiting. Default is 2.0.
        tile_grid_size (tuple): Grid dimensions for local tiles (rows, cols). Default is (8, 8).

    Returns:
        np.ndarray: Locally enhanced grayscale image (uint8).
    """
    _validate_image_input(image)

    if not isinstance(clip_limit, (int, float)) or clip_limit <= 0:
        raise ValueError(
            f"clip_limit must be a positive number, got {clip_limit}."
        )

    if (
        not isinstance(tile_grid_size, (tuple, list))
        or len(tile_grid_size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in tile_grid_size)
    ):
        raise ValueError(
            f"tile_grid_size must be a tuple of two positive integers, got {tile_grid_size}."
        )

    # Create OpenCV CLAHE instance with specified clipping and grid dimensions
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid_size[0], tile_grid_size[1]),
    )
    enhanced_image = clahe.apply(image)

    return enhanced_image


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Applies non-linear power-law (Gamma) correction to adjust image luminance.

    For low-light surveillance, gamma < 1.0 brightens underexposed shadow
    regions while avoiding blown-out highlights.
    Relationship: Output = 255 * (Input / 255) ** gamma.

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image (uint8).
        gamma (float): Gamma exponent (> 0). Values < 1.0 brighten shadows.

    Returns:
        np.ndarray: Luminance-adjusted grayscale image (uint8).
    """
    _validate_image_input(image)

    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError(f"gamma must be a positive number, got {gamma}.")

    # Build a 256-element lookup table (LUT) for efficient pixel mapping
    inv_gamma = float(gamma)
    lut = np.array(
        [np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )

    # Apply lookup table transformation using OpenCV's fast LUT
    enhanced_image = cv2.LUT(image, lut)

    return enhanced_image


def apply_contrast_stretching(image: np.ndarray) -> np.ndarray:
    """
    Applies linear min-max contrast stretching to utilize full [0, 255] range.

    Maps minimum intensity to 0 and maximum intensity to 255 via linear scaling:
    Output = (Input - min_val) * (255 / (max_val - min_val)).

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image (uint8).

    Returns:
        np.ndarray: Normalized grayscale image (uint8).
    """
    _validate_image_input(image)

    min_val, max_val = int(image.min()), int(image.max())

    # Avoid division by zero for uniform constant images
    if max_val == min_val:
        return image.copy()

    # OpenCV NORM_MINMAX performs linear stretching to range [0, 255]
    enhanced_image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return enhanced_image
