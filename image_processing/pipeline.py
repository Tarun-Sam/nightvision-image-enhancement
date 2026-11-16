"""
NIGHTVISION Integrated Processing Pipeline.

This module provides colour-image support using the OpenCV LAB colour space:
- Transforms RGB images to CIE LAB colour space.
- Isolates the L (Lightness/Luminance) channel for processing.
- Applies spatial domain denoising (Mean, Median, Gaussian, or None).
- Applies non-linear Gamma Correction to brighten low-light shadows.
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement.
- Recombines the enhanced L channel with the original chromaticity (A and B) channels.
- Transforms LAB back to RGB colour space, preserving colour fidelity.
"""

from typing import Tuple
import cv2
import numpy as np

from .filters import (
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    ALLOWED_KERNEL_SIZES,
)
from .enhancement import (
    apply_clahe,
    apply_gamma_correction,
)

ALLOWED_DENOISE_METHODS = ("none", "mean", "median", "gaussian")


def _validate_pipeline_inputs(
    image: np.ndarray,
    denoise_method: str,
    kernel_size: int,
    gamma: float,
    use_clahe: bool,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> str:
    """
    Validates input parameters for the enhance_image pipeline.

    Returns:
        str: Normalized lower-case denoise_method.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected 'image' to be a numpy.ndarray, got {type(image).__name__}."
        )

    if image.size == 0:
        raise ValueError("Input image array is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Input image must be a 3-channel RGB colour image (height, width, 3). "
            f"Received shape {image.shape} with {image.ndim} dimensions."
        )

    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected image data type to be uint8, got {image.dtype}. "
            f"Convert image to uint8 before enhancement."
        )

    if not isinstance(denoise_method, str):
        raise TypeError(
            f"Expected 'denoise_method' to be a string, got {type(denoise_method).__name__}."
        )

    norm_method = denoise_method.strip().lower()
    if norm_method not in ALLOWED_DENOISE_METHODS:
        raise ValueError(
            f"Unsupported denoise_method '{denoise_method}'. "
            f"Allowed methods are: {list(ALLOWED_DENOISE_METHODS)}."
        )

    if norm_method != "none":
        if not isinstance(kernel_size, (int, np.integer)) or isinstance(
            kernel_size, bool
        ):
            raise TypeError(
                f"Expected 'kernel_size' to be an integer, got {type(kernel_size).__name__}."
            )
        if kernel_size not in ALLOWED_KERNEL_SIZES:
            raise ValueError(
                f"Invalid kernel size: {kernel_size}. Allowed kernel sizes are {ALLOWED_KERNEL_SIZES}."
            )

    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
        raise TypeError(
            f"Expected 'gamma' to be a numeric value, got {type(gamma).__name__}."
        )

    if gamma <= 0:
        raise ValueError(f"gamma must be a positive number (> 0), got {gamma}.")

    if not isinstance(use_clahe, bool):
        raise TypeError(
            f"Expected 'use_clahe' to be a boolean, got {type(use_clahe).__name__}."
        )

    if use_clahe:
        if not isinstance(clip_limit, (int, float)) or isinstance(clip_limit, bool):
            raise TypeError(
                f"Expected 'clip_limit' to be numeric, got {type(clip_limit).__name__}."
            )
        if clip_limit <= 0:
            raise ValueError(
                f"clip_limit must be a positive number (> 0), got {clip_limit}."
            )

        if (
            not isinstance(tile_grid_size, (tuple, list))
            or len(tile_grid_size) != 2
            or not all(
                isinstance(dim, (int, np.integer))
                and not isinstance(dim, bool)
                and dim > 0
                for dim in tile_grid_size
            )
        ):
            raise ValueError(
                f"tile_grid_size must be a tuple of two positive integers, got {tile_grid_size}."
            )

    return norm_method


def enhance_image(
    image: np.ndarray,
    denoise_method: str = "gaussian",
    kernel_size: int = 3,
    gamma: float = 0.6,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhances a low-light RGB colour image using the NIGHTVISION pipeline.

    Workflow:
    1. Validates input RGB image dimensions, data type, and parameters.
    2. Converts RGB image to CIE LAB colour space.
    3. Extracts the L (Lightness) channel.
    4. Applies spatial domain filtering to the L channel (Mean, Median, Gaussian, or None).
    5. Applies Gamma Correction to the L channel to boost shadow luminance.
    6. Applies CLAHE to the L channel (if use_clahe=True) for adaptive local contrast.
    7. Merges the enhanced L channel with the preserved original A and B chrominance channels.
    8. Converts LAB back to RGB colour space.
    9. Returns the reconstructed RGB image.

    Parameters:
        image (np.ndarray): 3-channel RGB uint8 image.
        denoise_method (str): Spatial filter ('none', 'mean', 'median', 'gaussian'). Default 'gaussian'.
        kernel_size (int): Kernel window dimension (3, 5, or 7). Default 3.
        gamma (float): Power-law gamma value (> 0). Default 0.6.
        use_clahe (bool): Whether to apply CLAHE enhancement. Default True.
        clip_limit (float): Threshold for CLAHE contrast limiting. Default 2.0.
        tile_grid_size (tuple): Grid dimensions (rows, cols) for CLAHE tiles. Default (8, 8).

    Returns:
        np.ndarray: Enhanced RGB uint8 image preserving input dimensions and chromaticity.
    """
    # 1. Parameter validation
    norm_method = _validate_pipeline_inputs(
        image=image,
        denoise_method=denoise_method,
        kernel_size=kernel_size,
        gamma=gamma,
        use_clahe=use_clahe,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    # 2. Convert RGB to CIE LAB colour space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 3. Extract L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 4. Apply selected spatial domain filter to L channel only
    if norm_method == "mean":
        l_channel = apply_mean_filter(l_channel, kernel_size=kernel_size)
    elif norm_method == "median":
        l_channel = apply_median_filter(l_channel, kernel_size=kernel_size)
    elif norm_method == "gaussian":
        l_channel = apply_gaussian_filter(l_channel, kernel_size=kernel_size)
    # If norm_method == "none", no filter is applied

    # 5. Apply Gamma Correction to L channel
    l_channel = apply_gamma_correction(l_channel, gamma=gamma)

    # 6. Apply CLAHE to L channel if enabled
    if use_clahe:
        l_channel = apply_clahe(
            l_channel,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    # 7. Merge processed L channel with preserved original A and B channels
    merged_lab = cv2.merge([l_channel, a_channel, b_channel])

    # 8. Convert LAB back to RGB colour space
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    return enhanced_rgb
