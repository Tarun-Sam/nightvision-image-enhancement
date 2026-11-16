"""
Image Quality Evaluation Metrics Module.

This module provides quantitative objective quality metrics for denoised and enhanced images:
- Mean Squared Error (MSE): Measures cumulative squared pixel intensity difference.
- Peak Signal-to-Noise Ratio (PSNR): Ratio of maximum signal power to corrupting noise power (dB).
- Structural Similarity Index Measure (SSIM): Evaluates luminance, contrast, and structural preservation.

Supports both 2D single-channel grayscale and 3-channel RGB colour images (uint8).
"""

from typing import Dict
import math
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error


def _validate_evaluation_inputs(
    reference: np.ndarray, processed: np.ndarray
) -> None:
    """
    Validates that reference and processed images are compatible for metric evaluation.

    Parameters:
        reference: Ground-truth / reference image array.
        processed: Filtered / enhanced image array.

    Raises:
        TypeError: If either image is not a numpy array or not uint8.
        ValueError: If images are empty, dimensions/channels mismatch, or shape is unsupported.
    """
    if not isinstance(reference, np.ndarray) or not isinstance(processed, np.ndarray):
        raise TypeError(
            f"Both reference and processed must be numpy.ndarray instances. "
            f"Got {type(reference).__name__} and {type(processed).__name__}."
        )

    if reference.size == 0 or processed.size == 0:
        raise ValueError("Image arrays cannot be empty.")

    if reference.shape != processed.shape:
        raise ValueError(
            f"Image dimensions must match. Reference shape: {reference.shape}, "
            f"Processed shape: {processed.shape}."
        )

    if reference.dtype != np.uint8 or processed.dtype != np.uint8:
        raise TypeError(
            f"Both images must have data type uint8. "
            f"Got reference dtype {reference.dtype} and processed dtype {processed.dtype}."
        )

    if reference.ndim not in (2, 3):
        raise ValueError(
            f"Images must be 2D grayscale or 3D colour arrays. "
            f"Received array with {reference.ndim} dimensions."
        )

    if reference.ndim == 3 and reference.shape[2] != 3:
        raise ValueError(
            f"3D colour images must have exactly 3 channels (RGB). "
            f"Received array with {reference.shape[2]} channels."
        )


def _compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """Internal MSE computation assuming already validated inputs."""
    return float(mean_squared_error(reference, processed))


def _compute_psnr(
    reference: np.ndarray, processed: np.ndarray, mse_val: float = None
) -> float:
    """Internal PSNR computation assuming already validated inputs."""
    if mse_val is None:
        mse_val = _compute_mse(reference, processed)

    if mse_val == 0.0:
        return float("inf")

    max_intensity = 255.0
    return float(10.0 * math.log10((max_intensity**2) / mse_val))


def _compute_ssim(reference: np.ndarray, processed: np.ndarray) -> float:
    """Internal SSIM computation assuming already validated inputs."""
    channel_axis = 2 if reference.ndim == 3 else None
    return float(
        structural_similarity(
            reference, processed, channel_axis=channel_axis, data_range=255
        )
    )


def calculate_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between reference and processed images.

    Formula:
        MSE = (1 / N) * sum((I_ref - I_proc)^2)
        where N is total number of pixels (M * N for 2D, M * N * 3 for RGB).

    Parameters:
        reference (np.ndarray): Reference ground-truth uint8 image (2D or RGB).
        processed (np.ndarray): Processed uint8 image with matching dimensions.

    Returns:
        float: Calculated Mean Squared Error value (lower is closer to reference).
    """
    _validate_evaluation_inputs(reference, processed)
    return _compute_mse(reference, processed)


def calculate_psnr(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula:
        PSNR = 10 * log10(MAX_I^2 / MSE) = 20 * log10(255 / sqrt(MSE))
        where MAX_I = 255 for uint8 images.

    Handles identical images explicitly: if MSE == 0, returns float('inf').

    Parameters:
        reference (np.ndarray): Reference ground-truth uint8 image (2D or RGB).
        processed (np.ndarray): Processed uint8 image with matching dimensions.

    Returns:
        float: PSNR in dB (higher indicates superior reconstruction fidelity).
    """
    _validate_evaluation_inputs(reference, processed)
    return _compute_psnr(reference, processed)


def calculate_ssim(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM).

    Evaluates structural information degradation based on luminance, contrast,
    and structure. Values range from -1.0 to 1.0 (1.0 indicates identical images).

    Parameters:
        reference (np.ndarray): Reference ground-truth uint8 image (2D or RGB).
        processed (np.ndarray): Processed uint8 image with matching dimensions.

    Returns:
        float: SSIM score.
    """
    _validate_evaluation_inputs(reference, processed)
    return _compute_ssim(reference, processed)


def evaluate_image(
    reference: np.ndarray, processed: np.ndarray
) -> Dict[str, float]:
    """
    Computes all standard image quality metrics (MSE, PSNR, SSIM) simultaneously.

    Validates inputs once and reuses intermediate calculations for efficiency.

    Parameters:
        reference (np.ndarray): Reference ground-truth uint8 image (2D or RGB).
        processed (np.ndarray): Processed uint8 image with matching dimensions.

    Returns:
        dict: Dictionary containing 'mse', 'psnr', and 'ssim' scores.
    """
    _validate_evaluation_inputs(reference, processed)

    mse_val = _compute_mse(reference, processed)
    psnr_val = _compute_psnr(reference, processed, mse_val=mse_val)
    ssim_val = _compute_ssim(reference, processed)

    return {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }
