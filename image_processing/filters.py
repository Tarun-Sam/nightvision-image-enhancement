"""
Spatial Domain Filters for Image Denoising.

This module provides spatial filtering operations using OpenCV:
- Mean (Averaging) Filter: Blurs images and attenuates uniform noise.
- Median Filter: Effective non-linear filter for impulse (salt-and-pepper) noise.
- Gaussian Filter: Weighted smoothing filter for natural/Gaussian noise.
"""

import cv2
import numpy as np

# Allowed kernel dimensions for spatial domain filtering
ALLOWED_KERNEL_SIZES = (3, 5, 7)


def _validate_filter_inputs(image: np.ndarray, kernel_size: int) -> None:
    """
    Validates the input image and kernel size for spatial filtering.

    Parameters:
        image: Input image array.
        kernel_size: Size of the neighborhood filter window.

    Raises:
        TypeError: If image is not a numpy array or kernel_size is not an integer.
        ValueError: If image is not 2D grayscale or kernel_size is not 3, 5, or 7.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected 'image' to be a numpy.ndarray, got {type(image).__name__}."
        )

    if image.ndim != 2:
        raise ValueError(
            f"Input image must be a single-channel grayscale image (2D array). "
            f"Received an array with shape {image.shape} (ndim={image.ndim})."
        )

    if image.size == 0:
        raise ValueError("Input image array is empty.")

    # In Python, bool is a subclass of int, so explicitly exclude booleans
    if not isinstance(kernel_size, (int, np.integer)) or isinstance(kernel_size, bool):
        raise TypeError(
            f"Expected 'kernel_size' to be an integer, got {type(kernel_size).__name__}."
        )

    if kernel_size not in ALLOWED_KERNEL_SIZES:
        raise ValueError(
            f"Invalid kernel size: {kernel_size}. Allowed kernel sizes are {ALLOWED_KERNEL_SIZES}."
        )


def apply_mean_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a spatial mean (averaging) filter to a single-channel grayscale image.

    Each output pixel is the arithmetic mean of the neighboring pixels
    within the square kernel window.

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image.
        kernel_size (int): Size of the square kernel window (3, 5, or 7).

    Returns:
        np.ndarray: Denoised grayscale image with preserved shape and data type.
    """
    # Validate image dimensions and kernel size
    _validate_filter_inputs(image, kernel_size)

    # cv2.blur convolves the image with a normalized box filter of size (kernel_size, kernel_size).
    # cv2.BORDER_DEFAULT (BORDER_REFLECT_101) reflects edge pixels to prevent boundary artifacts.
    filtered_image = cv2.blur(
        image, (kernel_size, kernel_size), borderType=cv2.BORDER_DEFAULT
    )

    return filtered_image


def apply_median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a spatial median filter to a single-channel grayscale image.

    Each output pixel is replaced by the median value of all pixels
    in the square kernel window. Highly effective for impulse/salt-and-pepper noise.

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image.
        kernel_size (int): Size of the square kernel window (3, 5, or 7).

    Returns:
        np.ndarray: Denoised grayscale image with preserved shape and data type.
    """
    # Validate image dimensions and kernel size
    _validate_filter_inputs(image, kernel_size)

    # cv2.medianBlur replaces each pixel with the median of its neighborhood.
    # Border pixels are replicated/reflected automatically by OpenCV.
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image


def apply_gaussian_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a 2D Gaussian smoothing filter to a single-channel grayscale image.

    Weights neighboring pixels using a Gaussian distribution, giving higher
    weight to closer pixels. Effective for smoothing Gaussian sensor noise.

    Parameters:
        image (np.ndarray): Single-channel 2D grayscale image.
        kernel_size (int): Size of the square kernel window (3, 5, or 7).

    Returns:
        np.ndarray: Denoised grayscale image with preserved shape and data type.
    """
    # Validate image dimensions and kernel size
    _validate_filter_inputs(image, kernel_size)

    # sigmaX=0 directs OpenCV to calculate standard deviation based on kernel size:
    # sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    # cv2.BORDER_DEFAULT handles border reflection.
    filtered_image = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=0, borderType=cv2.BORDER_DEFAULT
    )

    return filtered_image
