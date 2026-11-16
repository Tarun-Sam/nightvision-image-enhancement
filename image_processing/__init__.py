"""
NIGHTVISION image processing package.
"""

from .filters import (
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    ALLOWED_KERNEL_SIZES,
)
from .enhancement import (
    apply_histogram_equalization,
    apply_clahe,
    apply_gamma_correction,
    apply_contrast_stretching,
)
from .pipeline import (
    enhance_image,
    ALLOWED_DENOISE_METHODS,
)
from .evaluation import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    evaluate_image,
)

__all__ = [
    # Spatial domain filters
    "apply_mean_filter",
    "apply_median_filter",
    "apply_gaussian_filter",
    "ALLOWED_KERNEL_SIZES",
    # Enhancement methods
    "apply_histogram_equalization",
    "apply_clahe",
    "apply_gamma_correction",
    "apply_contrast_stretching",
    # Pipeline
    "enhance_image",
    "ALLOWED_DENOISE_METHODS",
    # Evaluation metrics
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim",
    "evaluate_image",
]
