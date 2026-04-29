"""
Style Extractor Module

Extracts color/tone characteristics from a reference image.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class StyleProfile:
    """Container for extracted style characteristics."""
    # LAB statistics
    l_mean: float
    l_std: float
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float

    # Histogram data for tone curve matching
    l_histogram: np.ndarray

    # HSV saturation profile
    saturation_mean: float
    saturation_std: float

    # Contrast metric
    contrast: float

    # Original reference for histogram matching
    reference_lab: Optional[np.ndarray] = None


class StyleExtractor:
    """Extracts style profile from reference image."""

    def __init__(self, store_reference: bool = True):
        """
        Args:
            store_reference: If True, stores LAB reference for histogram matching
        """
        self.store_reference = store_reference

    def extract(self, image: np.ndarray) -> StyleProfile:
        """
        Extract style profile from reference image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            StyleProfile with extracted characteristics
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Extract LAB statistics
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        l_mean, l_std = np.mean(l_channel), np.std(l_channel)
        a_mean, a_std = np.mean(a_channel), np.std(a_channel)
        b_mean, b_std = np.mean(b_channel), np.std(b_channel)

        # Compute L channel histogram for tone curve
        l_histogram, _ = np.histogram(l_channel.flatten(), bins=256, range=(0, 256))
        l_histogram = l_histogram.astype(np.float32) / l_histogram.sum()

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]
        sat_mean, sat_std = np.mean(saturation), np.std(saturation)

        # Contrast metric (standard deviation of luminance)
        contrast = l_std

        # Store reference LAB if needed for histogram matching
        ref_lab = lab if self.store_reference else None

        return StyleProfile(
            l_mean=l_mean,
            l_std=l_std,
            a_mean=a_mean,
            a_std=a_std,
            b_mean=b_mean,
            b_std=b_std,
            l_histogram=l_histogram,
            saturation_mean=sat_mean,
            saturation_std=sat_std,
            contrast=contrast,
            reference_lab=ref_lab
        )

    def extract_from_path(self, image_path: str) -> StyleProfile:
        """
        Extract style profile from image file.

        Args:
            image_path: Path to reference image

        Returns:
            StyleProfile with extracted characteristics
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.extract(image)
