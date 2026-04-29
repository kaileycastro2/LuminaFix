"""
Skin Protection Module

Detects skin regions to protect them during color transfer.
Uses color-based detection (YCbCr) for M1 - no ML required.
"""

import cv2
import numpy as np
from typing import Tuple


class SkinProtection:
    """Detects and creates masks for skin regions."""

    # Standard skin color ranges in YCbCr space
    # These ranges work for a variety of skin tones
    YCBCR_MIN = np.array([0, 77, 133], dtype=np.uint8)
    YCBCR_MAX = np.array([255, 127, 173], dtype=np.uint8)

    # Alternative wider ranges for more inclusive detection
    YCBCR_MIN_WIDE = np.array([0, 70, 120], dtype=np.uint8)
    YCBCR_MAX_WIDE = np.array([255, 135, 180], dtype=np.uint8)

    def __init__(
        self,
        use_wide_range: bool = False,
        blur_kernel: int = 15,
        morph_kernel: int = 5,
        min_area: int = 500
    ):
        """
        Args:
            use_wide_range: Use wider color range for more inclusive detection
            blur_kernel: Gaussian blur kernel size for soft mask edges
            morph_kernel: Morphological operation kernel size
            min_area: Minimum contour area to consider as skin
        """
        self.use_wide_range = use_wide_range
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_area = min_area

        # Select range based on setting
        if use_wide_range:
            self.ycbcr_min = self.YCBCR_MIN_WIDE
            self.ycbcr_max = self.YCBCR_MAX_WIDE
        else:
            self.ycbcr_min = self.YCBCR_MIN
            self.ycbcr_max = self.YCBCR_MAX

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin regions in image.

        Args:
            image: BGR image

        Returns:
            Soft mask (float32, 0-1) where 1 = skin region
        """
        # Convert to YCbCr
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Create binary mask based on color range
        mask = cv2.inRange(ycbcr, self.ycbcr_min, self.ycbcr_max)

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel, self.morph_kernel)
        )

        # Opening to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Closing to fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove small regions
        mask = self._remove_small_regions(mask)

        # Create soft edges with Gaussian blur
        soft_mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (self.blur_kernel, self.blur_kernel),
            0
        )

        # Normalize to 0-1 range
        soft_mask = soft_mask / 255.0

        return soft_mask

    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected components from mask."""
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Create clean mask
        clean_mask = np.zeros_like(mask)

        # Keep only large enough contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)

        return clean_mask

    def detect_with_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative skin detection using HSV color space.
        Can be combined with YCbCr for better results.

        Args:
            image: BGR image

        Returns:
            Soft mask (float32, 0-1)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV
        # Hue: 0-50 (red-yellow range)
        # Saturation: 20-255 (some color, not grayscale)
        # Value: 40-255 (not too dark)
        hsv_min = np.array([0, 20, 40], dtype=np.uint8)
        hsv_max = np.array([50, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        # Also include high hue values (wrapping around to red)
        hsv_min2 = np.array([160, 20, 40], dtype=np.uint8)
        hsv_max2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, hsv_min2, hsv_max2)

        mask = cv2.bitwise_or(mask, mask2)

        # Clean and blur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        return soft_mask / 255.0

    def detect_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Combine YCbCr and HSV detection for more robust results.

        Args:
            image: BGR image

        Returns:
            Soft mask (float32, 0-1)
        """
        ycbcr_mask = self.detect(image)
        hsv_mask = self.detect_with_hsv(image)

        # Combine: pixel is skin if detected by either method
        combined = np.maximum(ycbcr_mask, hsv_mask)

        return combined
