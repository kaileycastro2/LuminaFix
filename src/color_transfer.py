"""
Color Transfer Module

LAB mean shift with skin and neon protection.
"""

import cv2
import numpy as np
from typing import Optional
from .style_extractor import StyleProfile


class ColorTransfer:
    """Applies color transfer using LAB mean shift with protections."""

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 1.5,
        neon_threshold: float = 0.6,
        neon_protection_strength: float = 0.8
    ):
        """
        Args:
            color_strength: Strength of A/B channel shift (0-1)
            luminance_strength: Strength of L channel shift (0-1), usually 0
            apply_clahe: Apply CLAHE for exposure normalization
            clahe_clip_limit: CLAHE clip limit
            neon_threshold: Saturation threshold for neon detection (0-1)
            neon_protection_strength: How much to protect neon colors (0-1)
        """
        self.color_strength = color_strength
        self.luminance_strength = luminance_strength
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.neon_threshold = neon_threshold
        self.neon_protection_strength = neon_protection_strength

    def apply(
        self,
        target: np.ndarray,
        style: StyleProfile,
        skin_mask: Optional[np.ndarray] = None,
        enable_neon_protection: bool = True
    ) -> np.ndarray:
        """
        Apply color shift to target image with skin and neon protection.

        Steps:
        1. CLAHE on L channel
        2. Detect neon/saturated regions
        3. Shift A/B means toward reference
        4. Protect skin and neon regions by blending back original
        """
        # Keep original for blending back protected regions
        original = target.copy()

        # Step 1: Apply CLAHE for exposure normalization
        if self.apply_clahe:
            target = self._apply_clahe(target)

        # Step 2: Detect neon/saturated regions (if enabled)
        neon_mask = self._detect_neon_regions(original) if enable_neon_protection else np.zeros((original.shape[0], original.shape[1]), dtype=np.float32)

        # Step 3: Convert to LAB and apply color shift
        lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Get target means
        a_mean_target = np.mean(lab[:, :, 1])
        b_mean_target = np.mean(lab[:, :, 2])

        # Calculate shifts toward reference
        a_shift = (style.a_mean - a_mean_target) * self.color_strength
        b_shift = (style.b_mean - b_mean_target) * self.color_strength

        # Apply shifts
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)

        # Optionally shift L channel too
        if self.luminance_strength > 0:
            l_mean_target = np.mean(lab[:, :, 0])
            l_shift = (style.l_mean - l_mean_target) * self.luminance_strength
            lab[:, :, 0] = np.clip(lab[:, :, 0] + l_shift, 0, 255)

        # Step 4: Protect skin regions - blend back original
        if skin_mask is not None and skin_mask.max() > 0:
            skin_protection = 0.5  # Keep 50% original in skin areas
            lab[:, :, 1] = lab[:, :, 1] * (1 - skin_mask * skin_protection) + \
                          original_lab[:, :, 1] * (skin_mask * skin_protection)
            lab[:, :, 2] = lab[:, :, 2] * (1 - skin_mask * skin_protection) + \
                          original_lab[:, :, 2] * (skin_mask * skin_protection)

        # Convert back to BGR
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Step 5: Protect neon regions - blend back ORIGINAL pixels entirely
        # This removes both color shift AND CLAHE effects from neon regions
        if neon_mask.max() > 0:
            neon_mask_3d = neon_mask[:, :, np.newaxis]
            neon_blend = self.neon_protection_strength
            result = (result.astype(np.float32) * (1 - neon_mask_3d * neon_blend) +
                     original.astype(np.float32) * (neon_mask_3d * neon_blend)).astype(np.uint8)

        return result

    def _detect_neon_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect highly saturated (neon) regions in image.

        Returns:
            Soft mask (float32, 0-1) where 1 = neon/saturated region
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Normalize saturation to 0-1
        saturation = hsv[:, :, 1] / 255.0

        # Create mask for high saturation pixels
        neon_mask = np.zeros_like(saturation)

        # Pixels above threshold are considered neon
        high_sat = saturation > self.neon_threshold

        # Smooth transition: ramp from threshold to 1.0
        # At threshold: mask = 0, at saturation=1.0: mask = 1
        if np.any(high_sat):
            neon_mask[high_sat] = (saturation[high_sat] - self.neon_threshold) / (1.0 - self.neon_threshold)

        # Apply Gaussian blur for soft edges
        neon_mask = cv2.GaussianBlur(neon_mask, (15, 15), 0)

        return neon_mask.astype(np.float32)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel ONLY on skin regions with 50% blend."""
        import numpy as np

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Detect skin regions
        skin_mask = self._detect_skin_mask(image)

        # Save original L channel
        l_original = lab[:, :, 0].copy()

        # Apply CLAHE to entire L channel first
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l_original.astype(np.uint8)).astype(np.float32)

        # Blend 50/50 ONLY on skin regions
        l_final = l_original.copy()
        l_final = l_original * (1.0 - skin_mask * 0.5) + l_enhanced * (skin_mask * 0.5)

        lab[:, :, 0] = l_final

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _detect_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """Detect skin tone regions using YCbCr and HSV color spaces."""
        # Convert to YCbCr color space
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Skin detection thresholds in YCbCr
        lower_ycbcr = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycbcr = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycbcr = cv2.inRange(ycbcr, lower_ycbcr, upper_ycbcr)

        # Also use HSV for additional filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine both masks
        skin_mask = cv2.bitwise_and(mask_ycbcr, mask_hsv).astype(np.float32) / 255.0

        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Smooth edges for gradual blending
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

        return skin_mask
