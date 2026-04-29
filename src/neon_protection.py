"""
Neon Protection Module

Prevents oversaturation and color clipping for highly saturated regions.
Handles neon lights, vibrant clothing, etc.
"""

import cv2
import numpy as np
from typing import Tuple


class NeonProtection:
    """Protects highly saturated colors from clipping/distortion."""

    def __init__(
        self,
        saturation_threshold: float = 0.75,
        saturation_cap: float = 0.90,
        soft_clamp: bool = True,
        preserve_hue: bool = True
    ):
        """
        Args:
            saturation_threshold: Saturation level above which to start protection (0-1)
            saturation_cap: Maximum allowed saturation (0-1)
            soft_clamp: Use soft roll-off instead of hard clamp
            preserve_hue: Preserve original hue when clamping saturation
        """
        self.saturation_threshold = saturation_threshold
        self.saturation_cap = saturation_cap
        self.soft_clamp = soft_clamp
        self.preserve_hue = preserve_hue

    def protect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply neon protection to prevent oversaturation.

        Args:
            image: BGR image

        Returns:
            Protected BGR image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Normalize saturation to 0-1
        saturation = hsv[:, :, 1] / 255.0

        if self.soft_clamp:
            # Soft roll-off: gradually reduce saturation above threshold
            saturation = self._soft_clamp_saturation(saturation)
        else:
            # Hard clamp
            saturation = np.clip(saturation, 0, self.saturation_cap)

        # Apply modified saturation
        hsv[:, :, 1] = saturation * 255.0

        # Convert back to BGR
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def _soft_clamp_saturation(self, saturation: np.ndarray) -> np.ndarray:
        """
        Apply soft roll-off to saturation values.
        Uses a smooth curve instead of hard clipping.

        Args:
            saturation: Saturation values (0-1)

        Returns:
            Clamped saturation values
        """
        threshold = self.saturation_threshold
        cap = self.saturation_cap

        # For values below threshold, keep unchanged
        # For values above threshold, apply smooth compression

        result = np.copy(saturation)

        # Find pixels above threshold
        high_sat_mask = saturation > threshold

        if np.any(high_sat_mask):
            # Map [threshold, 1.0] -> [threshold, cap] using smooth curve
            # Using tanh-like compression
            high_values = saturation[high_sat_mask]

            # Normalize to 0-1 range for the high region
            normalized = (high_values - threshold) / (1.0 - threshold)

            # Apply soft compression (using sqrt for gentle roll-off)
            compressed = np.sqrt(normalized) * (cap - threshold) + threshold

            result[high_sat_mask] = compressed

        return result

    def detect_neon_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect highly saturated (neon) regions in image.

        Args:
            image: BGR image

        Returns:
            Mask (float32, 0-1) where 1 = neon/high saturation region
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0

        # Neon regions: high saturation AND high value (bright)
        neon_mask = (saturation > self.saturation_threshold) & (value > 0.5)

        return neon_mask.astype(np.float32)

    def gamut_map(self, image: np.ndarray) -> np.ndarray:
        """
        Perform gamut mapping to bring out-of-gamut colors back.
        Prevents color clipping in RGB space.

        Args:
            image: BGR image (may have out-of-range values if float)

        Returns:
            Gamut-mapped BGR image
        """
        # Convert to float if not already
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32)
        else:
            img_float = image.copy()

        # Check for out-of-gamut pixels
        max_val = np.max(img_float, axis=2, keepdims=True)
        min_val = np.min(img_float, axis=2, keepdims=True)

        # Scale down pixels that exceed 255
        scale_down = np.where(max_val > 255, 255.0 / max_val, 1.0)
        img_float = img_float * scale_down

        # Scale up pixels that are below 0
        # (shift to positive then scale)
        needs_shift = min_val < 0
        if np.any(needs_shift):
            # Simple approach: clip negative values
            img_float = np.clip(img_float, 0, 255)

        return img_float.astype(np.uint8)

    def protect_with_mask(
        self,
        original: np.ndarray,
        transferred: np.ndarray,
        blend_factor: float = 0.5
    ) -> np.ndarray:
        """
        Protect neon regions by blending back original colors.

        Args:
            original: Original BGR image
            transferred: Color-transferred BGR image
            blend_factor: How much of original to blend in neon regions (0-1)

        Returns:
            Protected BGR image
        """
        # Detect neon regions in original
        neon_mask = self.detect_neon_regions(original)

        # Expand mask for smooth blending
        neon_mask = cv2.GaussianBlur(neon_mask, (11, 11), 0)
        neon_mask = neon_mask[:, :, np.newaxis]

        # Blend: in neon regions, keep more of original
        result = transferred * (1 - neon_mask * blend_factor) + original * (neon_mask * blend_factor)

        return result.astype(np.uint8)
