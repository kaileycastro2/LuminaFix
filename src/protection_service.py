"""
Protection Service - Encapsulates skin and neon protection logic.

Provides a unified interface for computing and applying protection masks.
"""

from typing import Optional
import numpy as np
import logging

from .transfers.base import ProtectionMasks
from .skin_protection import SkinProtection
from .neon_protection import NeonProtection
from .lip_protection import LipProtection
from .eye_protection import EyeProtection

logger = logging.getLogger(__name__)


class ProtectionService:
    """
    Service for computing and managing protection masks.

    Encapsulates the skin and neon detection logic and provides
    a clean interface for the MultiMethodProcessor.
    """

    def __init__(
        self,
        enable_skin_protection: bool = True,
        enable_neon_protection: bool = True,
        enable_lip_protection: bool = False,
        enable_eye_protection: bool = True,
        skin_wide_range: bool = True,
        neon_threshold: float = 0.75
    ):
        """
        Initialize protection service.

        Args:
            enable_skin_protection: Enable skin tone protection
            enable_neon_protection: Enable neon/saturated region protection
            enable_lip_protection: Enable lip region protection (MediaPipe)
            enable_eye_protection: Enable eye region protection (MediaPipe)
            skin_wide_range: Use wider range for skin detection
            neon_threshold: Saturation threshold for neon detection
        """
        self._enable_skin = enable_skin_protection
        self._enable_neon = enable_neon_protection
        self._enable_lips = enable_lip_protection
        self._enable_eyes = enable_eye_protection

        # Initialize detectors
        self._skin_detector = SkinProtection(
            use_wide_range=skin_wide_range,
            blur_kernel=15,
            morph_kernel=5,
            min_area=500
        )

        self._neon_detector = NeonProtection(
            saturation_threshold=neon_threshold,
            saturation_cap=0.90,
            soft_clamp=True
        )

        self._lip_detector = LipProtection(
            blur_kernel=15,
            expand_pixels=5,
            max_num_faces=5
        ) if enable_lip_protection else None

        self._eye_detector = EyeProtection(
            blur_kernel=11,
            expand_pixels=8,
            max_num_faces=5
        ) if enable_eye_protection else None

    @property
    def skin_enabled(self) -> bool:
        return self._enable_skin

    @property
    def neon_enabled(self) -> bool:
        return self._enable_neon

    @property
    def lips_enabled(self) -> bool:
        return self._enable_lips

    @property
    def eyes_enabled(self) -> bool:
        return self._enable_eyes

    def compute_masks(self, image: np.ndarray) -> ProtectionMasks:
        """
        Compute all protection masks for an image.

        Args:
            image: BGR image as numpy array

        Returns:
            ProtectionMasks containing skin, neon, and lip masks
        """
        logger.info(
            "Protection settings — skin=%s, neon=%s, lips=%s (detector=%s), eyes=%s",
            self._enable_skin, self._enable_neon,
            self._enable_lips, self._lip_detector is not None,
            self._enable_eyes
        )

        skin_mask = None
        neon_mask = None
        lip_mask = None
        eye_mask = None

        if self._enable_skin:
            try:
                skin_mask = self._skin_detector.detect(image)
                logger.debug(f"Skin mask computed, max value: {skin_mask.max():.3f}")
            except Exception as e:
                logger.warning(f"Skin detection failed: {e}")

        if self._enable_neon:
            try:
                neon_mask = self._neon_detector.detect_neon_regions(image)
                logger.debug(f"Neon mask computed, max value: {neon_mask.max():.3f}")
            except Exception as e:
                logger.warning(f"Neon detection failed: {e}")

        if self._enable_lips and self._lip_detector is not None:
            try:
                logger.info("Lip protection: running mediapipe face detection...")
                lip_mask = self._lip_detector.detect(image)
                if lip_mask is not None:
                    logger.info(f"Lip mask computed, max value: {lip_mask.max():.3f}")
                else:
                    logger.info("Lip protection: no faces detected by mediapipe")
            except Exception as e:
                logger.warning(f"Lip detection failed: {e}")
        elif self._enable_lips:
            logger.info("Lip protection enabled but detector is None!")
        else:
            logger.info("Lip protection is DISABLED")

        if self._enable_eyes and self._eye_detector is not None:
            try:
                eye_mask = self._eye_detector.detect(image)
                if eye_mask is not None:
                    logger.debug(f"Eye mask computed, max value: {eye_mask.max():.3f}")
                else:
                    logger.debug("No faces detected for eye protection")
            except Exception as e:
                logger.warning(f"Eye detection failed: {e}")

        return ProtectionMasks(skin=skin_mask, neon=neon_mask, lips=lip_mask, eyes=eye_mask)

    def apply_protection(
        self,
        original: np.ndarray,
        transferred: np.ndarray,
        masks: ProtectionMasks,
        skin_blend: float = 0.5,
        neon_blend: float = 0.8,
        lip_blend: float = 0.95,
        eye_blend: float = 0.90
    ) -> np.ndarray:
        """
        Apply protection masks by blending back original in protected regions.

        Args:
            original: Original BGR image
            transferred: Transferred BGR image
            masks: Protection masks
            skin_blend: Blend factor for skin regions (0=full transfer, 1=full original)
            neon_blend: Blend factor for neon regions
            lip_blend: Blend factor for lip regions
            eye_blend: Blend factor for eye regions

        Returns:
            Protected BGR image
        """
        result = transferred.astype(np.float32)
        original_f = original.astype(np.float32)

        # Apply skin protection
        if masks.has_skin():
            skin_mask = masks.skin[:, :, np.newaxis]
            result = result * (1 - skin_mask * skin_blend) + \
                     original_f * (skin_mask * skin_blend)

        # Apply neon protection
        if masks.has_neon():
            neon_mask = masks.neon[:, :, np.newaxis]
            result = result * (1 - neon_mask * neon_blend) + \
                     original_f * (neon_mask * neon_blend)

        # Apply lip protection
        if masks.has_lips():
            lip_mask = masks.lips[:, :, np.newaxis]
            result = result * (1 - lip_mask * lip_blend) + \
                     original_f * (lip_mask * lip_blend)

        # Apply eye protection
        if masks.has_eyes():
            eye_mask = masks.eyes[:, :, np.newaxis]
            result = result * (1 - eye_mask * eye_blend) + \
                     original_f * (eye_mask * eye_blend)

        return np.clip(result, 0, 255).astype(np.uint8)

    def update_settings(
        self,
        enable_skin: Optional[bool] = None,
        enable_neon: Optional[bool] = None,
        enable_lips: Optional[bool] = None,
        enable_eyes: Optional[bool] = None
    ) -> None:
        """Update protection settings."""
        if enable_skin is not None:
            self._enable_skin = enable_skin
        if enable_neon is not None:
            self._enable_neon = enable_neon
        if enable_lips is not None:
            self._enable_lips = enable_lips
            if enable_lips and self._lip_detector is None:
                self._lip_detector = LipProtection(
                    blur_kernel=9,
                    expand_pixels=3,
                    max_num_faces=5
                )
        if enable_eyes is not None:
            self._enable_eyes = enable_eyes
            if enable_eyes and self._eye_detector is None:
                self._eye_detector = EyeProtection(
                    blur_kernel=11,
                    expand_pixels=8,
                    max_num_faces=5
                )


# Module-level singleton
_protection_service: Optional[ProtectionService] = None


def get_protection_service(
    enable_skin_protection: bool = True,
    enable_neon_protection: bool = True,
    enable_lip_protection: bool = False,
    enable_eye_protection: bool = True
) -> ProtectionService:
    """Get or create the global ProtectionService instance."""
    global _protection_service
    if _protection_service is None:
        _protection_service = ProtectionService(
            enable_skin_protection=enable_skin_protection,
            enable_neon_protection=enable_neon_protection,
            enable_lip_protection=enable_lip_protection,
            enable_eye_protection=enable_eye_protection
        )
    else:
        # Update settings if different
        _protection_service.update_settings(
            enable_skin=enable_skin_protection,
            enable_neon=enable_neon_protection,
            enable_lips=enable_lip_protection,
            enable_eyes=enable_eye_protection
        )
    return _protection_service
