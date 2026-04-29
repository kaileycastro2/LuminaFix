"""
Abstract base class for style transfer methods.

Implements the Template Method pattern for common operations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import cv2

from .base import ProtectionMasks


class AbstractTransfer(ABC):
    """
    Abstract base class with template method for common operations.

    Subclasses must implement:
    - method_id (property)
    - method_name (property)
    - method_type (property)
    - is_available()
    - load_reference()
    - _apply_transfer()

    Optional overrides:
    - _preprocess()
    - _postprocess()
    """

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0
    ):
        """
        Initialize transfer method.

        Args:
            color_strength: Strength of color transfer (0-1)
            luminance_strength: Strength of luminance transfer (0-1)
        """
        self._color_strength = color_strength
        self._luminance_strength = luminance_strength
        self._reference_loaded = False

    @property
    def neon_blend_factor(self) -> float:
        """Neon blend factor for this method. Override for stronger blending."""
        return 0.8  # Default: 80% original in neon areas

    @property
    @abstractmethod
    def method_id(self) -> str:
        """Unique identifier for this method."""
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Human-readable name for display."""
        pass

    @property
    @abstractmethod
    def method_type(self) -> str:
        """Type: 'classic' or 'neural'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this method is available."""
        pass

    @abstractmethod
    def load_reference(self, image: np.ndarray) -> None:
        """Load and prepare reference image/style."""
        pass

    @abstractmethod
    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply the actual transfer logic.

        Subclasses implement their specific algorithm here.

        Args:
            image: Preprocessed BGR image
            strength: Transfer strength

        Returns:
            Transferred BGR image
        """
        pass

    def transfer(
        self,
        target: np.ndarray,
        strength: float = 1.0,
        masks: Optional[ProtectionMasks] = None
    ) -> np.ndarray:
        """
        Template method - defines the algorithm skeleton.

        1. Preprocess
        2. Apply transfer
        3. Apply protection (if masks provided)
        4. Postprocess (blend by strength)

        Args:
            target: BGR target image
            strength: Transfer strength (0-1)
            masks: Optional protection masks

        Returns:
            Processed BGR image
        """
        if not self._reference_loaded:
            raise RuntimeError(f"{self.method_name}: Reference not loaded. Call load_reference() first.")

        # Store original for blending
        original = target.copy()

        # Step 1: Preprocess
        preprocessed = self._preprocess(target)

        # Step 2: Apply transfer
        result = self._apply_transfer(preprocessed, strength)

        # Step 3: Apply protection masks
        if masks is not None:
            result = self._apply_protection(original, result, masks)

        # Step 4: Postprocess (strength blending)
        result = self._postprocess(result, original, strength)

        return result

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing step.

        Override in subclasses for method-specific preprocessing.
        Default: return image unchanged.
        """
        return image

    def _postprocess(
        self,
        result: np.ndarray,
        original: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Postprocess result with strength blending.

        Blends result with original based on strength parameter.
        """
        # Resize result if dimensions don't match original
        if result.shape[:2] != original.shape[:2]:
            result = cv2.resize(
                result,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        if strength >= 1.0:
            return result

        if strength <= 0.0:
            return original

        # Blend based on strength
        return cv2.addWeighted(
            original.astype(np.float32),
            1.0 - strength,
            result.astype(np.float32),
            strength,
            0
        ).astype(np.uint8)

    def _apply_protection(
        self,
        original: np.ndarray,
        transferred: np.ndarray,
        masks: ProtectionMasks
    ) -> np.ndarray:
        """
        Apply protection masks to preserve original in protected regions.

        Args:
            original: Original BGR image
            transferred: Transferred BGR image
            masks: Protection masks

        Returns:
            Blended image with protections applied
        """
        # Resize transferred image if dimensions don't match original
        # (neural networks may change dimensions due to padding/pooling)
        if transferred.shape[:2] != original.shape[:2]:
            transferred = cv2.resize(
                transferred,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        result = transferred.astype(np.float32)
        original_f = original.astype(np.float32)

        # Apply skin protection (50% blend back)
        if masks.has_skin():
            skin_mask = masks.skin[:, :, np.newaxis]
            skin_blend = 0.5  # Keep 50% original in skin areas
            result = result * (1 - skin_mask * skin_blend) + \
                     original_f * (skin_mask * skin_blend)

        # Apply neon protection (use method-specific blend factor)
        # Note: NILUT sets neon_blend_factor=0 as it handles neon internally
        if masks.has_neon() and self.neon_blend_factor > 0:
            neon_mask = masks.neon[:, :, np.newaxis]
            neon_blend = self.neon_blend_factor
            result = result * (1 - neon_mask * neon_blend) + \
                     original_f * (neon_mask * neon_blend)

        # Apply lip protection (95% blend back - strong preservation)
        if masks.has_lips():
            lip_mask = masks.lips[:, :, np.newaxis]
            lip_blend = 0.95
            result = result * (1 - lip_mask * lip_blend) + \
                     original_f * (lip_mask * lip_blend)

        # Apply eye protection (90% blend back - preserve iris color)
        if masks.has_eyes():
            eye_mask = masks.eyes[:, :, np.newaxis]
            eye_blend = 0.90
            result = result * (1 - eye_mask * eye_blend) + \
                     original_f * (eye_mask * eye_blend)

        return np.clip(result, 0, 255).astype(np.uint8)
