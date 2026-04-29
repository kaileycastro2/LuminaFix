"""
Reinhard Style Transfer - LAB mean-shift color transfer.

Wraps the existing ColorTransfer implementation with the new interface.
"""

from typing import Optional
import numpy as np

from .abstract_transfer import AbstractTransfer
from .base import ProtectionMasks
from .registry import TransferRegistry

# Import existing implementations
from ..style_extractor import StyleExtractor, StyleProfile
from ..color_transfer import ColorTransfer


@TransferRegistry.register("reinhard")
class ReinhardTransfer(AbstractTransfer):
    """
    Reinhard LAB mean-shift style transfer.

    This is the classic, fast method that works by shifting
    the mean of LAB color channels from target to match reference.

    Features:
    - Fast CPU-based processing
    - No external dependencies
    - Always available
    - Built-in CLAHE preprocessing
    """

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 1.0
    ):
        """
        Initialize Reinhard transfer.

        Args:
            color_strength: Strength of A/B channel transfer (0-1)
            luminance_strength: Strength of L channel transfer (0-1)
            apply_clahe: Apply CLAHE for exposure normalization
            clahe_clip_limit: CLAHE clip limit
        """
        super().__init__(color_strength, luminance_strength)

        self._apply_clahe = apply_clahe
        self._clahe_clip_limit = clahe_clip_limit

        # Initialize components
        self._style_extractor = StyleExtractor(store_reference=True)
        self._color_transfer: Optional[ColorTransfer] = None
        self._style_profile: Optional[StyleProfile] = None

    @property
    def method_id(self) -> str:
        return "reinhard"

    @property
    def method_name(self) -> str:
        return "Reinhard (LAB)"

    @property
    def method_type(self) -> str:
        return "classic"

    def is_available(self) -> bool:
        """Always available - no external dependencies."""
        return True

    def load_reference(self, image: np.ndarray) -> None:
        """
        Extract style profile from reference image.

        Args:
            image: BGR reference image
        """
        self._style_profile = self._style_extractor.extract(image)

        # Create color transfer instance with current settings
        self._color_transfer = ColorTransfer(
            color_strength=self._color_strength,
            luminance_strength=self._luminance_strength,
            apply_clahe=self._apply_clahe,
            clahe_clip_limit=self._clahe_clip_limit,
            neon_threshold=0.6,
            neon_protection_strength=0.8
        )

        self._reference_loaded = True

    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply Reinhard color transfer.

        Args:
            image: BGR target image
            strength: Transfer strength (modulates color_strength)

        Returns:
            Transferred BGR image
        """
        if self._color_transfer is None or self._style_profile is None:
            raise RuntimeError("Reference not loaded")

        # Update color strength based on provided strength
        effective_strength = self._color_strength * strength
        self._color_transfer.color_strength = effective_strength

        # Apply transfer without protection (we handle it in abstract base)
        result = self._color_transfer.apply(
            target=image,
            style=self._style_profile,
            skin_mask=None,  # We handle protection in AbstractTransfer
            enable_neon_protection=False  # We handle this in AbstractTransfer
        )

        return result

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Reinhard uses CLAHE in its internal pipeline,
        so no additional preprocessing needed here.
        """
        return image

    def update_settings(
        self,
        color_strength: Optional[float] = None,
        luminance_strength: Optional[float] = None
    ) -> None:
        """
        Update transfer settings.

        Args:
            color_strength: New color strength (0-1)
            luminance_strength: New luminance strength (0-1)
        """
        if color_strength is not None:
            self._color_strength = color_strength
        if luminance_strength is not None:
            self._luminance_strength = luminance_strength

        if self._color_transfer:
            self._color_transfer.color_strength = self._color_strength
            self._color_transfer.luminance_strength = self._luminance_strength
