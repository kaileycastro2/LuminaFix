"""
Base protocol and data classes for style transfer methods.

Defines the interface that all transfer methods must implement.
"""

from typing import Protocol, Optional, runtime_checkable
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass(frozen=True)
class ProtectionMasks:
    """Immutable container for protection masks."""
    skin: Optional[np.ndarray] = None
    neon: Optional[np.ndarray] = None
    lips: Optional[np.ndarray] = None
    eyes: Optional[np.ndarray] = None

    def has_skin(self) -> bool:
        return self.skin is not None and self.skin.max() > 0

    def has_neon(self) -> bool:
        return self.neon is not None and self.neon.max() > 0

    def has_lips(self) -> bool:
        return self.lips is not None and self.lips.max() > 0

    def has_eyes(self) -> bool:
        return self.eyes is not None and self.eyes.max() > 0


@dataclass
class TransferResult:
    """Result from a style transfer operation."""
    image: Optional[np.ndarray]
    method_id: str
    method_name: str
    success: bool
    error: Optional[str] = None
    processing_time_ms: float = 0.0

    @classmethod
    def success_result(
        cls,
        image: np.ndarray,
        method_id: str,
        method_name: str,
        processing_time_ms: float
    ) -> "TransferResult":
        """Create a successful result."""
        return cls(
            image=image,
            method_id=method_id,
            method_name=method_name,
            success=True,
            error=None,
            processing_time_ms=processing_time_ms
        )

    @classmethod
    def error_result(
        cls,
        method_id: str,
        method_name: str,
        error: str
    ) -> "TransferResult":
        """Create an error result."""
        return cls(
            image=None,
            method_id=method_id,
            method_name=method_name,
            success=False,
            error=error,
            processing_time_ms=0.0
        )


@runtime_checkable
class StyleTransferProtocol(Protocol):
    """
    Protocol defining the style transfer interface.

    All style transfer methods must implement this protocol
    to be usable with the MultiMethodProcessor.
    """

    @property
    def method_id(self) -> str:
        """Unique identifier for this method (e.g., 'reinhard', 'msgnet')."""
        ...

    @property
    def method_name(self) -> str:
        """Human-readable name for display (e.g., 'Reinhard (LAB)')."""
        ...

    @property
    def method_type(self) -> str:
        """Type of method: 'classic' or 'neural'."""
        ...

    def is_available(self) -> bool:
        """Check if this method is available (dependencies met, models loaded, etc.)."""
        ...

    def load_reference(self, image: np.ndarray) -> None:
        """
        Load and prepare reference image/style.

        Args:
            image: BGR image as numpy array (uint8)
        """
        ...

    def transfer(
        self,
        target: np.ndarray,
        strength: float = 1.0,
        masks: Optional[ProtectionMasks] = None
    ) -> np.ndarray:
        """
        Apply style transfer to target image.

        Args:
            target: BGR target image as numpy array (uint8)
            strength: Transfer strength from 0.0 to 1.0
            masks: Optional protection masks for skin/neon regions

        Returns:
            Processed BGR image as numpy array (uint8)
        """
        ...


class TransferTimer:
    """Context manager for timing transfer operations."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TransferTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
