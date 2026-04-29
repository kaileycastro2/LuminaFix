"""
Device Manager - Singleton for GPU/CPU detection and management.

Handles PyTorch device selection with fallback logic.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Singleton manager for compute device selection.

    Automatically detects available hardware (CUDA, MPS, CPU)
    and provides a consistent interface for device selection.
    """

    _instance: Optional["DeviceManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if DeviceManager._initialized:
            return

        self._device = None
        self._device_type: str = "cpu"
        self._torch_available: bool = False
        self._cuda_available: bool = False
        self._mps_available: bool = False

        self._detect_capabilities()
        DeviceManager._initialized = True

    def _detect_capabilities(self) -> None:
        """Detect available compute capabilities."""
        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            self._mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )

            # Auto-select best device
            if self._cuda_available:
                self._device = torch.device("cuda")
                self._device_type = "cuda"
                logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
            elif self._mps_available:
                self._device = torch.device("mps")
                self._device_type = "mps"
                logger.info("Apple MPS device detected")
            else:
                self._device = torch.device("cpu")
                self._device_type = "cpu"
                logger.info("Using CPU device")

        except ImportError:
            self._torch_available = False
            self._device_type = "cpu"
            logger.warning("PyTorch not available - neural methods will be disabled")

    @property
    def device(self) -> Any:
        """Get the selected torch device."""
        if not self._torch_available:
            raise RuntimeError("PyTorch is not installed")
        return self._device

    @property
    def device_type(self) -> str:
        """Get device type string (cuda, mps, cpu)."""
        return self._device_type

    @property
    def is_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        return self._torch_available

    @property
    def is_gpu_available(self) -> bool:
        """Check if any GPU (CUDA or MPS) is available."""
        return self._cuda_available or self._mps_available

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "torch_available": self._torch_available,
            "device_type": self._device_type,
            "cuda_available": self._cuda_available,
            "mps_available": self._mps_available,
        }

        if self._torch_available and self._cuda_available:
            import torch
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)

        return info

    def set_device(self, device_type: str) -> None:
        """
        Manually set device type.

        Args:
            device_type: One of "auto", "cuda", "mps", "cpu"
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch is not installed")

        import torch

        if device_type == "auto":
            self._detect_capabilities()
        elif device_type == "cuda":
            if not self._cuda_available:
                raise RuntimeError("CUDA is not available")
            self._device = torch.device("cuda")
            self._device_type = "cuda"
        elif device_type == "mps":
            if not self._mps_available:
                raise RuntimeError("MPS is not available")
            self._device = torch.device("mps")
            self._device_type = "mps"
        elif device_type == "cpu":
            self._device = torch.device("cpu")
            self._device_type = "cpu"
        else:
            raise ValueError(f"Unknown device type: {device_type}")


# Module-level singleton accessor
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
