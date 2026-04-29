"""
Transfers layer - Domain layer for style transfer methods.

Contains the protocol, base classes, registry, and concrete implementations.
"""

from .base import StyleTransferProtocol, TransferResult, ProtectionMasks
from .abstract_transfer import AbstractTransfer
from .registry import TransferRegistry, get_registry

__all__ = [
    "StyleTransferProtocol",
    "TransferResult",
    "ProtectionMasks",
    "AbstractTransfer",
    "TransferRegistry",
    "get_registry",
]
