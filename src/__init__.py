"""
LuminaFix Style Transfer
"""

# Core components
from .style_extractor import StyleExtractor
from .color_transfer import ColorTransfer
from .batch_processor import BatchProcessor

# Protection modules
from .skin_protection import SkinProtection
from .neon_protection import NeonProtection
from .lip_protection import LipProtection

# Export
from .xmp_generator import XMPPresetGenerator, LightroomParams
from .xmp_strategies import (
    XMPExtractionStrategy, BasicStrategy, ColorScienceStrategy,
    OptimizationStrategy, STRATEGIES,
)
from .export_service import ExportService, ExportItem

# Utilities
from .utils import parse_bool, load_image_as_cv2, IMAGE_EXTENSIONS

__all__ = [
    # Core
    'StyleExtractor',
    'ColorTransfer',
    'BatchProcessor',
    # Protections
    'SkinProtection',
    'NeonProtection',
    'LipProtection',
    # Export
    'XMPPresetGenerator',
    'LightroomParams',
    'XMPExtractionStrategy',
    'BasicStrategy',
    'ColorScienceStrategy',
    'OptimizationStrategy',
    'STRATEGIES',
    'ExportService',
    'ExportItem',
    # Utilities
    'parse_bool',
    'load_image_as_cv2',
    'IMAGE_EXTENSIONS',
]
