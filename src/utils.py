"""Shared utility functions for the LuminaFix project."""

import logging
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def parse_bool(value: str) -> bool:
    """Parse a string into a boolean. Handles form-data strings from the frontend."""
    return value.lower() in ('true', '1', 'yes', 'on')


def load_image_as_cv2(path: Path) -> Optional[np.ndarray]:
    """Load image with PIL (supports AVIF and more formats) and convert to OpenCV BGR."""
    from PIL import Image
    try:
        import pillow_avif
    except ImportError:
        pass

    try:
        pil_img = Image.open(path).convert('RGB')
        cv2_img = np.array(pil_img)
        return cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning("Failed to load image %s: %s", path, e)
        return None


def find_images(directory: Path, recursive: bool = False) -> List[Path]:
    """Find all supported image files in a directory."""
    pattern = "**/*" if recursive else "*"
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.glob(f"{pattern}{ext}"))
        images.extend(directory.glob(f"{pattern}{ext.upper()}"))
    return sorted(set(images))
