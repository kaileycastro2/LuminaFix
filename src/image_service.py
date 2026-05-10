"""
Image Service - Handles image file operations for the web layer.
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import get_config
from .utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class ImageService:
    """Service for image upload, save, and cleanup operations."""

    def __init__(self):
        config = get_config()
        self._upload_dir = config.web.upload_dir
        self._processed_dir = config.web.processed_dir
        self._reference_dir = config.web.reference_dir
        self._user_reference_dir = config.web.user_reference_dir

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    @property
    def processed_dir(self) -> Path:
        return self._processed_dir

    @property
    def reference_dir(self) -> Path:
        return self._reference_dir

    @property
    def user_reference_dir(self) -> Path:
        return self._user_reference_dir

    def save_upload(self, filename: str, contents: bytes) -> str:
        """Save uploaded file, return unique filename."""
        unique_id = uuid.uuid4().hex[:8]
        new_filename = f"{unique_id}_{filename}"
        file_path = self._upload_dir / new_filename
        file_path.write_bytes(contents)
        return new_filename

    def save_reference_upload(self, filename: str, contents: bytes) -> str:
        """Save uploaded reference, return unique filename."""
        suffix = Path(filename).suffix.lower()
        unique_id = uuid.uuid4().hex[:8]
        safe_name = Path(filename).stem.replace(" ", "_")
        new_filename = f"{safe_name}_{unique_id}{suffix}"
        file_path = self._user_reference_dir / new_filename
        file_path.write_bytes(contents)
        return new_filename

    def save_processed(self, image, method_id: str, ref_name: str, target_name: str) -> str:
        """Save a processed image as 16-bit PNG to preserve gradient precision."""
        output_filename = f"{method_id}_{ref_name}_{target_name}_{uuid.uuid4().hex[:6]}.png"
        output_path = self._processed_dir / output_filename
        if image.dtype == np.uint8:
            img16 = image.astype(np.uint16) * 257
        else:
            img16 = np.clip(image, 0, 255).astype(np.float32) * 257.0
            img16 = np.clip(img16, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), img16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return output_filename

    def resolve_reference_path(self, filename: str) -> Optional[Path]:
        """Find reference in preset, user-uploaded, or training_data category directories."""
        path = self._reference_dir / filename
        if path.exists():
            return path
        path = self._user_reference_dir / filename
        if path.exists():
            return path
        # Search in training_data/reference category subfolders
        training_ref_dir = self._reference_dir.parent / "training_data" / "reference"
        if training_ref_dir.exists():
            for category_dir in training_ref_dir.iterdir():
                if category_dir.is_dir():
                    path = category_dir / filename
                    if path.exists():
                        return path
        return None

    def resolve_upload_or_processed(self, filename: str) -> Optional[Path]:
        """Find file in uploads or processed directories."""
        for directory in (self._upload_dir, self._processed_dir):
            path = directory / filename
            if path.exists():
                return path
        return None

    def cleanup(self, filename: str) -> None:
        """Delete file from uploads and processed directories."""
        for directory in (self._upload_dir, self._processed_dir):
            path = directory / filename
            if path.exists():
                path.unlink()

    def validate_image_extension(self, filename: str) -> bool:
        """Check if a filename has a supported image extension."""
        return Path(filename).suffix.lower() in IMAGE_EXTENSIONS
