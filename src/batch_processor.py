"""
Batch Processor Module

Handles batch processing of multiple images with consistent style transfer.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass

from .style_extractor import StyleExtractor, StyleProfile
from .color_transfer import ColorTransfer
from .skin_protection import SkinProtection
from .neon_protection import NeonProtection

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    input_path: str
    output_path: str
    success: bool
    error: Optional[str] = None


class BatchProcessor:
    """Processes multiple images with consistent style transfer."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        color_strength: float = 0.8,
        luminance_strength: float = 0.5,
        enable_skin_protection: bool = True,
        enable_neon_protection: bool = True,
        enable_lip_protection: bool = False,
        enable_eye_protection: bool = True,
        jpeg_quality: int = 95,
        random_seed: int = 42
    ):
        """
        Args:
            color_strength: Strength of color transfer (0-1)
            luminance_strength: Strength of luminance transfer (0-1)
            enable_skin_protection: Enable skin tone protection
            enable_neon_protection: Enable neon/saturation protection
            enable_lip_protection: Enable lip region protection (MediaPipe)
            enable_eye_protection: Enable eye region protection (MediaPipe)
            jpeg_quality: JPEG output quality (1-100)
            random_seed: Seed for deterministic results
        """
        # Set random seed for determinism
        np.random.seed(random_seed)

        self.color_strength = color_strength
        self.luminance_strength = luminance_strength
        self.enable_skin_protection = enable_skin_protection
        self.enable_neon_protection = enable_neon_protection
        self.enable_lip_protection = enable_lip_protection
        self.enable_eye_protection = enable_eye_protection
        self.jpeg_quality = jpeg_quality

        # Initialize components
        self.style_extractor = StyleExtractor(store_reference=True)
        self.color_transfer = ColorTransfer(
            color_strength=color_strength,
            luminance_strength=luminance_strength,
            apply_clahe=True,
            clahe_clip_limit=0.2,
            neon_threshold=0.3,
            neon_protection_strength=1.0
        )
        self.skin_protection = SkinProtection(use_wide_range=True)
        self.neon_protection = NeonProtection(
            saturation_threshold=0.75,
            saturation_cap=0.90,
            soft_clamp=True
        )

        self.style_profile: Optional[StyleProfile] = None

    def load_reference(self, reference_path: str) -> StyleProfile:
        """
        Load and extract style from reference image.

        Args:
            reference_path: Path to reference image

        Returns:
            Extracted StyleProfile
        """
        logger.info("Loading reference: %s", reference_path)
        self.style_profile = self.style_extractor.extract_from_path(reference_path)

        logger.info("  LAB: L=%.1f, A=%.1f, B=%.1f",
                     self.style_profile.l_mean,
                     self.style_profile.a_mean,
                     self.style_profile.b_mean)
        logger.info("  Saturation: %.1f", self.style_profile.saturation_mean)
        logger.info("  Contrast: %.1f", self.style_profile.contrast)

        return self.style_profile

    def process_single(
        self,
        image: np.ndarray,
        style: Optional[StyleProfile] = None
    ) -> np.ndarray:
        """
        Process a single image with style transfer.

        Args:
            image: BGR input image
            style: StyleProfile to apply (uses loaded profile if None)

        Returns:
            Processed BGR image
        """
        if style is None:
            style = self.style_profile

        if style is None:
            raise ValueError("No style profile loaded. Call load_reference() first.")

        # Step 1: Detect skin regions (if enabled)
        skin_mask = None
        if self.enable_skin_protection:
            skin_mask = self.skin_protection.detect(image)

        # Step 2: Apply color transfer with integrated skin + neon protection
        result = self.color_transfer.apply(
            image,
            style,
            skin_mask,
            enable_neon_protection=self.enable_neon_protection
        )

        return result

    def process_batch(
        self,
        target_paths: List[str],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple images with the loaded style.

        Args:
            target_paths: List of paths to target images
            output_dir: Directory to save outputs
            progress_callback: Optional callback(current, total, filename)

        Returns:
            List of ProcessingResult for each image
        """
        if self.style_profile is None:
            raise ValueError("No style profile loaded. Call load_reference() first.")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(target_paths)

        logger.info("Processing %d images...", total)

        for i, target_path in enumerate(target_paths, 1):
            filename = Path(target_path).name

            if progress_callback:
                progress_callback(i, total, filename)

            logger.info("[%d/%d] Processing: %s", i, total, filename)

            try:
                # Load image
                image = cv2.imread(target_path)
                if image is None:
                    raise FileNotFoundError(f"Could not load: {target_path}")

                # Process
                result_image = self.process_single(image)

                # Save output
                output_filename = f"edited_{Path(target_path).stem}.jpg"
                output_file = output_path / output_filename

                cv2.imwrite(
                    str(output_file),
                    result_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )

                results.append(ProcessingResult(
                    input_path=target_path,
                    output_path=str(output_file),
                    success=True
                ))
                logger.info("         Saved: %s", output_filename)

            except Exception as e:
                results.append(ProcessingResult(
                    input_path=target_path,
                    output_path="",
                    success=False,
                    error=str(e)
                ))
                logger.error("         ERROR: %s", e)

        success_count = sum(1 for r in results if r.success)
        logger.info("Completed: %d/%d images processed successfully", success_count, total)

        return results

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False
    ) -> List[ProcessingResult]:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            recursive: Search subdirectories

        Returns:
            List of ProcessingResult
        """
        input_path = Path(input_dir)

        # Find all image files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        target_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            target_paths.extend(input_path.glob(f"{pattern}{ext}"))
            target_paths.extend(input_path.glob(f"{pattern}{ext.upper()}"))

        # Sort for deterministic order
        target_paths = sorted(set(target_paths))

        return self.process_batch(
            [str(p) for p in target_paths],
            output_dir
        )

    @staticmethod
    def find_images(directory: str, recursive: bool = False) -> List[str]:
        """
        Find all supported image files in directory.

        Args:
            directory: Directory to search
            recursive: Search subdirectories

        Returns:
            List of image paths
        """
        dir_path = Path(directory)
        pattern = "**/*" if recursive else "*"

        images = []
        for ext in BatchProcessor.SUPPORTED_EXTENSIONS:
            images.extend(dir_path.glob(f"{pattern}{ext}"))
            images.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

        return sorted(set(str(p) for p in images))
