"""
Lip Protection Module

Detects lip regions using MediaPipe FaceLandmarker to protect them
during color transfer. Lips have redder/pinker hues that fall
outside typical skin detection ranges.

If MediaPipe is unavailable, returns no mask (lips are not protected).
"""

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh lip landmark indices (outer lips contour)
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]


class LipProtection:
    """Detects lip regions using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        blur_kernel: int = 15,
        expand_pixels: int = 5,
        max_num_faces: int = 5
    ):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.expand_pixels = expand_pixels
        self.max_num_faces = max_num_faces

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip regions in image.

        Args:
            image: BGR image (uint8)

        Returns:
            Soft mask (float32, 0-1) where 1 = lip region,
            or None if MediaPipe unavailable / no faces found.
        """
        from .face_landmarker import detect_landmarks

        result = detect_landmarks(image)
        if result is None:
            logger.info("Lip detect: MediaPipe unavailable or no faces — skipping lip protection")
            return None

        logger.info("Lip detect: using MediaPipe FaceLandmarker (image %dx%d)", image.shape[1], image.shape[0])
        return self._detect_mediapipe(image, result)

    def _detect_mediapipe(self, image: np.ndarray, result) -> Optional[np.ndarray]:
        """Precise lip detection using MediaPipe FaceLandmarker landmarks."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for face_landmarks in result.face_landmarks:
            outer_pts = []
            for idx in OUTER_LIP_INDICES:
                lm = face_landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                outer_pts.append([x, y])

            outer_pts = np.array(outer_pts, dtype=np.int32)
            cv2.fillPoly(mask, [outer_pts], 255)

        return self._finalize_mask(mask, "MediaPipe", len(result.face_landmarks))

    def _finalize_mask(self, mask: np.ndarray, method: str, face_count: int) -> Optional[np.ndarray]:
        """Apply dilation, blur, and normalize the mask."""
        if self.expand_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.expand_pixels * 2 + 1, self.expand_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        soft_mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (self.blur_kernel, self.blur_kernel), 0
        ) / 255.0

        if soft_mask.max() == 0:
            return None

        logger.info(
            f"Lip mask ({method}): {face_count} face(s), "
            f"max={soft_mask.max():.3f}, "
            f"coverage={100 * (soft_mask > 0).sum() / soft_mask.size:.2f}%"
        )
        return soft_mask

    def release(self):
        """Release resources."""
        pass
