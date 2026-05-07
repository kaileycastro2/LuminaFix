"""
Eye Protection Module

Detects eye regions using MediaPipe FaceLandmarker to protect them
during color transfer. Eyes contain distinctive colors (iris) that
can become distorted during style transfer.

If MediaPipe is unavailable, returns no mask (eyes are not protected).
"""

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh eye contour landmark indices
LEFT_EYE_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7
]

RIGHT_EYE_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382
]

# Iris landmarks (available when refine_landmarks=True)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


class EyeProtection:
    """Detects eye regions using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        blur_kernel: int = 11,
        expand_pixels: int = 8,
        max_num_faces: int = 5
    ):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.expand_pixels = expand_pixels
        self.max_num_faces = max_num_faces

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect eye regions in image.

        Args:
            image: BGR image (uint8)

        Returns:
            Soft mask (float32, 0-1) where 1 = eye region,
            or None if MediaPipe unavailable / no faces found.
        """
        from .face_landmarker import detect_landmarks

        result = detect_landmarks(image)
        if result is None:
            logger.debug("Eye detect: MediaPipe unavailable or no faces — skipping eye protection")
            return None
        return self._detect_mediapipe(image, result)

    def _detect_mediapipe(self, image: np.ndarray, result) -> Optional[np.ndarray]:
        """Precise eye detection using MediaPipe FaceLandmarker landmarks."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for face_landmarks in result.face_landmarks:
            for eye_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
                pts = []
                for idx in eye_indices:
                    lm = face_landmarks[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append([x, y])
                pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            # Also fill iris regions for extra coverage
            for iris_indices in [LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES]:
                pts = []
                for idx in iris_indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        pts.append([x, y])
                if len(pts) >= 3:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.fillConvexPoly(mask, pts, 255)

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
            f"Eye mask ({method}): {face_count} face(s), "
            f"max={soft_mask.max():.3f}, "
            f"coverage={100 * (soft_mask > 0).sum() / soft_mask.size:.2f}%"
        )
        return soft_mask

    def release(self):
        """Release resources."""
        pass
