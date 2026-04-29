"""
Lip Protection Module

Detects lip regions using MediaPipe FaceLandmarker to protect them
during color transfer. Lips have redder/pinker hues that fall
outside typical skin detection ranges.

Falls back to OpenCV Haar cascade if MediaPipe is unavailable.
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
    """Detects lip regions using MediaPipe FaceLandmarker (with OpenCV fallback)."""

    def __init__(
        self,
        blur_kernel: int = 15,
        expand_pixels: int = 5,
        max_num_faces: int = 5
    ):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.expand_pixels = expand_pixels
        self.max_num_faces = max_num_faces
        self._face_cascade = None

    def _get_face_cascade(self):
        """Lazy-initialize OpenCV Haar cascade (fallback)."""
        if self._face_cascade is None:
            path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(path)
        return self._face_cascade

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lip regions in image.

        Args:
            image: BGR image (uint8)

        Returns:
            Soft mask (float32, 0-1) where 1 = lip region,
            or None if no face detected.
        """
        from .face_landmarker import detect_landmarks

        result = detect_landmarks(image)
        if result is not None:
            logger.info("Lip detect: using MediaPipe FaceLandmarker (image %dx%d)", image.shape[1], image.shape[0])
            return self._detect_mediapipe(image, result)

        logger.info("Lip detect: using OpenCV fallback (mediapipe unavailable or no faces)")
        return self._detect_opencv_fallback(image)

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

    def _detect_opencv_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback lip detection using Haar cascade + geometric estimation."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = self._get_face_cascade()
        min_size = max(30, int(min(w, h) * 0.05))

        try:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(min_size, min_size)
            )
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None

        if len(faces) == 0:
            logger.debug("No faces detected for lip protection (fallback)")
            return None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        faces = faces[:self.max_num_faces]

        mask = np.zeros((h, w), dtype=np.uint8)

        for (fx, fy, fw, fh) in faces:
            lip_center_x = int(fx + fw * 0.50)
            lip_center_y = int(fy + fh * 0.71)
            lip_half_w = max(int(fw * 0.24), 5)
            lip_half_h = max(int(fh * 0.08), 3)

            cv2.ellipse(
                mask, (lip_center_x, lip_center_y),
                (lip_half_w, lip_half_h), 0, 0, 360, 255, -1
            )

        return self._finalize_mask(mask, "OpenCV fallback", len(faces))

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
        self._face_cascade = None
