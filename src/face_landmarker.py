"""
Shared MediaPipe FaceLandmarker singleton.

MediaPipe 0.10.x+ removed the legacy mp.solutions API. The new API uses
mediapipe.tasks.vision.FaceLandmarker with a downloaded .task model file.

Both lip_protection and eye_protection import from here to share a single
FaceLandmarker instance.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "face_landmarker.task"
_MODEL_SEARCH_PATHS = [
    Path(__file__).resolve().parent.parent / "models" / _MODEL_FILENAME,
    Path("/app/models") / _MODEL_FILENAME,
]

_landmarker = None
_landmarker_failed = False


def _find_model_path() -> Optional[Path]:
    for p in _MODEL_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def get_face_landmarker():
    """Return a shared FaceLandmarker instance (lazy-initialized)."""
    global _landmarker, _landmarker_failed
    if _landmarker_failed:
        return None
    if _landmarker is not None:
        return _landmarker

    model_path = _find_model_path()
    if model_path is None:
        logger.error(
            "FaceLandmarker model not found. Searched: %s",
            [str(p) for p in _MODEL_SEARCH_PATHS],
        )
        _landmarker_failed = True
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker loaded from %s", model_path)
        return _landmarker
    except Exception as e:
        logger.exception("Failed to create FaceLandmarker: %s", e)
        _landmarker_failed = True
        return None


def detect_landmarks(image: np.ndarray):
    """Run face landmark detection on a BGR image.

    Returns:
        FaceLandmarkerResult or None if landmarker unavailable / no faces found.
    """
    landmarker = get_face_landmarker()
    if landmarker is None:
        return None

    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result
    except Exception as e:
        logger.warning("FaceLandmarker detection failed: %s", e)
        return None
