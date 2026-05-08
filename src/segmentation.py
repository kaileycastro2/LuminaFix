"""
Semantic segmentation for per-region NILUT tuning.

Uses SegFormer-B0 (ADE20K, 150 classes). For each target image we:
  1. Run SegFormer once at low resolution (downscaled to 512px max).
  2. Cache the per-pixel class label map keyed by image path.
  3. Expose:
     - top_segments(...) -> the N most prominent ADE20K classes in this image
       (ranked by pixel coverage), so the UI can build dynamic sliders.
     - build_strength_map_for_image(...) -> per-pixel NILUT strength map from
       a {class_name: strength} dict supplied by the UI.

No hardcoded bucket mapping — classes are surfaced as-is.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INFERENCE_MAX_SIDE = 512
_CACHE_MAX_ENTRIES = 8

# Reserved name for "all classes not listed by the UI" — surfaced as a single
# slider so users can affect everything else without listing 150 classes.
OTHER_BUCKET = "other"


class _LazyModel:
    """Lazy-loads SegFormer on first use; shared across calls."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._id2label: Dict[int, str] = {}
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return self._model is not None
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            self._processor = SegformerImageProcessor.from_pretrained(model_name)
            self._model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self._model.eval()
            # ADE20K id->label map embedded in model config
            self._id2label = {
                int(k): str(v).split(",")[0].strip().lower()
                for k, v in self._model.config.id2label.items()
            }
            self._loaded = True
            logger.info("SegFormer-B0 loaded (%d classes)", len(self._id2label))
            return True
        except Exception as e:
            logger.warning("SegFormer load failed (%s) — segmentation disabled", e)
            self._loaded = True
            self._processor = None
            self._model = None
            return False

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label


_LAZY = _LazyModel()


# In-memory cache: cache_key -> (label_map_HxW int32, original_h, original_w)
_CACHE: "OrderedDict[str, Tuple[np.ndarray, int, int]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def is_available() -> bool:
    return _LAZY.load()


def class_names() -> List[str]:
    """Return the full list of ADE20K class names known to the model."""
    if not _LAZY.load():
        return []
    return [_LAZY.id2label[i] for i in sorted(_LAZY.id2label)]


def _cache_key_for(image_path: Optional[str], image_bgr: np.ndarray) -> str:
    """Stable cache key based on image path + mtime, with shape fallback."""
    if image_path:
        try:
            st = os.stat(image_path)
            return f"{image_path}::{st.st_mtime_ns}::{st.st_size}"
        except OSError:
            pass
    h, w = image_bgr.shape[:2]
    return f"shape::{h}x{w}::{int(image_bgr.sum())}"


def _run_inference(image_bgr: np.ndarray) -> np.ndarray:
    """Run SegFormer; return HxW int32 class-id map at the ORIGINAL image size."""
    import torch

    h, w = image_bgr.shape[:2]
    long_side = max(h, w)
    if long_side > _INFERENCE_MAX_SIDE:
        scale = _INFERENCE_MAX_SIDE / long_side
        small = cv2.resize(
            image_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    inputs = _LAZY.processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = _LAZY.model(**inputs).logits  # [1, C, H', W']
    upsampled = torch.nn.functional.interpolate(
        logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred_small = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    if pred_small.shape != (h, w):
        # Use nearest to preserve discrete class ids when scaling up.
        pred_full = cv2.resize(
            pred_small.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
    else:
        pred_full = pred_small
    return pred_full


def _get_or_compute_label_map(
    image_bgr: np.ndarray,
    image_path: Optional[str],
) -> Optional[np.ndarray]:
    """Cached label map for this image. Returns None if segmentation unavailable."""
    if not _LAZY.load():
        return None

    key = _cache_key_for(image_path, image_bgr)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            label_map, _, _ = cached
            _CACHE.move_to_end(key)
            return label_map

    try:
        label_map = _run_inference(image_bgr)
    except Exception as e:
        logger.warning("SegFormer inference failed: %s", e)
        return None

    with _CACHE_LOCK:
        h, w = label_map.shape
        _CACHE[key] = (label_map, h, w)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return label_map


def top_segments(
    image_bgr: np.ndarray,
    image_path: Optional[str] = None,
    max_segments: int = 8,
    min_pixel_pct: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Return the most prominent ADE20K classes in this image.

    Args:
        image_bgr: HxWx3 uint8 BGR image.
        image_path: optional path used as cache key.
        max_segments: keep at most this many top classes.
        min_pixel_pct: drop classes covering less than this percent of pixels.

    Returns:
        List of dicts: [{"name": "tree", "pixel_pct": 14.2}, ...] sorted descending.
        An "other" entry is appended if any pixels weren't included.
    """
    label_map = _get_or_compute_label_map(image_bgr, image_path)
    if label_map is None:
        return []

    total = label_map.size
    ids, counts = np.unique(label_map, return_counts=True)
    pcts = (counts / total) * 100.0
    order = np.argsort(-pcts)

    id2label = _LAZY.id2label
    picked: List[Dict[str, float]] = []
    used_pixel_count = 0
    for idx in order:
        cid = int(ids[idx])
        pct = float(pcts[idx])
        if pct < min_pixel_pct:
            continue
        name = id2label.get(cid)
        if not name:
            continue
        picked.append({"name": name, "pixel_pct": round(pct, 1)})
        used_pixel_count += int(counts[idx])
        if len(picked) >= max_segments:
            break

    other_pct = round((1.0 - used_pixel_count / total) * 100.0, 1)
    if other_pct > 0.5:
        picked.append({"name": OTHER_BUCKET, "pixel_pct": other_pct})

    return picked


def build_strength_map_for_image(
    image_bgr: np.ndarray,
    per_segment_strengths: Dict[str, float],
    default_strength: float = 1.0,
    image_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Build an HxW float32 strength map using ADE20K class names from the UI.

    Per-segment values are ABSOLUTE — they directly become the NILUT blend
    weight for matching pixels and are NOT scaled by a global multiplier.

    The `default_strength` (typically the global "Filter Strength" slider) is
    only applied to pixels that don't match any named segment — i.e. it acts
    as the strength for the "Other" / unmatched region.

    Args:
        image_bgr: target image (BGR).
        per_segment_strengths: {"tree": 0.2, "sky": 1.0, ...}. Absolute weights.
            An "other" key here is ignored (the global slider controls that).
        default_strength: NILUT strength for pixels not in any named segment.
        image_path: cache key.

    Returns:
        HxW float32 array, or None if segmentation unavailable.
    """
    label_map = _get_or_compute_label_map(image_bgr, image_path)
    if label_map is None:
        return None

    id2label = _LAZY.id2label
    # "Other" / unmatched pixels follow the global slider only.
    out = np.full(label_map.shape, float(default_strength), dtype=np.float32)

    name_to_weight: Dict[str, float] = {}
    for name, val in per_segment_strengths.items():
        if name == OTHER_BUCKET:
            continue  # global slider owns this; per-segment "other" is ignored
        try:
            name_to_weight[str(name).lower()] = float(val)
        except (TypeError, ValueError):
            continue

    for class_id, label in id2label.items():
        w = name_to_weight.get(label)
        if w is None:
            continue
        out[label_map == class_id] = w

    return out
