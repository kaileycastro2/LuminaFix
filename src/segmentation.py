"""
Semantic segmentation for per-region NILUT tuning.

Uses SegFormer-B0 (ADE20K) to label each pixel as one of:
sky, grass, building, skin, other.

Inference is run on a downscaled copy (max 512px) to control memory;
masks are upsampled back to the original image size.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Segment names exposed to the rest of the system.
SEGMENT_NAMES = ("sky", "grass", "building", "skin", "other")

# ADE20K class IDs grouped into our 5 buckets.
# Reference: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
_SKY_IDS = {2}
_GRASS_IDS = {9, 17, 29, 66}        # grass, plant, field, tree(low) — foliage-ish
_TREE_IDS = {4}                      # tree (full) — counted as grass for warmth control
_BUILDING_IDS = {1, 25, 84}          # building, house, tower
_SKIN_IDS = {12, 126}                # person, animal — handled like skin
_ADE_TO_BUCKET = {}
for _id in _SKY_IDS:
    _ADE_TO_BUCKET[_id] = "sky"
for _id in _GRASS_IDS | _TREE_IDS:
    _ADE_TO_BUCKET[_id] = "grass"
for _id in _BUILDING_IDS:
    _ADE_TO_BUCKET[_id] = "building"
for _id in _SKIN_IDS:
    _ADE_TO_BUCKET[_id] = "skin"

_INFERENCE_MAX_SIDE = 512


class _LazyModel:
    """Lazy-loads SegFormer on first use; shared across calls."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
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
            self._loaded = True
            logger.info("SegFormer-B0 loaded")
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


_LAZY = _LazyModel()


def is_available() -> bool:
    """Check if SegFormer can be loaded (transformers + model)."""
    return _LAZY.load()


def segment(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segment a BGR image into 5 region masks.

    Args:
        image_bgr: HxWx3 uint8 BGR image.

    Returns:
        Dict mapping segment name -> HxW float32 mask in [0, 1].
        Masks are mutually exclusive (each pixel belongs to one bucket);
        "other" catches everything not in sky/grass/building/skin.
        If segmentation fails, returns all zeros except "other"=1.
    """
    h, w = image_bgr.shape[:2]
    empty: Dict[str, np.ndarray] = {
        name: np.zeros((h, w), dtype=np.float32) for name in SEGMENT_NAMES
    }

    if not _LAZY.load():
        empty["other"][:] = 1.0
        return empty

    try:
        import torch
    except ImportError:
        empty["other"][:] = 1.0
        return empty

    # Downscale for inference
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

    # SegFormer expects RGB
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    inputs = _LAZY.processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = _LAZY.model(**inputs).logits  # [1, C, H', W']

    # Upsample logits to small image size, then argmax
    upsampled = torch.nn.functional.interpolate(
        logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred_small = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # H'xW'

    # Map class IDs to bucket index per pixel
    bucket_index = np.full(pred_small.shape, len(SEGMENT_NAMES) - 1, dtype=np.int32)  # default "other"
    for class_id, bucket_name in _ADE_TO_BUCKET.items():
        b = SEGMENT_NAMES.index(bucket_name)
        bucket_index[pred_small == class_id] = b

    # Build per-bucket binary mask at small res, then upsample to full res
    masks: Dict[str, np.ndarray] = {}
    for i, name in enumerate(SEGMENT_NAMES):
        small_mask = (bucket_index == i).astype(np.uint8) * 255
        if small_mask.shape != (h, w):
            full = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            full = small_mask
        masks[name] = (full.astype(np.float32) / 255.0)

    # Renormalize so masks sum to 1 per pixel (handles bilinear bleed at edges)
    stacked = np.stack([masks[n] for n in SEGMENT_NAMES], axis=0)
    total = stacked.sum(axis=0, keepdims=True)
    total = np.where(total < 1e-6, 1.0, total)
    stacked = stacked / total
    for i, name in enumerate(SEGMENT_NAMES):
        masks[name] = stacked[i]

    return masks


def build_strength_map(
    masks: Dict[str, np.ndarray],
    per_segment_strengths: Dict[str, float],
    default_strength: float = 1.0,
) -> np.ndarray:
    """
    Combine per-segment masks + scalar strengths into a per-pixel strength map.

    Args:
        masks: dict from segment() — values are HxW float32 in [0,1] summing to 1.
        per_segment_strengths: e.g. {"sky": 1.0, "grass": 0.2, "skin": 0.3, ...}.
            Missing keys fall back to default_strength.
        default_strength: applied to segments not listed in per_segment_strengths.

    Returns:
        HxW float32 array; each pixel's value is the strength to use there.
    """
    sample = next(iter(masks.values()))
    out = np.zeros_like(sample, dtype=np.float32)
    for name, mask in masks.items():
        s = float(per_segment_strengths.get(name, default_strength))
        out += mask * s
    return out
