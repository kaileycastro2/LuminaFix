"""
Multi-Method Processor - Orchestrates style transfer with multiple methods.

Provides the main service for processing images with all available
style transfer methods and returning comparative results.
"""

from typing import Callable, Dict, List, Optional, Any, Tuple
import gc
import numpy as np
import logging

from .config import ProcessingConfig
from .transfers.base import TransferResult, ProtectionMasks, TransferTimer
from .transfers.registry import TransferRegistry
from .protection_service import ProtectionService

logger = logging.getLogger(__name__)

# NILUT variant definitions: (variant_key, enhancement_method_name, display_suffix)
# None for enhancement_method_name means base NILUT (no enhancement).
NILUT_VARIANT_DEFS: List[Tuple[str, Optional[str], str]] = [
    ("nilut", None, ""),
    ("nilut_contrast", "apply_clahe_enhancement", " + CLAHE"),
    ("nilut_tonecurve", "apply_tonecurve_enhancement", " + ToneCurve"),
    ("nilut_tonecurve_sat", "apply_tonecurve_saturation_enhancement", " + ToneCurve + Saturation"),
    ("nilut_chroma", None, " + Chroma"),  # Uses apply_chroma_boost with args
]

# All known NILUT variant keys
NILUT_VARIANT_KEYS = {vdef[0] for vdef in NILUT_VARIANT_DEFS}


class MultiMethodProcessor:
    """
    Orchestrates processing with multiple style transfer methods.

    This is the main service class that:
    1. Manages available transfer methods
    2. Computes protection masks once
    3. Processes images with each method
    4. Returns comparative results
    """

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0,
        enable_skin_protection: bool = True,
        enable_neon_protection: bool = True,
        enable_lip_protection: bool = False,
        enable_eye_protection: bool = True,
        config: Optional[ProcessingConfig] = None
    ):
        self._color_strength = color_strength
        self._luminance_strength = luminance_strength

        self._config = config or ProcessingConfig(
            color_strength=color_strength,
            luminance_strength=luminance_strength,
            enable_skin_protection=enable_skin_protection,
            enable_neon_protection=enable_neon_protection
        )

        self._protection = ProtectionService(
            enable_skin_protection=enable_skin_protection,
            enable_neon_protection=enable_neon_protection,
            enable_lip_protection=enable_lip_protection,
            enable_eye_protection=enable_eye_protection
        )

        self._methods_cache: Dict[str, Any] = {}
        self._reference_loaded: bool = False

    @property
    def protection_service(self) -> ProtectionService:
        """Expose protection service for external use (e.g., route handlers)."""
        return self._protection

    def _get_method(self, method_id: str):
        """Get or create a transfer method instance."""
        if method_id not in self._methods_cache:
            self._methods_cache[method_id] = TransferRegistry.get(
                method_id,
                color_strength=self._color_strength,
                luminance_strength=self._luminance_strength
            )
        return self._methods_cache[method_id]

    def _get_available_methods(self) -> List[Any]:
        """Get all available transfer method instances."""
        methods = []
        for method_id in ["reinhard", "adain", "nilut"]:
            try:
                method = self._get_method(method_id)
                if method.is_available():
                    methods.append(method)
                else:
                    logger.debug("Method %s not available", method_id)
            except Exception as e:
                logger.warning("Failed to get method %s: %s", method_id, e)
        return methods

    def load_reference(self, reference: np.ndarray) -> Dict[str, bool]:
        """Load reference image for all available methods."""
        results = {}
        for method in self._get_available_methods():
            try:
                method.load_reference(reference)
                results[method.method_id] = True
                logger.info("Loaded reference for %s", method.method_name)
            except Exception as e:
                results[method.method_id] = False
                logger.error("Failed to load reference for %s: %s", method.method_name, e)

        self._reference_loaded = any(results.values())
        return results

    def compute_masks(self, target: np.ndarray) -> ProtectionMasks:
        """Compute protection masks for a target image."""
        return self._protection.compute_masks(target)

    def process_single(
        self,
        method_id: str,
        target: np.ndarray,
        reference: np.ndarray,
        masks: Optional[ProtectionMasks] = None,
        reference_name: Optional[str] = None,
        nilut_mode: str = "per_reference"
    ) -> TransferResult:
        """Process with a single method."""
        try:
            method = self._get_method(method_id)

            if not method.is_available():
                return TransferResult.error_result(
                    method_id=method_id,
                    method_name=method.method_name,
                    error=f"{method.method_name} is not available"
                )

            # Load reference (with pre-trained model path for NILUT if available)
            if method_id == "nilut":
                use_universal = (nilut_mode == "universal")
                if use_universal:
                    method.load_reference(reference, use_universal=True)
                elif reference_name:
                    from pathlib import Path
                    model_path = Path(__file__).parent.parent.parent / "models" / "nilut" / f"{reference_name}.pt"
                    method.load_reference(reference, model_path=str(model_path) if model_path.exists() else None)
                else:
                    method.load_reference(reference)
            else:
                method.load_reference(reference)

            if masks is None:
                masks = self._protection.compute_masks(target)

            with TransferTimer() as timer:
                result_image = method.transfer(
                    target=target,
                    strength=self._color_strength,
                    masks=masks
                )

            return TransferResult.success_result(
                image=result_image,
                method_id=method_id,
                method_name=method.method_name,
                processing_time_ms=timer.elapsed_ms
            )

        except Exception as e:
            logger.exception("Error processing with %s", method_id)
            method = self._methods_cache.get(method_id)
            method_name = method.method_name if method else method_id
            return TransferResult.error_result(
                method_id=method_id,
                method_name=method_name,
                error=str(e)
            )

    def process_nilut_variants(
        self,
        target_image: np.ndarray,
        reference_image: np.ndarray,
        masks: ProtectionMasks,
        model_path: str,
        model_display_name: str,
        model_id: str,
        requested_variants: List[str],
        color_strength: float,
        per_segment_strengths: Optional[Dict[str, float]] = None,
        target_image_path: Optional[str] = None,
        curve_strength: float = 0.5,
        saturation_boost: float = 1.4,
    ) -> Dict[str, TransferResult]:
        """
        Process all requested NILUT variants for a single model version.

        This is the single authoritative implementation for NILUT variant
        processing. Both the web layer and process_all() delegate here.

        Args:
            target_image: BGR target image
            reference_image: BGR reference image
            masks: Pre-computed protection masks
            model_path: Path to the universal .pt model file
            model_display_name: Human-readable model name for results
            model_id: Raw model identifier (e.g., "latest", "20260123_010537")
            requested_variants: List like ["nilut", "nilut_contrast", ...]
            color_strength: Transfer strength

        Returns:
            Dict mapping result_key -> TransferResult
        """
        from .transfers.nilut_transfer import NILUTTransfer

        results: Dict[str, TransferResult] = {}

        nilut = NILUTTransfer(
            color_strength=color_strength,
            luminance_strength=self._luminance_strength
        )

        try:
            nilut.load_universal_model(model_path)
            nilut.load_reference(reference_image, use_universal=True)

            # If per-segment strengths supplied, build a per-pixel strength map
            # from ADE20K class names. Per-segment values are ABSOLUTE; the
            # global color_strength only applies to "other" / unmatched pixels.
            strength_map = None
            if per_segment_strengths:
                try:
                    from .segmentation import build_strength_map_for_image
                    strength_map = build_strength_map_for_image(
                        target_image,
                        per_segment_strengths,
                        default_strength=color_strength,
                        image_path=target_image_path,
                    )
                except Exception as seg_err:
                    logger.warning(
                        "Segmentation failed (%s) — falling back to scalar strength",
                        seg_err,
                    )
                    strength_map = None

            with TransferTimer() as timer:
                if strength_map is not None:
                    base_result = nilut.transfer_with_strength_map(
                        target=target_image,
                        strength_map=strength_map,
                        masks=masks,
                    )
                else:
                    base_result = nilut.transfer(
                        target=target_image, strength=color_strength, masks=masks
                    )

            for variant_key, enhance_method, display_suffix in NILUT_VARIANT_DEFS:
                if variant_key not in requested_variants:
                    continue

                result_key = f"{variant_key}_{model_id}"

                if variant_key == "nilut":
                    # Base NILUT - already computed
                    results[result_key] = TransferResult.success_result(
                        image=base_result.copy(),
                        method_id=result_key,
                        method_name=f"NILUT ({model_display_name})",
                        processing_time_ms=timer.elapsed_ms
                    )
                elif variant_key == "nilut_chroma":
                    # Chroma boost needs special args
                    enhanced = nilut.apply_chroma_boost(base_result.copy(), boost_strength=1.35)
                    enhanced = self._protection.apply_protection(target_image, enhanced, masks)
                    results[result_key] = TransferResult.success_result(
                        image=enhanced,
                        method_id=result_key,
                        method_name=f"NILUT{display_suffix} ({model_display_name})",
                        processing_time_ms=timer.elapsed_ms
                    )
                else:
                    # Standard enhancement via method name
                    enhance_fn = getattr(nilut, enhance_method)
                    if enhance_method == "apply_tonecurve_saturation_enhancement":
                        enhanced = enhance_fn(
                            base_result.copy(),
                            curve_strength=curve_strength,
                            saturation_boost=saturation_boost,
                        )
                    elif enhance_method == "apply_tonecurve_enhancement":
                        enhanced = enhance_fn(
                            base_result.copy(),
                            curve_strength=curve_strength,
                        )
                    else:
                        enhanced = enhance_fn(base_result.copy())
                    enhanced = self._protection.apply_protection(target_image, enhanced, masks)
                    results[result_key] = TransferResult.success_result(
                        image=enhanced,
                        method_id=result_key,
                        method_name=f"NILUT{display_suffix} ({model_display_name})",
                        processing_time_ms=timer.elapsed_ms
                    )

        except Exception as e:
            logger.exception("Error processing NILUT model %s", model_id)
            for variant_key in requested_variants:
                result_key = f"{variant_key}_{model_id}"
                display = variant_key.replace("_", " ").title()
                results[result_key] = TransferResult.error_result(
                    method_id=result_key,
                    method_name=f"{display} ({model_display_name})",
                    error=str(e)
                )
        finally:
            # Free NILUT model and intermediate data from memory
            del nilut
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        return results

    def process_all(
        self,
        target: np.ndarray,
        reference: np.ndarray,
        selected_methods: list = None,
        reference_name: Optional[str] = None,
        nilut_mode: str = "per_reference"
    ) -> Dict[str, TransferResult]:
        """
        Process with selected or all available methods.

        Returns:
            Dict mapping method_id to TransferResult
        """
        results = {}

        # Compute protection masks once
        logger.debug("Computing protection masks...")
        masks = self._protection.compute_masks(target)

        # Separate NILUT variants from non-NILUT methods
        nilut_variants = [m for m in (selected_methods or []) if m in NILUT_VARIANT_KEYS]
        non_nilut_methods = [m for m in (selected_methods or []) if m not in NILUT_VARIANT_KEYS]

        # Get available methods for non-NILUT processing
        available_methods = self._get_available_methods()
        if selected_methods:
            methods_to_use = [m for m in available_methods if m.method_id in non_nilut_methods]
        else:
            methods_to_use = available_methods

        # Process non-NILUT methods
        for method in methods_to_use:
            if method.method_id == "nilut":
                continue  # NILUT handled via process_nilut_variants
            logger.info("Processing with %s...", method.method_name)
            result = self.process_single(
                method_id=method.method_id,
                target=target,
                reference=reference,
                masks=masks,
                reference_name=reference_name,
                nilut_mode=nilut_mode
            )
            results[method.method_id] = result
            if result.success:
                logger.info("%s completed in %.1fms", method.method_name, result.processing_time_ms)
            else:
                logger.error("%s failed: %s", method.method_name, result.error)

        # Process NILUT variants if any requested
        if nilut_variants:
            from .nilut_model_service import NILUTModelService
            service = NILUTModelService()
            model_path = str(service.get_universal_model_path("latest"))
            variant_results = self.process_nilut_variants(
                target_image=target,
                reference_image=reference,
                masks=masks,
                model_path=model_path,
                model_display_name="Latest",
                model_id="latest",
                requested_variants=nilut_variants,
                color_strength=self._color_strength,
            )
            results.update(variant_results)

        return results

    def list_available_methods(self) -> List[Dict[str, Any]]:
        """List all available methods with metadata."""
        result = []
        for method_id in ["reinhard", "adain", "nilut"]:
            try:
                method = self._get_method(method_id)
                result.append({
                    "id": method.method_id,
                    "name": method.method_name,
                    "type": method.method_type,
                    "available": method.is_available()
                })
            except Exception as e:
                result.append({
                    "id": method_id,
                    "name": method_id,
                    "type": "unknown",
                    "available": False,
                    "error": str(e)
                })
        return result

    def update_settings(
        self,
        color_strength: Optional[float] = None,
        luminance_strength: Optional[float] = None,
        enable_skin_protection: Optional[bool] = None,
        enable_neon_protection: Optional[bool] = None,
        enable_lip_protection: Optional[bool] = None,
        enable_eye_protection: Optional[bool] = None
    ) -> None:
        """Update processor settings."""
        if color_strength is not None:
            self._color_strength = color_strength
        if luminance_strength is not None:
            self._luminance_strength = luminance_strength

        self._protection.update_settings(
            enable_skin=enable_skin_protection,
            enable_neon=enable_neon_protection,
            enable_lips=enable_lip_protection,
            enable_eyes=enable_eye_protection
        )

        self._methods_cache.clear()
