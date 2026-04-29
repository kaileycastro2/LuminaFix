"""
XMP Preset Generator Module

Maps LAB/HSV color deltas between target and reference images
to Adobe Lightroom parameter space and generates XMP preset files.
"""

import uuid
import logging
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LightroomParams:
    """Lightroom-compatible adjustment parameters."""
    # Basic Tone
    temperature: int = 5500
    tint: int = 0
    exposure: float = 0.0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    whites: int = 0
    blacks: int = 0

    # Presence
    texture: int = 0
    clarity: int = 0
    dehaze: int = 0
    vibrance: int = 0
    saturation: int = 0

    # HSL - Hue
    hue_red: int = 0
    hue_orange: int = 0
    hue_yellow: int = 0
    hue_green: int = 0
    hue_aqua: int = 0
    hue_blue: int = 0
    hue_purple: int = 0
    hue_magenta: int = 0

    # HSL - Saturation
    sat_red: int = 0
    sat_orange: int = 0
    sat_yellow: int = 0
    sat_green: int = 0
    sat_aqua: int = 0
    sat_blue: int = 0
    sat_purple: int = 0
    sat_magenta: int = 0

    # HSL - Luminance
    lum_red: int = 0
    lum_orange: int = 0
    lum_yellow: int = 0
    lum_green: int = 0
    lum_aqua: int = 0
    lum_blue: int = 0
    lum_purple: int = 0
    lum_magenta: int = 0

    # Tone Curve (Parametric)
    param_highlights: int = 0
    param_lights: int = 0
    param_darks: int = 0
    param_shadows: int = 0

    # Color Grading
    cg_shadow_hue: int = 0
    cg_shadow_sat: int = 0
    cg_shadow_lum: int = 0
    cg_midtone_hue: int = 0
    cg_midtone_sat: int = 0
    cg_midtone_lum: int = 0
    cg_highlight_hue: int = 0
    cg_highlight_sat: int = 0
    cg_highlight_lum: int = 0
    cg_blending: int = 50
    cg_balance: int = 0

    # Sharpening
    sharpen_amount: int = 0
    sharpen_radius: float = 1.0
    sharpen_detail: int = 25
    sharpen_masking: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class XMPPresetGenerator:
    """Generates Lightroom-compatible XMP presets from image analysis."""

    def extract_params(
        self,
        target_image: np.ndarray,
        reference_image: np.ndarray,
        strength: float = 0.7,
        method: str = "basic",
    ) -> LightroomParams:
        """
        Extract Lightroom parameters by comparing reference to target.

        Args:
            target_image: BGR input image (what we started with)
            reference_image: BGR reference image (the style we want)
            strength: Color strength multiplier (0.0-1.0)
            method: Extraction strategy — "basic", "color_science", or "optimization"

        Returns:
            LightroomParams with all mapped values
        """
        from .xmp_strategies import STRATEGIES

        strategy_cls = STRATEGIES.get(method)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown XMP extraction method: {method!r}. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        strategy = strategy_cls()
        return strategy.extract_params(target_image, reference_image, strength)

    def generate_xmp(self, params: LightroomParams, preset_name: str) -> str:
        """
        Generate XMP preset XML string compatible with Adobe Lightroom.

        Args:
            params: LightroomParams to encode
            preset_name: Human-readable preset name

        Returns:
            XMP XML string
        """
        preset_uuid = uuid.uuid4().hex.upper()

        def _signed(val) -> str:
            """Format number with explicit +/- sign for Lightroom Mobile."""
            if isinstance(val, float):
                return f"{val:+.2f}"
            return f"{val:+d}"

        lines = [
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
            ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
            '  <rdf:Description rdf:about=""',
            '   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"',
            '   crs:PresetType="Normal"',
            f'   crs:UUID="{preset_uuid}"',
            '   crs:Version="15.0"',
            '   crs:ProcessVersion="11.0"',
            # --- White Balance ---
            '   crs:WhiteBalance="Custom"',
            f'   crs:Temperature="{params.temperature}"',
            f'   crs:Tint="{_signed(params.tint)}"',
            # --- Basic Tone ---
            f'   crs:Exposure2012="{_signed(params.exposure)}"',
            f'   crs:Contrast2012="{_signed(params.contrast)}"',
            f'   crs:Highlights2012="{_signed(params.highlights)}"',
            f'   crs:Shadows2012="{_signed(params.shadows)}"',
            f'   crs:Whites2012="{_signed(params.whites)}"',
            f'   crs:Blacks2012="{_signed(params.blacks)}"',
            # --- Presence ---
            f'   crs:Texture="{_signed(params.texture)}"',
            f'   crs:Clarity2012="{_signed(params.clarity)}"',
            f'   crs:Dehaze="{_signed(params.dehaze)}"',
            f'   crs:Vibrance="{_signed(params.vibrance)}"',
            f'   crs:Saturation="{_signed(params.saturation)}"',
            # --- HSL Hue ---
            f'   crs:HueAdjustmentRed="{_signed(params.hue_red)}"',
            f'   crs:HueAdjustmentOrange="{_signed(params.hue_orange)}"',
            f'   crs:HueAdjustmentYellow="{_signed(params.hue_yellow)}"',
            f'   crs:HueAdjustmentGreen="{_signed(params.hue_green)}"',
            f'   crs:HueAdjustmentAqua="{_signed(params.hue_aqua)}"',
            f'   crs:HueAdjustmentBlue="{_signed(params.hue_blue)}"',
            f'   crs:HueAdjustmentPurple="{_signed(params.hue_purple)}"',
            f'   crs:HueAdjustmentMagenta="{_signed(params.hue_magenta)}"',
            # --- HSL Saturation ---
            f'   crs:SaturationAdjustmentRed="{_signed(params.sat_red)}"',
            f'   crs:SaturationAdjustmentOrange="{_signed(params.sat_orange)}"',
            f'   crs:SaturationAdjustmentYellow="{_signed(params.sat_yellow)}"',
            f'   crs:SaturationAdjustmentGreen="{_signed(params.sat_green)}"',
            f'   crs:SaturationAdjustmentAqua="{_signed(params.sat_aqua)}"',
            f'   crs:SaturationAdjustmentBlue="{_signed(params.sat_blue)}"',
            f'   crs:SaturationAdjustmentPurple="{_signed(params.sat_purple)}"',
            f'   crs:SaturationAdjustmentMagenta="{_signed(params.sat_magenta)}"',
            # --- HSL Luminance ---
            f'   crs:LuminanceAdjustmentRed="{_signed(params.lum_red)}"',
            f'   crs:LuminanceAdjustmentOrange="{_signed(params.lum_orange)}"',
            f'   crs:LuminanceAdjustmentYellow="{_signed(params.lum_yellow)}"',
            f'   crs:LuminanceAdjustmentGreen="{_signed(params.lum_green)}"',
            f'   crs:LuminanceAdjustmentAqua="{_signed(params.lum_aqua)}"',
            f'   crs:LuminanceAdjustmentBlue="{_signed(params.lum_blue)}"',
            f'   crs:LuminanceAdjustmentPurple="{_signed(params.lum_purple)}"',
            f'   crs:LuminanceAdjustmentMagenta="{_signed(params.lum_magenta)}"',
            # --- Tone Curve (Parametric) ---
            f'   crs:ParametricShadows="{_signed(params.param_shadows)}"',
            f'   crs:ParametricDarks="{_signed(params.param_darks)}"',
            f'   crs:ParametricLights="{_signed(params.param_lights)}"',
            f'   crs:ParametricHighlights="{_signed(params.param_highlights)}"',
            # --- Color Grading ---
            f'   crs:ColorGradeShadowHue="{params.cg_shadow_hue}"',
            f'   crs:ColorGradeShadowSat="{params.cg_shadow_sat}"',
            f'   crs:ColorGradeShadowLum="{_signed(params.cg_shadow_lum)}"',
            f'   crs:ColorGradeMidtoneHue="{params.cg_midtone_hue}"',
            f'   crs:ColorGradeMidtoneSat="{params.cg_midtone_sat}"',
            f'   crs:ColorGradeMidtoneLum="{_signed(params.cg_midtone_lum)}"',
            f'   crs:ColorGradeHighlightHue="{params.cg_highlight_hue}"',
            f'   crs:ColorGradeHighlightSat="{params.cg_highlight_sat}"',
            f'   crs:ColorGradeHighlightLum="{_signed(params.cg_highlight_lum)}"',
            f'   crs:ColorGradeBlending="{params.cg_blending}"',
            f'   crs:ColorGradeGlobalHue="+0"',
            f'   crs:ColorGradeGlobalSat="+0"',
            f'   crs:ColorGradeGlobalLum="+0"',
            # --- Sharpening ---
            f'   crs:Sharpness="{params.sharpen_amount}"',
            f'   crs:SharpenRadius="{_signed(params.sharpen_radius)}"',
            f'   crs:SharpenDetail="{params.sharpen_detail}"',
            f'   crs:SharpenEdgeMasking="{params.sharpen_masking}"',
            '   >',
            '   <crs:Name>',
            '    <rdf:Alt>',
            f'     <rdf:li xml:lang="x-default">{preset_name}</rdf:li>',
            '    </rdf:Alt>',
            '   </crs:Name>',
            '  </rdf:Description>',
            ' </rdf:RDF>',
            '</x:xmpmeta>',
        ]
        return '\n'.join(lines)

    def generate_xmp_file(
        self,
        params: LightroomParams,
        preset_name: str,
        output_path: str | Path,
    ) -> Path:
        """
        Write XMP preset to file.

        Args:
            params: LightroomParams to encode
            preset_name: Human-readable preset name
            output_path: Where to write the .xmp file

        Returns:
            Path to the written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xmp_content = self.generate_xmp(params, preset_name)
        output_path.write_text(xmp_content, encoding='utf-8')
        return output_path
