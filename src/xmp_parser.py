"""
XMP Preset Parser

Parses Adobe Lightroom XMP preset files and extracts LightroomParams.
"""

import re
import logging
from xml.etree import ElementTree as ET

from .xmp_generator import LightroomParams

logger = logging.getLogger(__name__)

# Map XMP attribute names to LightroomParams field names
_XMP_TO_PARAM = {
    'Temperature': 'temperature',
    'Tint': 'tint',
    'Exposure2012': 'exposure',
    'Contrast2012': 'contrast',
    'Highlights2012': 'highlights',
    'Shadows2012': 'shadows',
    'Whites2012': 'whites',
    'Blacks2012': 'blacks',
    'Texture': 'texture',
    'Clarity2012': 'clarity',
    'Dehaze': 'dehaze',
    'Vibrance': 'vibrance',
    'Saturation': 'saturation',
    # HSL Hue
    'HueAdjustmentRed': 'hue_red',
    'HueAdjustmentOrange': 'hue_orange',
    'HueAdjustmentYellow': 'hue_yellow',
    'HueAdjustmentGreen': 'hue_green',
    'HueAdjustmentAqua': 'hue_aqua',
    'HueAdjustmentBlue': 'hue_blue',
    'HueAdjustmentPurple': 'hue_purple',
    'HueAdjustmentMagenta': 'hue_magenta',
    # HSL Saturation
    'SaturationAdjustmentRed': 'sat_red',
    'SaturationAdjustmentOrange': 'sat_orange',
    'SaturationAdjustmentYellow': 'sat_yellow',
    'SaturationAdjustmentGreen': 'sat_green',
    'SaturationAdjustmentAqua': 'sat_aqua',
    'SaturationAdjustmentBlue': 'sat_blue',
    'SaturationAdjustmentPurple': 'sat_purple',
    'SaturationAdjustmentMagenta': 'sat_magenta',
    # HSL Luminance
    'LuminanceAdjustmentRed': 'lum_red',
    'LuminanceAdjustmentOrange': 'lum_orange',
    'LuminanceAdjustmentYellow': 'lum_yellow',
    'LuminanceAdjustmentGreen': 'lum_green',
    'LuminanceAdjustmentAqua': 'lum_aqua',
    'LuminanceAdjustmentBlue': 'lum_blue',
    'LuminanceAdjustmentPurple': 'lum_purple',
    'LuminanceAdjustmentMagenta': 'lum_magenta',
    # Tone Curve
    'ParametricShadows': 'param_shadows',
    'ParametricDarks': 'param_darks',
    'ParametricLights': 'param_lights',
    'ParametricHighlights': 'param_highlights',
    # Color Grading
    'ColorGradeShadowHue': 'cg_shadow_hue',
    'ColorGradeShadowSat': 'cg_shadow_sat',
    'ColorGradeShadowLum': 'cg_shadow_lum',
    'ColorGradeMidtoneHue': 'cg_midtone_hue',
    'ColorGradeMidtoneSat': 'cg_midtone_sat',
    'ColorGradeMidtoneLum': 'cg_midtone_lum',
    'ColorGradeHighlightHue': 'cg_highlight_hue',
    'ColorGradeHighlightSat': 'cg_highlight_sat',
    'ColorGradeHighlightLum': 'cg_highlight_lum',
    'ColorGradeBlending': 'cg_blending',
    # Sharpening
    'Sharpness': 'sharpen_amount',
    'SharpenRadius': 'sharpen_radius',
    'SharpenDetail': 'sharpen_detail',
    'SharpenEdgeMasking': 'sharpen_masking',
}

# Fields that should be parsed as float
_FLOAT_FIELDS = {'exposure', 'sharpen_radius'}


def _parse_value(field_name: str, raw: str):
    """Parse a raw XMP attribute value to the correct Python type."""
    raw = raw.strip().lstrip('+')
    if field_name in _FLOAT_FIELDS:
        return round(float(raw), 2)
    return int(float(raw))


def parse_xmp_to_params(xmp_content: str) -> tuple[LightroomParams, str]:
    """
    Parse XMP preset XML and return LightroomParams + preset name.

    Args:
        xmp_content: Raw XMP XML string

    Returns:
        Tuple of (LightroomParams, preset_name)
    """
    namespaces = {
        'x': 'adobe:ns:meta/',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
    }

    try:
        root = ET.fromstring(xmp_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XMP XML: {e}")

    # Find the rdf:Description element
    desc = root.find('.//rdf:Description', namespaces)
    if desc is None:
        raise ValueError("No rdf:Description element found in XMP")

    # Extract preset name
    preset_name = ""
    name_elem = desc.find('.//crs:Name/rdf:Alt/rdf:li', namespaces)
    if name_elem is not None and name_elem.text:
        preset_name = name_elem.text

    # Extract parameters from attributes
    param_values = {}
    for attr_name, attr_value in desc.attrib.items():
        # Strip namespace prefix (e.g., '{http://...}Temperature' -> 'Temperature')
        local_name = attr_name.split('}')[-1] if '}' in attr_name else attr_name
        # Also handle 'crs:Temperature' format
        if ':' in local_name:
            local_name = local_name.split(':')[-1]

        if local_name in _XMP_TO_PARAM:
            field = _XMP_TO_PARAM[local_name]
            try:
                param_values[field] = _parse_value(field, attr_value)
            except (ValueError, TypeError):
                logger.warning("Could not parse XMP attribute %s=%r", local_name, attr_value)

    params = LightroomParams(**param_values)
    logger.info(
        "Parsed XMP preset '%s' with %d parameters",
        preset_name, len(param_values),
    )
    return params, preset_name
