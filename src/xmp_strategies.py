"""
XMP Preset Extraction Strategies

Three methods for mapping image differences to Lightroom slider values:
  - BasicStrategy: Original hardcoded scale factors
  - ColorScienceStrategy: Proper CCT, Duv, log2 exposure via colour-science
  - OptimizationStrategy: Forward model + scipy.optimize
"""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .xmp_generator import LightroomParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants shared across strategies
# ---------------------------------------------------------------------------
TEMP_MIN, TEMP_MAX, TEMP_DEFAULT = 2000, 50000, 5500
TINT_MIN, TINT_MAX = -150, 150
EXPOSURE_MIN, EXPOSURE_MAX = -5.0, 5.0
CONTRAST_MIN, CONTRAST_MAX = -100, 100
HIGHLIGHTS_MIN, HIGHLIGHTS_MAX = -100, 100
SHADOWS_MIN, SHADOWS_MAX = -100, 100
SATURATION_MIN, SATURATION_MAX = -100, 100
VIBRANCE_MIN, VIBRANCE_MAX = -100, 100

# OpenCV HSV hue ranges (0-179) mapped to Lightroom's 8 color channels
HSL_RANGES = {
    'red':     ((0, 8), (173, 180)),
    'orange':  ((8, 23),),
    'yellow':  ((23, 38),),
    'green':   ((38, 83),),
    'aqua':    ((83, 98),),
    'blue':    ((98, 128),),
    'purple':  ((128, 143),),
    'magenta': ((143, 173),),
}


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _get_hue_mask(hue_channel: np.ndarray, color: str) -> np.ndarray:
    ranges = HSL_RANGES[color]
    mask = np.zeros(hue_channel.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (hue_channel >= lo) & (hue_channel < hi)
    return mask


def _extract_hsl(target_hsv: np.ndarray, ref_hsv: np.ndarray, strength: float) -> dict:
    t_h, t_s, t_v = target_hsv[:, :, 0], target_hsv[:, :, 1], target_hsv[:, :, 2]
    r_h, r_s, r_v = ref_hsv[:, :, 0], ref_hsv[:, :, 1], ref_hsv[:, :, 2]

    hsl = {}
    for color in HSL_RANGES:
        t_mask = _get_hue_mask(t_h, color)
        r_mask = _get_hue_mask(r_h, color)

        min_pixels = 100
        if t_mask.sum() < min_pixels or r_mask.sum() < min_pixels:
            hsl[f'hue_{color}'] = 0
            hsl[f'sat_{color}'] = 0
            hsl[f'lum_{color}'] = 0
            continue

        t_hue_mean = float(np.mean(t_h[t_mask]))
        r_hue_mean = float(np.mean(r_h[r_mask]))
        hue_diff = r_hue_mean - t_hue_mean
        if hue_diff > 90:
            hue_diff -= 180
        elif hue_diff < -90:
            hue_diff += 180
        hsl[f'hue_{color}'] = int(np.clip(hue_diff * 0.7 * strength, -100, 100))

        t_sat_mean = float(np.mean(t_s[t_mask]))
        r_sat_mean = float(np.mean(r_s[r_mask]))
        sat_diff = r_sat_mean - t_sat_mean
        hsl[f'sat_{color}'] = int(np.clip((sat_diff / 255.0) * 130.0 * strength, -100, 100))

        t_val_mean = float(np.mean(t_v[t_mask]))
        r_val_mean = float(np.mean(r_v[r_mask]))
        val_diff = r_val_mean - t_val_mean
        hsl[f'lum_{color}'] = int(np.clip((val_diff / 255.0) * 130.0 * strength, -100, 100))

    return hsl


def _extract_tone_curve(t_l: np.ndarray, r_l: np.ndarray, strength: float) -> dict:
    zones = {
        'param_shadows':    (0, 25),
        'param_darks':      (25, 50),
        'param_lights':     (50, 75),
        'param_highlights': (75, 100),
    }
    result = {}
    for name, (plo, phi) in zones.items():
        t_lo = np.percentile(t_l, plo)
        t_hi = np.percentile(t_l, phi)
        r_lo = np.percentile(r_l, plo)
        r_hi = np.percentile(r_l, phi)
        t_mask = (t_l >= t_lo) & (t_l <= t_hi)
        r_mask = (r_l >= r_lo) & (r_l <= r_hi)
        if t_mask.any() and r_mask.any():
            shift = float(np.mean(r_l[r_mask]) - np.mean(t_l[t_mask]))
            result[name] = int(np.clip(shift * 1.0 * strength, -100, 100))
        else:
            result[name] = 0
    return result


def _extract_color_grading(target_lab: np.ndarray, ref_lab: np.ndarray, strength: float) -> dict:
    t_l = target_lab[:, :, 0]
    r_l = ref_lab[:, :, 0]
    t_a, t_b = target_lab[:, :, 1], target_lab[:, :, 2]
    r_a, r_b = ref_lab[:, :, 1], ref_lab[:, :, 2]

    zones = {
        'shadow':    (0, 33),
        'midtone':   (33, 66),
        'highlight': (66, 100),
    }

    result = {}
    for zone_name, (plo, phi) in zones.items():
        t_lo_val = np.percentile(t_l, plo)
        t_hi_val = np.percentile(t_l, phi)
        r_lo_val = np.percentile(r_l, plo)
        r_hi_val = np.percentile(r_l, phi)

        t_mask = (t_l >= t_lo_val) & (t_l <= t_hi_val)
        r_mask = (r_l >= r_lo_val) & (r_l <= r_hi_val)

        if t_mask.any() and r_mask.any():
            t_a_mean = float(np.mean(t_a[t_mask])) - 128
            t_b_mean = float(np.mean(t_b[t_mask])) - 128
            r_a_mean = float(np.mean(r_a[r_mask])) - 128
            r_b_mean = float(np.mean(r_b[r_mask])) - 128

            a_shift = (r_a_mean - t_a_mean) * strength
            b_shift = (r_b_mean - t_b_mean) * strength

            if abs(a_shift) > 0.5 or abs(b_shift) > 0.5:
                hue_angle = np.degrees(np.arctan2(a_shift, b_shift)) % 360
                sat_magnitude = min(100, int(np.sqrt(a_shift**2 + b_shift**2) * 1.5))
            else:
                hue_angle = 0
                sat_magnitude = 0

            lum_shift = float(np.mean(r_l[r_mask]) - np.mean(t_l[t_mask]))
            lum_val = int(np.clip(lum_shift * 0.5 * strength, -100, 100))

            result[f'cg_{zone_name}_hue'] = int(hue_angle)
            result[f'cg_{zone_name}_sat'] = sat_magnitude
            result[f'cg_{zone_name}_lum'] = lum_val
        else:
            result[f'cg_{zone_name}_hue'] = 0
            result[f'cg_{zone_name}_sat'] = 0
            result[f'cg_{zone_name}_lum'] = 0

    result['cg_blending'] = 50
    result['cg_balance'] = 0
    return result


def _extract_presence(
    t_l: np.ndarray, r_l: np.ndarray,
    target_image: np.ndarray, reference_image: np.ndarray,
    strength: float,
) -> dict:
    t_whites_mask = t_l >= np.percentile(t_l, 95)
    r_whites_mask = r_l >= np.percentile(r_l, 95)
    if t_whites_mask.any() and r_whites_mask.any():
        whites = int(np.clip(
            float(np.mean(r_l[r_whites_mask]) - np.mean(t_l[t_whites_mask])) * strength,
            -100, 100,
        ))
    else:
        whites = 0

    t_blacks_mask = t_l <= np.percentile(t_l, 5)
    r_blacks_mask = r_l <= np.percentile(r_l, 5)
    if t_blacks_mask.any() and r_blacks_mask.any():
        blacks = int(np.clip(
            float(np.mean(r_l[r_blacks_mask]) - np.mean(t_l[t_blacks_mask])) * strength,
            -100, 100,
        ))
    else:
        blacks = 0

    t_mid_mask = (t_l >= np.percentile(t_l, 25)) & (t_l <= np.percentile(t_l, 75))
    r_mid_mask = (r_l >= np.percentile(r_l, 25)) & (r_l <= np.percentile(r_l, 75))
    if t_mid_mask.any() and r_mid_mask.any():
        clarity = int(np.clip(
            float(np.std(r_l[r_mid_mask]) - np.std(t_l[t_mid_mask])) * 2.0 * strength,
            -100, 100,
        ))
    else:
        clarity = 0

    t_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    t_texture = cv2.Laplacian(t_gray, cv2.CV_64F).var()
    r_texture = cv2.Laplacian(r_gray, cv2.CV_64F).var()
    texture_ratio = (r_texture - t_texture) / max(t_texture, 1.0)
    texture = int(np.clip(texture_ratio * 30 * strength, -100, 100))

    def _dark_channel(img, patch_size=15):
        b, g, r = cv2.split(img)
        min_channel = np.minimum(np.minimum(b, g), r)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        return cv2.erode(min_channel, kernel)

    t_dark = _dark_channel(target_image).astype(np.float32)
    r_dark = _dark_channel(reference_image).astype(np.float32)
    haze_diff = float(np.mean(r_dark) - np.mean(t_dark))
    dehaze = int(np.clip((haze_diff / 255.0) * 120.0 * strength, -100, 100))

    return {
        'whites': whites, 'blacks': blacks, 'clarity': clarity,
        'texture': texture, 'dehaze': dehaze,
    }


def _extract_sharpening(
    target_image: np.ndarray, reference_image: np.ndarray, strength: float,
) -> dict:
    t_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    t_edges = cv2.Laplacian(t_gray, cv2.CV_64F)
    r_edges = cv2.Laplacian(r_gray, cv2.CV_64F)
    t_edge_strength = np.mean(np.abs(t_edges))
    r_edge_strength = np.mean(np.abs(r_edges))
    edge_ratio = (r_edge_strength - t_edge_strength) / max(t_edge_strength, 1.0)

    sharpen_amount = int(np.clip(30 + edge_ratio * 40 * strength, 0, 100))
    sharpen_radius = round(float(np.clip(1.0 + edge_ratio * 0.4 * strength, 0.5, 2.5)), 1)
    sharpen_detail = int(np.clip(25 + edge_ratio * 20 * strength, 0, 80))

    t_canny = cv2.Canny(t_gray, 50, 150)
    r_canny = cv2.Canny(r_gray, 50, 150)
    masking_diff = np.mean(t_canny > 0) * 100 - np.mean(r_canny > 0) * 100
    sharpen_masking = int(np.clip(masking_diff * 2 * strength, 0, 100))

    return {
        'sharpen_amount': sharpen_amount, 'sharpen_radius': sharpen_radius,
        'sharpen_detail': sharpen_detail, 'sharpen_masking': sharpen_masking,
    }


def _build_lightroom_params(basic: dict, hsl: dict, tone_curve: dict,
                            color_grading: dict, presence: dict,
                            sharpening: dict) -> LightroomParams:
    """Assemble a LightroomParams from the individual extraction dicts."""
    return LightroomParams(
        temperature=basic['temperature'],
        tint=basic['tint'],
        exposure=basic['exposure'],
        contrast=basic['contrast'],
        highlights=basic['highlights'],
        shadows=basic['shadows'],
        whites=presence['whites'],
        blacks=presence['blacks'],
        texture=presence['texture'],
        clarity=presence['clarity'],
        dehaze=presence['dehaze'],
        vibrance=basic['vibrance'],
        saturation=basic['saturation'],
        hue_red=hsl['hue_red'], hue_orange=hsl['hue_orange'],
        hue_yellow=hsl['hue_yellow'], hue_green=hsl['hue_green'],
        hue_aqua=hsl['hue_aqua'], hue_blue=hsl['hue_blue'],
        hue_purple=hsl['hue_purple'], hue_magenta=hsl['hue_magenta'],
        sat_red=hsl['sat_red'], sat_orange=hsl['sat_orange'],
        sat_yellow=hsl['sat_yellow'], sat_green=hsl['sat_green'],
        sat_aqua=hsl['sat_aqua'], sat_blue=hsl['sat_blue'],
        sat_purple=hsl['sat_purple'], sat_magenta=hsl['sat_magenta'],
        lum_red=hsl['lum_red'], lum_orange=hsl['lum_orange'],
        lum_yellow=hsl['lum_yellow'], lum_green=hsl['lum_green'],
        lum_aqua=hsl['lum_aqua'], lum_blue=hsl['lum_blue'],
        lum_purple=hsl['lum_purple'], lum_magenta=hsl['lum_magenta'],
        param_highlights=tone_curve['param_highlights'],
        param_lights=tone_curve['param_lights'],
        param_darks=tone_curve['param_darks'],
        param_shadows=tone_curve['param_shadows'],
        cg_shadow_hue=color_grading['cg_shadow_hue'],
        cg_shadow_sat=color_grading['cg_shadow_sat'],
        cg_shadow_lum=color_grading['cg_shadow_lum'],
        cg_midtone_hue=color_grading['cg_midtone_hue'],
        cg_midtone_sat=color_grading['cg_midtone_sat'],
        cg_midtone_lum=color_grading['cg_midtone_lum'],
        cg_highlight_hue=color_grading['cg_highlight_hue'],
        cg_highlight_sat=color_grading['cg_highlight_sat'],
        cg_highlight_lum=color_grading['cg_highlight_lum'],
        cg_blending=color_grading['cg_blending'],
        cg_balance=color_grading['cg_balance'],
        sharpen_amount=sharpening['sharpen_amount'],
        sharpen_radius=sharpening['sharpen_radius'],
        sharpen_detail=sharpening['sharpen_detail'],
        sharpen_masking=sharpening['sharpen_masking'],
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class XMPExtractionStrategy(ABC):
    @abstractmethod
    def extract_params(
        self,
        target_image: np.ndarray,
        reference_image: np.ndarray,
        strength: float,
    ) -> LightroomParams:
        ...


# ---------------------------------------------------------------------------
# Strategy 1: Basic (original hardcoded scale factors)
# ---------------------------------------------------------------------------

class BasicStrategy(XMPExtractionStrategy):
    """Original approach using hardcoded scale factors on LAB/HSV deltas."""

    def extract_params(self, target_image, reference_image, strength):
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        t_l, t_a, t_b = [target_lab[:, :, i] for i in range(3)]
        r_l, r_a, r_b = [ref_lab[:, :, i] for i in range(3)]

        l_shift = float(np.mean(r_l) - np.mean(t_l))
        a_shift = float(np.mean(r_a) - np.mean(t_a))
        b_shift = float(np.mean(r_b) - np.mean(t_b))
        l_std_change = float(np.std(r_l) - np.std(t_l))

        t_upper_mask = t_l >= np.percentile(t_l, 75)
        r_upper_mask = r_l >= np.percentile(r_l, 75)
        t_lower_mask = t_l <= np.percentile(t_l, 25)
        r_lower_mask = r_l <= np.percentile(r_l, 25)

        t_upper = float(np.mean(t_l[t_upper_mask])) if t_upper_mask.any() else float(np.mean(t_l))
        r_upper = float(np.mean(r_l[r_upper_mask])) if r_upper_mask.any() else float(np.mean(r_l))
        t_lower = float(np.mean(t_l[t_lower_mask])) if t_lower_mask.any() else float(np.mean(t_l))
        r_lower = float(np.mean(r_l[r_lower_mask])) if r_lower_mask.any() else float(np.mean(r_l))

        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_shift = float(np.mean(ref_hsv[:, :, 1]) - np.mean(target_hsv[:, :, 1]))

        temperature = int(np.clip(TEMP_DEFAULT + b_shift * strength, TEMP_MIN, TEMP_MAX))
        tint = int(np.clip(a_shift * strength, TINT_MIN, TINT_MAX))
        exposure = round(float(np.clip((l_shift / 255.0) * strength, EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        contrast = int(np.clip(l_std_change * strength, CONTRAST_MIN, CONTRAST_MAX))
        highlights = int(np.clip((r_upper - t_upper) * strength, HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        shadows = int(np.clip((r_lower - t_lower) * strength, SHADOWS_MIN, SHADOWS_MAX))
        saturation_val = int(np.clip((sat_shift / 255.0) * strength, SATURATION_MIN, SATURATION_MAX))
        vibrance = int(np.clip(saturation_val, VIBRANCE_MIN, VIBRANCE_MAX))

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        hsl = _extract_hsl(target_hsv, ref_hsv, strength)
        tone_curve = _extract_tone_curve(t_l, r_l, strength)
        color_grading = _extract_color_grading(target_lab, ref_lab, strength)
        presence = _extract_presence(t_l, r_l, target_image, reference_image, strength)
        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# Strategy 2: Color Science (CCT, Duv, log2 exposure)
# ---------------------------------------------------------------------------

class ColorScienceStrategy(XMPExtractionStrategy):
    """Uses colour-science library for proper CCT, Duv, and log2 exposure."""

    @staticmethod
    def _l_to_luminance(l_value: float) -> float:
        """Convert OpenCV LAB L (0-255) to relative luminance Y (0-1)."""
        l_star = l_value * 100.0 / 255.0
        if l_star > 7.9996:
            return ((l_star + 16.0) / 116.0) ** 3
        else:
            return l_star / 903.3

    @staticmethod
    def _estimate_cct_duv(bgr_image: np.ndarray) -> tuple[float, float]:
        """Estimate CCT and Duv from mean color of an image."""
        import colour

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        mean_rgb = np.mean(rgb.reshape(-1, 3), axis=0)
        mean_rgb = np.clip(mean_rgb, 1e-10, 1.0)

        xyz = colour.sRGB_to_XYZ(mean_rgb)
        ucs = colour.XYZ_to_UCS(xyz)
        uv = colour.UCS_to_uv(ucs)

        cct, duv = colour.uv_to_CCT(uv, method='Ohno 2013')
        cct = float(np.clip(cct, TEMP_MIN, TEMP_MAX))
        return cct, float(duv)

    def extract_params(self, target_image, reference_image, strength):
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        t_l = target_lab[:, :, 0]
        r_l = ref_lab[:, :, 0]

        # --- Temperature via CCT ---
        try:
            target_cct, target_duv = self._estimate_cct_duv(target_image)
            ref_cct, ref_duv = self._estimate_cct_duv(reference_image)
            # Relative offset: Lightroom temp is relative to as-shot WB
            cct_delta = (ref_cct - target_cct) * strength
            temperature = int(np.clip(
                TEMP_DEFAULT + cct_delta,
                TEMP_MIN, TEMP_MAX,
            ))
            # --- Tint via Duv ---
            # Duv typically ranges -0.02 to +0.02; Lightroom tint is -150 to +150
            duv_delta = (ref_duv - target_duv) * strength
            tint = int(np.clip(duv_delta * 7500, TINT_MIN, TINT_MAX))
        except Exception:
            logger.warning("CCT estimation failed, falling back to LAB-based")
            a_shift = float(np.mean(ref_lab[:, :, 1]) - np.mean(target_lab[:, :, 1]))
            b_shift = float(np.mean(ref_lab[:, :, 2]) - np.mean(target_lab[:, :, 2]))
            temperature = int(np.clip(TEMP_DEFAULT + b_shift * 100 * strength, TEMP_MIN, TEMP_MAX))
            tint = int(np.clip(a_shift * 3.5 * strength, TINT_MIN, TINT_MAX))

        # --- Exposure via log2 (EV stops) ---
        t_l_mean = float(np.mean(t_l))
        r_l_mean = float(np.mean(r_l))
        target_Y = self._l_to_luminance(t_l_mean)
        ref_Y = self._l_to_luminance(r_l_mean)
        if target_Y > 1e-6 and ref_Y > 1e-6:
            exposure_ev = float(np.log2(ref_Y / target_Y)) * strength
        else:
            exposure_ev = 0.0
        exposure = round(float(np.clip(exposure_ev, EXPOSURE_MIN, EXPOSURE_MAX)), 2)

        # --- Contrast via L* std ratio ---
        t_std = float(np.std(t_l))
        r_std = float(np.std(r_l))
        if t_std > 1e-6:
            contrast_ratio = (r_std / t_std) - 1.0
            contrast = int(np.clip(contrast_ratio * 100 * strength, CONTRAST_MIN, CONTRAST_MAX))
        else:
            contrast = 0

        # --- Highlights / Shadows (same quartile approach) ---
        t_upper_mask = t_l >= np.percentile(t_l, 75)
        r_upper_mask = r_l >= np.percentile(r_l, 75)
        t_lower_mask = t_l <= np.percentile(t_l, 25)
        r_lower_mask = r_l <= np.percentile(r_l, 25)

        t_upper = float(np.mean(t_l[t_upper_mask])) if t_upper_mask.any() else t_l_mean
        r_upper = float(np.mean(r_l[r_upper_mask])) if r_upper_mask.any() else r_l_mean
        t_lower = float(np.mean(t_l[t_lower_mask])) if t_lower_mask.any() else t_l_mean
        r_lower = float(np.mean(r_l[r_lower_mask])) if r_lower_mask.any() else r_l_mean

        highlights = int(np.clip((r_upper - t_upper) * strength, HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        shadows = int(np.clip((r_lower - t_lower) * strength, SHADOWS_MIN, SHADOWS_MAX))

        # --- Saturation via CIELAB chroma ---
        t_a_c = target_lab[:, :, 1].astype(np.float64) - 128
        t_b_c = target_lab[:, :, 2].astype(np.float64) - 128
        r_a_c = ref_lab[:, :, 1].astype(np.float64) - 128
        r_b_c = ref_lab[:, :, 2].astype(np.float64) - 128
        t_chroma = float(np.mean(np.sqrt(t_a_c**2 + t_b_c**2)))
        r_chroma = float(np.mean(np.sqrt(r_a_c**2 + r_b_c**2)))
        if t_chroma > 1e-6:
            chroma_ratio = (r_chroma / t_chroma) - 1.0
            saturation_val = int(np.clip(chroma_ratio * 100 * strength, SATURATION_MIN, SATURATION_MAX))
        else:
            saturation_val = 0

        # --- Vibrance: targets low-saturation pixels more ---
        t_low_sat_mask = np.sqrt(t_a_c**2 + t_b_c**2) < np.percentile(
            np.sqrt(t_a_c**2 + t_b_c**2), 50
        )
        r_low_sat_mask = np.sqrt(r_a_c**2 + r_b_c**2) < np.percentile(
            np.sqrt(r_a_c**2 + r_b_c**2), 50
        )
        t_low_chroma = float(np.mean(np.sqrt(t_a_c[t_low_sat_mask]**2 + t_b_c[t_low_sat_mask]**2)))
        r_low_chroma = float(np.mean(np.sqrt(r_a_c[r_low_sat_mask]**2 + r_b_c[r_low_sat_mask]**2)))
        if t_low_chroma > 1e-6:
            vibrance = int(np.clip(((r_low_chroma / t_low_chroma) - 1.0) * 100 * strength,
                                   VIBRANCE_MIN, VIBRANCE_MAX))
        else:
            vibrance = 0

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsl = _extract_hsl(target_hsv, ref_hsv, strength)
        tone_curve = _extract_tone_curve(t_l, r_l, strength)
        color_grading = _extract_color_grading(target_lab, ref_lab, strength)
        presence = _extract_presence(t_l, r_l, target_image, reference_image, strength)
        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# Strategy 3: Optimization (forward model + scipy.optimize)
# ---------------------------------------------------------------------------

class LightroomForwardModel:
    """Simplified forward model of key Lightroom adjustments in LAB space."""

    def __init__(self, source_lab: np.ndarray):
        self.source_lab = source_lab.copy()

    @staticmethod
    def _l_to_Y(l_channel: np.ndarray) -> np.ndarray:
        """OpenCV LAB L (0-255) -> relative luminance Y (0-1)."""
        l_star = l_channel * (100.0 / 255.0)
        Y = np.where(l_star > 7.9996, ((l_star + 16.0) / 116.0) ** 3, l_star / 903.3)
        return Y

    @staticmethod
    def _Y_to_l(Y: np.ndarray) -> np.ndarray:
        """Relative luminance Y (0-1) -> OpenCV LAB L (0-255)."""
        l_star = np.where(Y > 0.008856, 116.0 * np.cbrt(Y) - 16.0, 903.3 * Y)
        return l_star * (255.0 / 100.0)

    def apply(self, params: np.ndarray) -> np.ndarray:
        """
        Apply parameter vector to source image.

        params: [temperature, tint, exposure, contrast, highlights, shadows,
                 whites, blacks, vibrance, saturation,
                 param_highlights, param_lights, param_darks, param_shadows]
        """
        lab = self.source_lab.copy()
        l, a, b = lab[:, :, 0].copy(), lab[:, :, 1].copy(), lab[:, :, 2].copy()

        # Exposure in EV stops
        Y = self._l_to_Y(l)
        Y = Y * (2.0 ** params[2])
        Y = np.clip(Y, 0, 1)
        l = self._Y_to_l(Y)

        # Contrast: expand/compress around midpoint
        l_mid = np.mean(l)
        contrast_factor = 1.0 + params[3] / 100.0
        l = l_mid + (l - l_mid) * contrast_factor

        # Highlights (top quarter)
        hi_mask = np.clip((l - 191) / 64.0, 0, 1)
        l = l + hi_mask * params[4] * 0.5

        # Shadows (bottom quarter)
        sh_mask = np.clip((64 - l) / 64.0, 0, 1)
        l = l + sh_mask * params[5] * 0.5

        # Whites (extreme top)
        w_mask = np.clip((l - 230) / 25.0, 0, 1)
        l = l + w_mask * params[6] * 0.5

        # Blacks (extreme bottom)
        bk_mask = np.clip((25 - l) / 25.0, 0, 1)
        l = l + bk_mask * params[7] * 0.5

        # Tone curve parametric zones
        tc_zones = [
            (params[13], 0, 64),     # param_shadows
            (params[12], 64, 128),   # param_darks
            (params[11], 128, 191),  # param_lights
            (params[10], 191, 255),  # param_highlights
        ]
        for val, lo, hi in tc_zones:
            zone_mask = np.clip(1.0 - np.abs(l - (lo + hi) / 2.0) / ((hi - lo) / 2.0), 0, 1)
            l = l + zone_mask * val * 0.3

        # Temperature: shift B channel
        temp_shift = (params[0] - TEMP_DEFAULT) / 100.0
        b = b + temp_shift * 0.5

        # Tint: shift A channel
        a = a + params[1] * 0.3

        # Saturation: scale chroma
        a_centered = a - 128
        b_centered = b - 128
        sat_factor = 1.0 + params[9] / 100.0
        a = a_centered * sat_factor + 128
        b = b_centered * sat_factor + 128

        # Vibrance: scale low-chroma pixels more
        chroma = np.sqrt(a_centered**2 + b_centered**2)
        max_chroma = np.percentile(chroma, 95) if chroma.max() > 0 else 1.0
        vib_weight = np.clip(1.0 - chroma / max(max_chroma, 1.0), 0, 1)
        vib_factor = 1.0 + (params[8] / 100.0) * vib_weight
        a = (a - 128) * vib_factor + 128
        b = (b - 128) * vib_factor + 128

        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        return np.stack([l, a, b], axis=-1)


# Parameter vector indices for OptimizationStrategy
_OPT_PARAM_NAMES = [
    'temperature', 'tint', 'exposure', 'contrast',
    'highlights', 'shadows', 'whites', 'blacks',
    'vibrance', 'saturation',
    'param_highlights', 'param_lights', 'param_darks', 'param_shadows',
]

_OPT_BOUNDS = [
    (TEMP_MIN, TEMP_MAX),      # temperature
    (TINT_MIN, TINT_MAX),      # tint
    (EXPOSURE_MIN, EXPOSURE_MAX),  # exposure
    (CONTRAST_MIN, CONTRAST_MAX),  # contrast
    (HIGHLIGHTS_MIN, HIGHLIGHTS_MAX),  # highlights
    (SHADOWS_MIN, SHADOWS_MAX),  # shadows
    (-100, 100),               # whites
    (-100, 100),               # blacks
    (VIBRANCE_MIN, VIBRANCE_MAX),  # vibrance
    (SATURATION_MIN, SATURATION_MAX),  # saturation
    (-100, 100),               # param_highlights
    (-100, 100),               # param_lights
    (-100, 100),               # param_darks
    (-100, 100),               # param_shadows
]


class OptimizationStrategy(XMPExtractionStrategy):
    """Uses a forward model + scipy.optimize to find best-fit slider values."""

    def _params_to_vector(self, params: LightroomParams) -> np.ndarray:
        d = params.to_dict()
        return np.array([float(d[name]) for name in _OPT_PARAM_NAMES])

    def _vector_to_dict(self, x: np.ndarray) -> dict:
        return {name: x[i] for i, name in enumerate(_OPT_PARAM_NAMES)}

    def extract_params(self, target_image, reference_image, strength):
        from scipy.optimize import minimize

        # Initial guess from ColorScienceStrategy
        color_sci = ColorScienceStrategy()
        initial_params = color_sci.extract_params(target_image, reference_image, strength)

        # Prepare images in LAB
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Downsample for speed
        max_dim = max(target_lab.shape[:2])
        scale = min(1.0, 256.0 / max_dim)
        if scale < 1.0:
            t_small = cv2.resize(target_lab, None, fx=scale, fy=scale)
            r_small = cv2.resize(ref_lab, None, fx=scale, fy=scale)
        else:
            t_small = target_lab
            r_small = ref_lab

        model = LightroomForwardModel(t_small)

        # L channel matters more than A/B for visual accuracy
        weights = np.array([2.0, 1.0, 1.0], dtype=np.float32)

        def cost(x):
            predicted = model.apply(x)
            diff = predicted - r_small
            mse = float(np.mean(diff**2 * weights[np.newaxis, np.newaxis, :]))
            return mse

        x0 = self._params_to_vector(initial_params)

        try:
            result = minimize(
                cost, x0, method='L-BFGS-B', bounds=_OPT_BOUNDS,
                options={'maxiter': 200, 'ftol': 1e-6},
            )
            optimized = self._vector_to_dict(result.x)
        except Exception:
            logger.warning("Optimization failed, using ColorScience results")
            return initial_params

        # Merge: optimized values for the 14 key sliders, rest from ColorScience
        merged = initial_params.to_dict()
        merged['temperature'] = int(np.clip(optimized['temperature'], TEMP_MIN, TEMP_MAX))
        merged['tint'] = int(np.clip(optimized['tint'], TINT_MIN, TINT_MAX))
        merged['exposure'] = round(float(np.clip(optimized['exposure'], EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        merged['contrast'] = int(np.clip(optimized['contrast'], CONTRAST_MIN, CONTRAST_MAX))
        merged['highlights'] = int(np.clip(optimized['highlights'], HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        merged['shadows'] = int(np.clip(optimized['shadows'], SHADOWS_MIN, SHADOWS_MAX))
        merged['whites'] = int(np.clip(optimized['whites'], -100, 100))
        merged['blacks'] = int(np.clip(optimized['blacks'], -100, 100))
        merged['vibrance'] = int(np.clip(optimized['vibrance'], VIBRANCE_MIN, VIBRANCE_MAX))
        merged['saturation'] = int(np.clip(optimized['saturation'], SATURATION_MIN, SATURATION_MAX))
        merged['param_highlights'] = int(np.clip(optimized['param_highlights'], -100, 100))
        merged['param_lights'] = int(np.clip(optimized['param_lights'], -100, 100))
        merged['param_darks'] = int(np.clip(optimized['param_darks'], -100, 100))
        merged['param_shadows'] = int(np.clip(optimized['param_shadows'], -100, 100))

        return LightroomParams(**merged)


# ---------------------------------------------------------------------------
# Strategy 3b: Basic + Optimizer (uses Basic as initial guess, then optimizes)
# ---------------------------------------------------------------------------

class BasicOptimizedStrategy(XMPExtractionStrategy):
    """Uses Basic strategy as initial guess, then refines with forward model + scipy.optimize."""

    def _params_to_vector(self, params: LightroomParams) -> np.ndarray:
        d = params.to_dict()
        return np.array([float(d[name]) for name in _OPT_PARAM_NAMES])

    def _vector_to_dict(self, x: np.ndarray) -> dict:
        return {name: x[i] for i, name in enumerate(_OPT_PARAM_NAMES)}

    def extract_params(self, target_image, reference_image, strength):
        from scipy.optimize import minimize

        # Initial guess from BasicStrategy
        basic = BasicStrategy()
        initial_params = basic.extract_params(target_image, reference_image, strength)

        # Prepare images in LAB
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Downsample for speed
        max_dim = max(target_lab.shape[:2])
        scale = min(1.0, 256.0 / max_dim)
        if scale < 1.0:
            t_small = cv2.resize(target_lab, None, fx=scale, fy=scale)
            r_small = cv2.resize(ref_lab, None, fx=scale, fy=scale)
        else:
            t_small = target_lab
            r_small = ref_lab

        model = LightroomForwardModel(t_small)

        # L channel matters more than A/B for visual accuracy
        weights = np.array([2.0, 1.0, 1.0], dtype=np.float32)

        def cost(x):
            predicted = model.apply(x)
            diff = predicted - r_small
            mse = float(np.mean(diff**2 * weights[np.newaxis, np.newaxis, :]))
            return mse

        x0 = self._params_to_vector(initial_params)

        try:
            result = minimize(
                cost, x0, method='L-BFGS-B', bounds=_OPT_BOUNDS,
                options={'maxiter': 200, 'ftol': 1e-6},
            )
            optimized = self._vector_to_dict(result.x)
        except Exception:
            logger.warning("Optimization failed, using Basic results")
            return initial_params

        # Merge: optimized values for the 14 key sliders, rest from Basic
        merged = initial_params.to_dict()
        merged['temperature'] = int(np.clip(optimized['temperature'], TEMP_MIN, TEMP_MAX))
        merged['tint'] = int(np.clip(optimized['tint'], TINT_MIN, TINT_MAX))
        merged['exposure'] = round(float(np.clip(optimized['exposure'], EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        merged['contrast'] = int(np.clip(optimized['contrast'], CONTRAST_MIN, CONTRAST_MAX))
        merged['highlights'] = int(np.clip(optimized['highlights'], HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        merged['shadows'] = int(np.clip(optimized['shadows'], SHADOWS_MIN, SHADOWS_MAX))
        merged['whites'] = int(np.clip(optimized['whites'], -100, 100))
        merged['blacks'] = int(np.clip(optimized['blacks'], -100, 100))
        merged['vibrance'] = int(np.clip(optimized['vibrance'], VIBRANCE_MIN, VIBRANCE_MAX))
        merged['saturation'] = int(np.clip(optimized['saturation'], SATURATION_MIN, SATURATION_MAX))
        merged['param_highlights'] = int(np.clip(optimized['param_highlights'], -100, 100))
        merged['param_lights'] = int(np.clip(optimized['param_lights'], -100, 100))
        merged['param_darks'] = int(np.clip(optimized['param_darks'], -100, 100))
        merged['param_shadows'] = int(np.clip(optimized['param_shadows'], -100, 100))

        return LightroomParams(**merged)


# ---------------------------------------------------------------------------
# Strategy 4: Deep Preset (ML-based, predicts 69 Lightroom params via CNN)
# ---------------------------------------------------------------------------

class DeepPresetStrategy(XMPExtractionStrategy):
    """Uses the Deep Preset CNN to predict Lightroom slider values directly."""

    _model = None
    _preset_handler = None
    _device = None

    @classmethod
    def _ensure_loaded(cls):
        """Lazy-load model and preset handler on first use."""
        if cls._model is not None:
            return

        import torch
        from pathlib import Path
        from .deep_preset.networks.network import get_model, PRESET_PREDICTION_IMG_SIZE
        from .deep_preset.utils import PresetHandler

        # Determine device
        cls._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Find checkpoint
        project_root = Path(__file__).parent.parent
        ckpt_path = project_root / 'models' / 'deep_preset' / 'dp_woPPL.pth.tar'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Deep Preset weights not found at {ckpt_path}. "
                "Download with: gdown 1cSJpobfUP3hjNv-gGh3Cs9QT4keb9SeV -O models/deep_preset/dp_woPPL.pth.tar"
            )

        logger.info("Loading Deep Preset model from %s ...", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=cls._device, weights_only=False)
        cls._model = get_model(ckpt['opts'].g_net)(ckpt['opts']).to(cls._device)
        cls._model.load_state_dict(ckpt['G'])
        cls._model.eval()
        cls._preset_handler = PresetHandler()
        logger.info("Deep Preset model loaded on %s", cls._device)

    @staticmethod
    def _bgr_to_tensor(bgr_image: np.ndarray, size=(352, 352)):
        """Convert BGR cv2 image to normalized tensor for Deep Preset."""
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(size, resample=Image.BICUBIC)
        arr = np.array(pil_img).astype(np.float64)
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    # Map Deep Preset's 69 parameter names to our LightroomParams fields
    _PARAM_MAP = {
        # Basic Tone
        'Exposure2012': 'exposure',
        'Contrast2012': 'contrast',
        'Highlights2012': 'highlights',
        'Shadows2012': 'shadows',
        'Whites2012': 'whites',
        'Blacks2012': 'blacks',
        # Presence
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
        'ParametricHighlights': 'param_highlights',
        'ParametricLights': 'param_lights',
        'ParametricDarks': 'param_darks',
        'ParametricShadows': 'param_shadows',
        # Sharpening
        'Sharpness': 'sharpen_amount',
        'SharpenRadius': 'sharpen_radius',
        'SharpenDetail': 'sharpen_detail',
        'SharpenEdgeMasking': 'sharpen_masking',
        # Temperature/Tint (incremental -100..+100 -> our absolute scale)
        'IncrementalTemperature': '_inc_temperature',
        'IncrementalTint': '_inc_tint',
    }

    def extract_params(self, target_image, reference_image, strength):
        import torch

        self._ensure_loaded()

        content = self._bgr_to_tensor(target_image).to(self._device)
        style = self._bgr_to_tensor(reference_image).to(self._device)

        with torch.no_grad():
            _, preset_out, _ = self._model.stylize(
                content, style,
                preset_out_flag=True,
                preset_only=True,
            )

        # Denormalize from [-1, +1] to actual Lightroom values
        preset_np = preset_out[0].cpu().numpy()
        preset_values = self._preset_handler.unnorm_preset(preset_np)
        preset_keys = self._preset_handler.keys

        # Build a dict of Deep Preset param name -> value
        dp_params = {k: v for k, v in zip(preset_keys, preset_values)}

        # Map to our LightroomParams fields
        lr_dict = {}
        for dp_name, lr_name in self._PARAM_MAP.items():
            if dp_name in dp_params:
                lr_dict[lr_name] = dp_params[dp_name]

        # Handle IncrementalTemperature -> absolute Kelvin temperature
        # Deep Preset outputs -100..+100 incremental. Map to 2000-50000K range.
        # 0 incremental = 5500K (daylight). Scale: roughly ±100 -> ±4500K
        inc_temp = lr_dict.pop('_inc_temperature', 0)
        lr_dict['temperature'] = int(np.clip(
            TEMP_DEFAULT + inc_temp * 45, TEMP_MIN, TEMP_MAX
        ))

        inc_tint = lr_dict.pop('_inc_tint', 0)
        lr_dict['tint'] = int(np.clip(inc_tint, TINT_MIN, TINT_MAX))

        # Apply strength scaling to all values except temperature
        if strength != 1.0:
            for key in lr_dict:
                if key == 'temperature':
                    # Scale temperature offset from default
                    offset = lr_dict[key] - TEMP_DEFAULT
                    lr_dict[key] = int(TEMP_DEFAULT + offset * strength)
                elif key == 'exposure':
                    lr_dict[key] = round(float(lr_dict[key]) * strength, 2)
                elif key == 'sharpen_radius':
                    # Don't scale radius by strength
                    pass
                else:
                    lr_dict[key] = int(lr_dict[key] * strength)

        # Ensure exposure is float
        lr_dict['exposure'] = round(float(lr_dict.get('exposure', 0.0)), 2)
        lr_dict['sharpen_radius'] = round(float(lr_dict.get('sharpen_radius', 1.0)), 1)

        # Fill defaults for fields Deep Preset doesn't predict
        # (color grading — Deep Preset uses SplitToning instead)
        lr_dict.setdefault('cg_shadow_hue', 0)
        lr_dict.setdefault('cg_shadow_sat', 0)
        lr_dict.setdefault('cg_shadow_lum', 0)
        lr_dict.setdefault('cg_midtone_hue', 0)
        lr_dict.setdefault('cg_midtone_sat', 0)
        lr_dict.setdefault('cg_midtone_lum', 0)
        lr_dict.setdefault('cg_highlight_hue', 0)
        lr_dict.setdefault('cg_highlight_sat', 0)
        lr_dict.setdefault('cg_highlight_lum', 0)
        lr_dict.setdefault('cg_blending', 50)
        lr_dict.setdefault('cg_balance', 0)

        return LightroomParams(**lr_dict)


# ---------------------------------------------------------------------------
# Strategy 5: darktable-based (inverse of darktable forward formulas)
# ---------------------------------------------------------------------------

# Gaussian hue bins for darktable/RapidRAW-style HSL (centers in degrees, OpenCV 0-179 scale)
_GAUSSIAN_HUE_BINS = {
    'red':     (0, 17),     # center=0°, width=35° -> OpenCV: 0, 17
    'orange':  (12, 22),    # center=25°, width=45° -> 12, 22
    'yellow':  (30, 20),    # center=60°, width=40° -> 30, 20
    'green':   (57, 45),    # center=115°, width=90° -> 57, 45
    'aqua':    (90, 30),    # center=180°, width=60° -> 90, 30
    'blue':    (112, 30),   # center=225°, width=60° -> 112, 30
    'purple':  (140, 27),   # center=280°, width=55° -> 140, 27
    'magenta': (165, 25),   # center=330°, width=50° -> 165, 25
}


def _gaussian_hue_weight(hue_channel: np.ndarray, center: float, width: float) -> np.ndarray:
    """Compute Gaussian influence weight for a hue bin (circular distance)."""
    diff = np.abs(hue_channel - center)
    diff = np.minimum(diff, 180 - diff)  # circular
    falloff = diff / (width * 0.5)
    return np.exp(-1.5 * falloff ** 2)


def _extract_hsl_gaussian(target_hsv: np.ndarray, ref_hsv: np.ndarray, strength: float) -> dict:
    """Extract HSL using Gaussian hue bins (RapidRAW-style)."""
    t_h, t_s, t_v = target_hsv[:, :, 0], target_hsv[:, :, 1], target_hsv[:, :, 2]
    r_h, r_s, r_v = ref_hsv[:, :, 0], ref_hsv[:, :, 1], ref_hsv[:, :, 2]

    hsl = {}
    for color, (center, width) in _GAUSSIAN_HUE_BINS.items():
        t_w = _gaussian_hue_weight(t_h, center, width)
        r_w = _gaussian_hue_weight(r_h, center, width)

        # Require minimum total weight
        if t_w.sum() < 100 or r_w.sum() < 100:
            hsl[f'hue_{color}'] = 0
            hsl[f'sat_{color}'] = 0
            hsl[f'lum_{color}'] = 0
            continue

        # Weighted median approximation using weighted mean
        t_hue_w = float(np.average(t_h, weights=t_w))
        r_hue_w = float(np.average(r_h, weights=r_w))
        hue_diff = r_hue_w - t_hue_w
        if hue_diff > 90:
            hue_diff -= 180
        elif hue_diff < -90:
            hue_diff += 180
        hsl[f'hue_{color}'] = int(np.clip(hue_diff * 0.7 * strength, -100, 100))

        # Saturation: ratio-based
        t_sat_w = float(np.average(t_s, weights=t_w))
        r_sat_w = float(np.average(r_s, weights=r_w))
        if t_sat_w > 1.0:
            sat_ratio = (r_sat_w / t_sat_w) - 1.0
            hsl[f'sat_{color}'] = int(np.clip(sat_ratio * 100 * strength, -100, 100))
        else:
            hsl[f'sat_{color}'] = 0

        # Luminance: ratio-based
        t_val_w = float(np.average(t_v, weights=t_w))
        r_val_w = float(np.average(r_v, weights=r_w))
        if t_val_w > 1.0:
            lum_ratio = (r_val_w / t_val_w) - 1.0
            hsl[f'lum_{color}'] = int(np.clip(lum_ratio * 100 * strength, -100, 100))
        else:
            hsl[f'lum_{color}'] = 0

    return hsl


def _extract_tone_curve_spline(t_l: np.ndarray, r_l: np.ndarray, strength: float) -> dict:
    """Extract tone curve by fitting monotone mapping (original->edited) per zone."""
    zones = {
        'param_shadows':    (0, 25),
        'param_darks':      (25, 50),
        'param_lights':     (50, 75),
        'param_highlights': (75, 100),
    }
    result = {}
    for name, (plo, phi) in zones.items():
        t_lo = np.percentile(t_l, plo)
        t_hi = np.percentile(t_l, phi)
        t_mask = (t_l >= t_lo) & (t_l <= t_hi)
        r_lo = np.percentile(r_l, plo)
        r_hi = np.percentile(r_l, phi)
        r_mask = (r_l >= r_lo) & (r_l <= r_hi)
        if t_mask.any() and r_mask.any():
            t_mean = float(np.mean(t_l[t_mask]))
            r_mean = float(np.mean(r_l[r_mask]))
            if t_mean > 0:
                # Ratio-based instead of difference-based
                ratio = r_mean / t_mean
                shift = (ratio - 1.0) * 100.0 * strength
            else:
                shift = 0.0
            result[name] = int(np.clip(shift, -100, 100))
        else:
            result[name] = 0
    return result


def _extract_color_grading_smoothstep(target_lab: np.ndarray, ref_lab: np.ndarray, strength: float) -> dict:
    """Extract color grading using smoothstep zone masks (RapidRAW/darktable style)."""
    t_l = target_lab[:, :, 0] / 255.0  # normalize to 0-1
    r_l = ref_lab[:, :, 0] / 255.0
    t_a, t_b = target_lab[:, :, 1].astype(np.float32), target_lab[:, :, 2].astype(np.float32)
    r_a, r_b = ref_lab[:, :, 1].astype(np.float32), ref_lab[:, :, 2].astype(np.float32)

    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0, 1)
        return t * t * (3 - 2 * t)

    feather = 0.15
    shadow_mask = 1.0 - smoothstep(0.1 - feather, 0.1 + feather, t_l)
    highlight_mask = smoothstep(0.5 - feather, 0.5 + feather, t_l)
    midtone_mask = np.clip(1.0 - shadow_mask - highlight_mask, 0, 1)

    r_shadow_mask = 1.0 - smoothstep(0.1 - feather, 0.1 + feather, r_l)
    r_highlight_mask = smoothstep(0.5 - feather, 0.5 + feather, r_l)
    r_midtone_mask = np.clip(1.0 - r_shadow_mask - r_highlight_mask, 0, 1)

    result = {}
    for zone_name, t_mask, r_mask in [
        ('shadow', shadow_mask, r_shadow_mask),
        ('midtone', midtone_mask, r_midtone_mask),
        ('highlight', highlight_mask, r_highlight_mask),
    ]:
        t_weight = t_mask.sum()
        r_weight = r_mask.sum()
        if t_weight > 100 and r_weight > 100:
            t_a_mean = float(np.average(t_a, weights=t_mask + 1e-10)) - 128
            t_b_mean = float(np.average(t_b, weights=t_mask + 1e-10)) - 128
            r_a_mean = float(np.average(r_a, weights=r_mask + 1e-10)) - 128
            r_b_mean = float(np.average(r_b, weights=r_mask + 1e-10)) - 128

            a_shift = (r_a_mean - t_a_mean) * strength
            b_shift = (r_b_mean - t_b_mean) * strength

            if abs(a_shift) > 0.5 or abs(b_shift) > 0.5:
                hue_angle = np.degrees(np.arctan2(a_shift, b_shift)) % 360
                sat_magnitude = min(100, int(np.sqrt(a_shift ** 2 + b_shift ** 2) * 1.5))
            else:
                hue_angle = 0
                sat_magnitude = 0

            lum_shift = float(np.average(r_l * 255, weights=r_mask + 1e-10) -
                              np.average(t_l * 255, weights=t_mask + 1e-10))
            lum_val = int(np.clip(lum_shift * 0.5 * strength, -100, 100))

            result[f'cg_{zone_name}_hue'] = int(hue_angle)
            result[f'cg_{zone_name}_sat'] = sat_magnitude
            result[f'cg_{zone_name}_lum'] = lum_val
        else:
            result[f'cg_{zone_name}_hue'] = 0
            result[f'cg_{zone_name}_sat'] = 0
            result[f'cg_{zone_name}_lum'] = 0

    result['cg_blending'] = 50
    result['cg_balance'] = 0
    return result


def _extract_presence_log(
    t_l: np.ndarray, r_l: np.ndarray,
    target_image: np.ndarray, reference_image: np.ndarray,
    strength: float,
) -> dict:
    """Extract presence params using log-domain and ratio-based approach."""
    # Whites: ratio-based
    t_top = t_l[t_l >= np.percentile(t_l, 95)]
    r_top = r_l[r_l >= np.percentile(r_l, 95)]
    if len(t_top) > 0 and len(r_top) > 0 and np.mean(t_top) > 1:
        whites = int(np.clip((np.mean(r_top) / np.mean(t_top) - 1.0) * 100 * strength, -100, 100))
    else:
        whites = 0

    # Blacks: ratio-based
    t_bot = t_l[t_l <= np.percentile(t_l, 5)]
    r_bot = r_l[r_l <= np.percentile(r_l, 5)]
    if len(t_bot) > 0 and len(r_bot) > 0 and np.mean(t_bot) > 1:
        blacks = int(np.clip((np.mean(r_bot) / np.mean(t_bot) - 1.0) * 100 * strength, -100, 100))
    else:
        blacks = 0

    # Clarity: log-domain local contrast
    t_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1
    r_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1
    blur_size = max(3, min(t_gray.shape[:2]) // 10) | 1  # odd kernel
    t_blur = cv2.GaussianBlur(t_gray, (blur_size, blur_size), 0)
    r_blur = cv2.GaussianBlur(r_gray, (blur_size, blur_size), 0)
    t_local = np.log2(t_gray / np.maximum(t_blur, 1))
    r_local = np.log2(r_gray / np.maximum(r_blur, 1))
    clarity_diff = float(np.std(r_local) - np.std(t_local))
    clarity = int(np.clip(clarity_diff * 200 * strength, -100, 100))

    # Texture: Laplacian variance ratio
    t_gray_u8 = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    r_gray_u8 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    t_texture = cv2.Laplacian(t_gray_u8, cv2.CV_64F).var()
    r_texture = cv2.Laplacian(r_gray_u8, cv2.CV_64F).var()
    if t_texture > 0:
        texture_ratio = (r_texture / t_texture) - 1.0
    else:
        texture_ratio = 0
    texture = int(np.clip(texture_ratio * 30 * strength, -100, 100))

    # Dehaze: dark channel
    def _dark_channel(img, patch_size=15):
        b, g, r = cv2.split(img)
        min_channel = np.minimum(np.minimum(b, g), r)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        return cv2.erode(min_channel, kernel)

    t_dark = _dark_channel(target_image).astype(np.float32)
    r_dark = _dark_channel(reference_image).astype(np.float32)
    haze_diff = float(np.mean(r_dark) - np.mean(t_dark))
    dehaze = int(np.clip((haze_diff / 255.0) * 120.0 * strength, -100, 100))

    return {
        'whites': whites, 'blacks': blacks, 'clarity': clarity,
        'texture': texture, 'dehaze': dehaze,
    }


class DarktableStrategy(XMPExtractionStrategy):
    """Inverse of darktable forward formulas: Planck's law CCT, sigmoid contrast, soft-light overlay."""

    @staticmethod
    def _blackbody_spd(wavelength_m, temp_k):
        """Planck's law spectral power distribution (darktable temperature.c)."""
        c1 = 3.7417715246641281639549488324352159753e-16
        c2 = 0.014387769599838156481252937624049081933
        return c1 / (wavelength_m ** 5 * (np.exp(c2 / (wavelength_m * temp_k)) - 1))

    @staticmethod
    def _spd_to_xyz(temp_k):
        """Integrate blackbody SPD against CIE 1931 2-degree observer (darktable temperature.c).

        Uses 81 wavelengths from 380-780nm (5nm steps) with standard observer data.
        """
        # CIE 1931 2-degree observer (380-780nm, 5nm steps) — abbreviated but covers key range
        wavelengths = np.arange(380, 785, 5) * 1e-9  # meters
        # Standard CIE 1931 xbar, ybar, zbar at 5nm intervals (380-780nm)
        xbar = np.array([
            0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148,
            0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511, 0.1954, 0.1421,
            0.0956, 0.058, 0.032, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096,
            0.1655, 0.2257, 0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
            0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384, 0.8544, 0.7514,
            0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636,
            0.0468, 0.0329, 0.0227, 0.0158, 0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.002,
            0.0014, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001,
            0.0
        ])
        ybar = np.array([
            0.0, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.004, 0.0073,
            0.0116, 0.017, 0.0241, 0.0328, 0.0468, 0.061, 0.079, 0.1026, 0.1382, 0.1852,
            0.2536, 0.3391, 0.4608, 0.6067, 0.7618, 0.875, 0.962, 0.9918, 0.9973, 0.9824,
            0.9556, 0.9152, 0.868, 0.8163, 0.757, 0.6949, 0.631, 0.5668, 0.503, 0.4412,
            0.381, 0.321, 0.265, 0.217, 0.175, 0.1382, 0.107, 0.0816, 0.061, 0.0446,
            0.032, 0.0232, 0.017, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.002, 0.0014,
            0.001, 0.0007, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0
        ])
        zbar = np.array([
            0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391,
            1.3856, 1.623, 1.7471, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281, 1.2876, 1.0419,
            0.8130, 0.6162, 0.4652, 0.3533, 0.272, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573,
            0.0422, 0.0298, 0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.002, 0.0016,
            0.0012, 0.0008, 0.0006, 0.0003, 0.0002, 0.0002, 0.0001, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0
        ])
        n = min(len(wavelengths), len(xbar), len(ybar), len(zbar))
        spd = np.array([DarktableStrategy._blackbody_spd(wavelengths[i], temp_k)
                        for i in range(n)])
        X = float(np.sum(spd[:n] * xbar[:n]))
        Y = float(np.sum(spd[:n] * ybar[:n]))
        Z = float(np.sum(spd[:n] * zbar[:n]))
        norm = max(X, Y, Z, 1e-10)
        return X / norm, Y / norm, Z / norm

    @staticmethod
    def _xyz_to_cct(X, Y, Z):
        """Convert XYZ to CCT using McCamy's approximation from chromaticity."""
        denom = X + Y + Z
        if denom < 1e-10:
            return 6500
        x = X / denom
        y = Y / denom
        n = (x - 0.3320) / (0.1858 - y + 1e-10)
        cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
        return max(min(cct, 50000), 2000)

    @staticmethod
    def _image_to_cct(image_bgr):
        """Estimate CCT of an image via SPD→XYZ→CCT (darktable Planck's law approach).

        Computes mean sRGB→XYZ, then finds the blackbody temperature whose
        XYZ chromaticity is closest (binary search like darktable's mul2temp).
        """
        rgb = image_bgr.astype(np.float32) / 255.0
        r_mean = float(np.mean(rgb[:, :, 2]))
        g_mean = float(np.mean(rgb[:, :, 1]))
        b_mean = float(np.mean(rgb[:, :, 0]))
        # sRGB to XYZ (D65)
        X = 0.4124564 * r_mean + 0.3575761 * g_mean + 0.1804375 * b_mean
        Y = 0.2126729 * r_mean + 0.7151522 * g_mean + 0.0721750 * b_mean
        Z = 0.0193339 * r_mean + 0.1191920 * g_mean + 0.9503041 * b_mean
        img_cct = DarktableStrategy._xyz_to_cct(X, Y, Z)

        # Refine via binary search against blackbody SPD (darktable's approach)
        lo, hi = 2000.0, 25000.0
        img_ratio = Z / max(X, 1e-10)
        for _ in range(30):
            mid = (lo + hi) / 2
            bX, bY, bZ = DarktableStrategy._spd_to_xyz(mid)
            bb_ratio = bZ / max(bX, 1e-10)
            if bb_ratio > img_ratio:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def _image_duv(image_bgr):
        """Compute Duv (distance from Planckian locus) for tint estimation."""
        rgb = image_bgr.astype(np.float32) / 255.0
        r_mean = float(np.mean(rgb[:, :, 2]))
        g_mean = float(np.mean(rgb[:, :, 1]))
        b_mean = float(np.mean(rgb[:, :, 0]))
        X = 0.4124564 * r_mean + 0.3575761 * g_mean + 0.1804375 * b_mean
        Y = 0.2126729 * r_mean + 0.7151522 * g_mean + 0.0721750 * b_mean
        Z = 0.0193339 * r_mean + 0.1191920 * g_mean + 0.9503041 * b_mean
        denom = X + Y + Z
        if denom < 1e-10:
            return 0.0
        u = 4 * X / (X + 15 * Y + 3 * Z)
        v = 6 * Y / (X + 15 * Y + 3 * Z)
        # Planckian locus approximation at D65 (~6500K): u=0.1978, v=0.3122
        # Duv = signed distance from locus (positive = green, negative = magenta)
        return float(v - 0.3122 - 2.87 * (u - 0.1978))

    @staticmethod
    def _sigmoid_contrast_inverse(t_l, r_l):
        """Inverse of darktable colisa.c sigmoid contrast.

        darktable forward: L_out = 50 * (scale * k / sqrt(1 + csq * k^2) + 1)
        where boost=20, csq = boost * (contrast-1)^2, scale = sqrt(1+csq), k = 2*(L/100)-1

        We estimate contrast by comparing std deviation spread through the sigmoid.
        """
        boost = 20.0
        t_l_norm = t_l * (100.0 / 255.0)
        r_l_norm = r_l * (100.0 / 255.0)

        # Measure how much the midtone spread changed
        # The sigmoid maps k -> scale * k / sqrt(1 + csq * k^2)
        # For small k (midtones), this ≈ scale * k, so ratio ≈ scale = sqrt(1 + csq)
        t_k = 2 * t_l_norm / 100.0 - 1
        r_k = 2 * r_l_norm / 100.0 - 1

        # Use midtone region (|k| < 0.5) for stable estimation
        mid_mask = np.abs(t_k) < 0.5
        if mid_mask.sum() < 100:
            return 0

        t_spread = float(np.std(t_k[mid_mask]))
        r_spread = float(np.std(r_k[mid_mask]))

        if t_spread < 0.01:
            return 0

        # ratio ≈ contrastscale = sqrt(1 + csq)
        scale_est = r_spread / t_spread
        csq = max(scale_est ** 2 - 1.0, 0)

        # csq = boost * (contrast - 1)^2
        if csq > 0:
            contrast_m1 = np.sqrt(csq / boost)
            contrast_val = 1.0 + contrast_m1 if r_spread > t_spread else 1.0 - contrast_m1
            # Map from darktable's [0.5, 2] range to Lightroom's [-100, 100]
            return int(np.clip((contrast_val - 1.0) * 200, CONTRAST_MIN, CONTRAST_MAX))
        return 0

    @staticmethod
    def _softlight_overlay_inverse(t_l, r_l, strength, is_highlights=True):
        """Inverse of darktable shadhi.c soft-light overlay for highlights/shadows.

        darktable: applies soft-light blend with Gaussian-blurred mask, iterated up to 4x.
        We estimate the slider value by measuring how much the highlight/shadow
        region shifted, accounting for the overlay blend behavior.
        """
        # Normalize to 0-1
        t_norm = t_l / 255.0
        r_norm = r_l / 255.0

        if is_highlights:
            # Top quartile
            thresh = np.percentile(t_norm, 75)
            mask = t_norm >= thresh
        else:
            # Bottom quartile
            thresh = np.percentile(t_norm, 25)
            mask = t_norm <= thresh

        if mask.sum() < 100:
            return 0

        t_zone = t_norm[mask]
        r_zone = r_norm[mask]

        # Soft-light overlay: for base > 0.5: result = 1 - (1-2*(base-0.5))*(1-blend)
        # The shift is non-linear — we estimate the effective slider value from
        # the mean shift, accounting for the overlay's compression behavior
        mean_shift = float(np.mean(r_zone) - np.mean(t_zone))

        # darktable applies up to 4 iterations with opacity chunks of 0.25-1.0
        # Each iteration's effect is roughly additive for small adjustments
        # The total effect maps approximately to slider * 0.5 (from opacity * blend)
        slider_val = mean_shift * 200 * strength

        return int(np.clip(slider_val, -100, 100))

    def extract_params(self, target_image, reference_image, strength):
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        t_l, t_a, t_b = [target_lab[:, :, i] for i in range(3)]
        r_l, r_a, r_b = [ref_lab[:, :, i] for i in range(3)]

        # --- Exposure: log2 with proper L->Y (darktable _l_to_Y) ---
        def l_to_Y(l_channel):
            l_star = l_channel * (100.0 / 255.0)
            return np.where(l_star > 7.9996, ((l_star + 16.0) / 116.0) ** 3, l_star / 903.3)

        Y_t = l_to_Y(t_l)
        Y_r = l_to_Y(r_l)
        valid = (Y_t > 1e-6) & (Y_r > 1e-6)
        if valid.any():
            ev_per_pixel = np.log2(Y_r[valid] / Y_t[valid])
            exposure = round(float(np.clip(np.median(ev_per_pixel) * strength, EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        else:
            exposure = 0.0

        # --- Temperature: SPD → XYZ → CCT via Planck's law (darktable temperature.c) ---
        t_cct = self._image_to_cct(target_image)
        r_cct = self._image_to_cct(reference_image)
        temperature = int(np.clip(TEMP_DEFAULT + (r_cct - t_cct) * strength, TEMP_MIN, TEMP_MAX))

        # --- Tint: Duv from Planckian locus (darktable temperature.c) ---
        t_duv = self._image_duv(target_image)
        r_duv = self._image_duv(reference_image)
        tint = int(np.clip((r_duv - t_duv) * 7500 * strength, TINT_MIN, TINT_MAX))

        # --- Contrast: sigmoid inverse (darktable colisa.c) ---
        contrast = int(np.clip(
            self._sigmoid_contrast_inverse(t_l, r_l) * strength,
            CONTRAST_MIN, CONTRAST_MAX
        ))

        # --- Highlights: soft-light overlay inverse (darktable shadhi.c) ---
        highlights = self._softlight_overlay_inverse(t_l, r_l, strength, is_highlights=True)

        # --- Shadows: soft-light overlay inverse (darktable shadhi.c) ---
        shadows = self._softlight_overlay_inverse(t_l, r_l, strength, is_highlights=False)

        # --- Whites/Blacks: linear remap inverse (darktable basicadj.c) ---
        # whites = (top5_edited - top5_original) / (white - black) × 100
        t_white = float(np.percentile(t_l, 95))
        r_white = float(np.percentile(r_l, 95))
        t_black = float(np.percentile(t_l, 5))
        r_black = float(np.percentile(r_l, 5))
        tonal_range = max(t_white - t_black, 1)

        # --- Saturation: Lab chroma ratio (darktable colisa.c: a*=a*sat, b*=b*sat) ---
        t_chroma = np.sqrt((t_a - 128) ** 2 + (t_b - 128) ** 2)
        r_chroma = np.sqrt((r_a - 128) ** 2 + (r_b - 128) ** 2)
        t_chroma_mean = float(np.mean(t_chroma))
        r_chroma_mean = float(np.mean(r_chroma))
        if t_chroma_mean > 1:
            saturation_val = int(np.clip((r_chroma_mean / t_chroma_mean - 1.0) * 100 * strength,
                                         SATURATION_MIN, SATURATION_MAX))
        else:
            saturation_val = 0

        # --- Vibrance: low-chroma weighted (darktable vibrance.c) ---
        # darktable: sw = sqrt(a^2+b^2)/256, ss = 1 + amount*sw
        # Inverse: only measure boost on low-chroma pixels
        median_chroma = float(np.median(t_chroma))
        low_chroma_mask = t_chroma < median_chroma
        if low_chroma_mask.any():
            t_low = float(np.mean(t_chroma[low_chroma_mask]))
            r_low = float(np.mean(r_chroma[low_chroma_mask]))
            if t_low > 1:
                vibrance = int(np.clip((r_low / t_low - 1.0) * 100 * strength, VIBRANCE_MIN, VIBRANCE_MAX))
            else:
                vibrance = 0
        else:
            vibrance = int(saturation_val * 0.6)

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # HSL: LCh-style Gaussian hue bins (darktable colorzones.c)
        hsl = _extract_hsl_gaussian(target_hsv, ref_hsv, strength)
        # Tone curve: spline fit to L mapping (darktable tonecurve.c)
        tone_curve = _extract_tone_curve_spline(t_l, r_l, strength)
        # Color grading: Ych sigmoid masks (darktable colorbalancergb.c)
        color_grading = _extract_color_grading_smoothstep(target_lab, ref_lab, strength)
        # Presence: log-domain clarity (darktable bilat.c / locallaplacian)
        presence = _extract_presence_log(t_l, r_l, target_image, reference_image, strength)
        presence['whites'] = int(np.clip((r_white - t_white) / tonal_range * 100 * strength, -100, 100))
        presence['blacks'] = int(np.clip((r_black - t_black) / tonal_range * 100 * strength, -100, 100))
        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# Strategy 6: RawTherapee-based (inverse of RawTherapee forward formulas)
# ---------------------------------------------------------------------------

class RawTherapeeStrategy(XMPExtractionStrategy):
    """Inverse of RawTherapee formulas: McCamy CCT, NURBS contrast, gamma highlights/shadows, EPD clarity."""

    @staticmethod
    def _rgb_to_cct_mccamy(rgb_mean):
        """Approximate CCT from mean RGB using McCamy's formula (RawTherapee colortemp.cc).

        RawTherapee uses 97-wavelength CIE matching with binary search.
        McCamy's formula approximates this from chromaticity coordinates.
        """
        r, g, b = rgb_mean
        # sRGB to XYZ (D65) — same matrix as RawTherapee iccmatrices.h
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        denom = X + Y + Z
        if denom < 1e-6:
            return 6500
        x = X / denom
        y = Y / denom
        n = (x - 0.3320) / (0.1858 - y + 1e-10)
        cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
        return max(min(cct, 50000), 2000)

    @staticmethod
    def _nurbs_contrast_inverse(t_l, r_l):
        """Inverse of RawTherapee curves.cc NURBS contrast.

        RawTherapee forward:
          toe_x = avg - avg*(0.6 - contr/250), toe_y = avg - avg*(0.6 + contr/250)
          shoulder_x = avg + (1-avg)*(0.6 - contr/250), shoulder_y = avg + (1-avg)*(0.6 + contr/250)

        We measure how the NURBS S-curve changed the spread around the mean.
        """
        t_norm = t_l / 255.0
        r_norm = r_l / 255.0
        avg = float(np.mean(t_norm))

        # RawTherapee NURBS pivots around avg
        # Measure spread below and above avg separately
        t_below = t_norm[t_norm < avg]
        r_below = r_norm[r_norm < avg]
        t_above = t_norm[t_norm >= avg]
        r_above = r_norm[r_norm >= avg]

        shifts = []
        if len(t_below) > 100 and len(r_below) > 100:
            # Toe region: contr pushes toe_y down → darker shadows
            # toe_y = avg - avg*(0.6 + contr/250)
            t_toe = float(np.percentile(t_below, 25))
            r_toe = float(np.percentile(r_below, 25))
            if avg > 0.01:
                # toe shift normalized by avg
                toe_shift = (t_toe - r_toe) / avg  # positive = darkened = more contrast
                shifts.append(toe_shift * 250)

        if len(t_above) > 100 and len(r_above) > 100:
            # Shoulder region: contr pushes shoulder_y up → brighter highlights
            t_shoulder = float(np.percentile(t_above, 75))
            r_shoulder = float(np.percentile(r_above, 75))
            one_minus_avg = max(1.0 - avg, 0.01)
            shoulder_shift = (r_shoulder - t_shoulder) / one_minus_avg
            shifts.append(shoulder_shift * 250)

        if shifts:
            return int(np.clip(np.mean(shifts), CONTRAST_MIN, CONTRAST_MAX))
        return 0

    @staticmethod
    def _epd_clarity(target_image, reference_image, strength):
        """Clarity via Edge-Preserving Decomposition (RawTherapee EdgePreservingDecomposition.cc).

        RawTherapee EPD: log-compress luminance, smooth via FEM solver, extract detail layer.
        We approximate: log(Source) → Gaussian smooth → detail = log(Source) - smooth.
        Then compare detail energy between images.
        """
        # Downsample for speed (bilateral filter is O(n²) per pixel)
        max_dim = max(target_image.shape[:2])
        scale = min(1.0, 256.0 / max_dim)
        if scale < 1.0:
            t_img = cv2.resize(target_image, None, fx=scale, fy=scale)
            r_img = cv2.resize(reference_image, None, fx=scale, fy=scale)
        else:
            t_img = target_image
            r_img = reference_image

        t_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Log-compress luminance (EPD step 1)
        eps = 0.0001
        t_log = np.log(t_gray / 255.0 + eps)
        r_log = np.log(r_gray / 255.0 + eps)

        # Edge-preserving smooth approximation using bilateral filter
        # (EPD uses FEM solver; bilateral is a reasonable proxy)
        sigma_s = max(3, min(t_gray.shape[:2]) // 10)
        sigma_r = 0.1
        t_smooth = cv2.bilateralFilter(t_log, d=-1,
                                       sigmaColor=sigma_r, sigmaSpace=sigma_s)
        r_smooth = cv2.bilateralFilter(r_log, d=-1,
                                       sigmaColor=sigma_r, sigmaSpace=sigma_s)

        # Detail layer = log(Source) - smooth (EPD step 2)
        t_detail = t_log - t_smooth
        r_detail = r_log - r_smooth

        # Compare detail energy (EPD DetailBoost parameter)
        t_energy = float(np.std(t_detail))
        r_energy = float(np.std(r_detail))

        if t_energy > 1e-6:
            # EPD DetailBoost maps linearly to local contrast enhancement
            clarity = int(np.clip((r_energy / t_energy - 1.0) * 100 * strength, -100, 100))
        else:
            clarity = 0
        return clarity

    def extract_params(self, target_image, reference_image, strength):
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        t_l, t_a, t_b = [target_lab[:, :, i] for i in range(3)]
        r_l, r_a, r_b = [ref_lab[:, :, i] for i in range(3)]

        # --- Exposure: log2 (universal) ---
        t_luma = np.mean(target_image.astype(np.float32), axis=2) / 255.0
        r_luma = np.mean(reference_image.astype(np.float32), axis=2) / 255.0
        valid = (t_luma > 1e-4) & (r_luma > 1e-4)
        if valid.any():
            ev = np.log2(r_luma[valid] / t_luma[valid])
            exposure = round(float(np.clip(np.median(ev) * strength, EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        else:
            exposure = 0.0

        # --- Temperature: CCT via McCamy (approximation of RT 97-wavelength) ---
        target_rgb = target_image.astype(np.float32) / 255.0
        ref_rgb = reference_image.astype(np.float32) / 255.0
        t_rgb_mean = np.mean(target_rgb, axis=(0, 1))[::-1]  # BGR -> RGB
        r_rgb_mean = np.mean(ref_rgb, axis=(0, 1))[::-1]
        t_cct = self._rgb_to_cct_mccamy(t_rgb_mean)
        r_cct = self._rgb_to_cct_mccamy(r_rgb_mean)
        temperature = int(np.clip(TEMP_DEFAULT + (r_cct - t_cct) * strength, TEMP_MIN, TEMP_MAX))

        # --- Tint: green correction factor (RawTherapee colortemp.cc mul2temp) ---
        t_g_mean = float(t_rgb_mean[1])
        r_g_mean = float(r_rgb_mean[1])
        if t_g_mean > 0.01:
            # RT: tint recovered as (tmpg/tmpr) / (gmul/rmul) ratio
            tint = int(np.clip((1.0 - r_g_mean / t_g_mean) * 100 * strength, TINT_MIN, TINT_MAX))
        else:
            tint = 0

        # --- Contrast: NURBS S-curve inverse (RawTherapee curves.cc) ---
        contrast = int(np.clip(
            self._nurbs_contrast_inverse(t_l, r_l) * strength,
            CONTRAST_MIN, CONTRAST_MAX
        ))

        # --- Highlights: gamma inverse log4 (RawTherapee ipshadowshighlights.cc) ---
        # RT: base = 4^(amount/100), gamma_hl = base
        # Inverse: amount = 100 * log4(ratio)
        t_hi = t_l[t_l >= np.percentile(t_l, 75)]
        r_hi = r_l[r_l >= np.percentile(r_l, 75)]
        if len(t_hi) > 0 and len(r_hi) > 0 and np.mean(t_hi) > 1:
            ratio = float(np.mean(r_hi) / np.mean(t_hi))
            highlights = int(np.clip(100 * np.log(max(ratio, 1e-6)) / np.log(4) * strength,
                                     HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        else:
            highlights = 0

        # --- Shadows: gamma inverse log4 (RawTherapee ipshadowshighlights.cc) ---
        # RT: gamma_sh = 1/base = 4^(-amount/100)
        t_sh = t_l[t_l <= np.percentile(t_l, 25)]
        r_sh = r_l[r_l <= np.percentile(r_l, 25)]
        if len(t_sh) > 0 and len(r_sh) > 0 and np.mean(t_sh) > 1:
            ratio = float(np.mean(r_sh) / np.mean(t_sh))
            shadows = int(np.clip(100 * np.log(max(ratio, 1e-6)) / np.log(4) * strength,
                                  SHADOWS_MIN, SHADOWS_MAX))
        else:
            shadows = 0

        # --- Saturation: LCH chroma ratio (RawTherapee improcfun.cc) ---
        t_chroma = np.sqrt((t_a - 128) ** 2 + (t_b - 128) ** 2)
        r_chroma = np.sqrt((r_a - 128) ** 2 + (r_b - 128) ** 2)
        t_chroma_mean = float(np.mean(t_chroma))
        r_chroma_mean = float(np.mean(r_chroma))
        if t_chroma_mean > 1:
            saturation_val = int(np.clip((r_chroma_mean / t_chroma_mean - 1.0) * 100 * strength,
                                         SATURATION_MIN, SATURATION_MAX))
        else:
            saturation_val = 0

        # --- Vibrance: hue-dependent, skin-excluded (RawTherapee ipvibrance.cc) ---
        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        t_hue = target_hsv[:, :, 0] * 2  # OpenCV 0-179 -> 0-358 degrees

        # RT skin hue: Color::SkinSat reduces at skin hues (~15-45°)
        skin_mask = (t_hue >= 15) & (t_hue <= 45)
        low_chroma = t_chroma < np.median(t_chroma)
        non_skin_low = low_chroma & ~skin_mask

        if non_skin_low.any():
            t_low = float(np.mean(t_chroma[non_skin_low]))
            r_low = float(np.mean(r_chroma[non_skin_low]))
            if t_low > 1:
                vibrance = int(np.clip((r_low / t_low - 1.0) * 100 * strength, VIBRANCE_MIN, VIBRANCE_MAX))
            else:
                vibrance = 0
        else:
            vibrance = int(saturation_val * 0.6)

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        # HSL with Gaussian bins (smooth feathering like RT)
        hsl = _extract_hsl_gaussian(target_hsv, ref_hsv, strength)
        # Tone curve: ratio-based zones
        tone_curve = _extract_tone_curve_spline(t_l, r_l, strength)
        # Color grading with smoothstep masks + gamma-corrected strength
        color_grading = _extract_color_grading_smoothstep(target_lab, ref_lab, strength)
        # Presence: EPD-based clarity (RawTherapee EdgePreservingDecomposition.cc)
        presence = _extract_presence_log(t_l, r_l, target_image, reference_image, strength)
        presence['clarity'] = self._epd_clarity(target_image, reference_image, strength)
        # Sharpening: Sobel edge ratio (RawTherapee)
        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# Strategy 7: RapidRAW-based (inverse of RapidRAW WGSL shader formulas)
# ---------------------------------------------------------------------------

class RapidRAWStrategy(XMPExtractionStrategy):
    """Inverse of RapidRAW WGSL shader formulas: RGB multiplier temp, perceptual contrast, skin-aware vibrance."""

    @staticmethod
    def _extract_hsl_rapidraw(target_hsv, ref_hsv, strength):
        """Extract HSL using RapidRAW's exact 8 Gaussian bins with ×0.3 hue scale and weighted median."""
        t_h, t_s, t_v = target_hsv[:, :, 0], target_hsv[:, :, 1], target_hsv[:, :, 2]
        r_h, r_s, r_v = ref_hsv[:, :, 0], ref_hsv[:, :, 1], ref_hsv[:, :, 2]

        # RapidRAW exact hue centers and widths (in degrees, converted to OpenCV 0-179)
        # From shader: Red=358, Orange=25, Yellow=60, Green=115, Aqua=180, Blue=225, Purple=280, Magenta=330
        rapidraw_bins = {
            'red':     (179, 17),   # 358°/2 = 179, width 35°/2 = 17
            'orange':  (12, 22),    # 25°/2 = 12, width 45°/2 = 22
            'yellow':  (30, 20),    # 60°/2 = 30, width 40°/2 = 20
            'green':   (57, 45),    # 115°/2 = 57, width 90°/2 = 45
            'aqua':    (90, 30),    # 180°/2 = 90, width 60°/2 = 30
            'blue':    (112, 30),   # 225°/2 = 112, width 60°/2 = 30
            'purple':  (140, 27),   # 280°/2 = 140, width 55°/2 = 27
            'magenta': (165, 25),   # 330°/2 = 165, width 50°/2 = 25
        }

        hsl = {}
        for color, (center, width) in rapidraw_bins.items():
            t_w = _gaussian_hue_weight(t_h, center, width)
            r_w = _gaussian_hue_weight(r_h, center, width)

            if t_w.sum() < 100 or r_w.sum() < 100:
                hsl[f'hue_{color}'] = 0
                hsl[f'sat_{color}'] = 0
                hsl[f'lum_{color}'] = 0
                continue

            # RapidRAW saturation_mask: smoothstep(0.05, 0.20, hsv.saturation)
            t_sat_norm = t_s / 255.0
            sat_mask = np.clip((t_sat_norm - 0.05) / 0.15, 0, 1)
            sat_mask = sat_mask * sat_mask * (3 - 2 * sat_mask)
            combined_w = t_w * sat_mask

            if combined_w.sum() < 50:
                hsl[f'hue_{color}'] = 0
                hsl[f'sat_{color}'] = 0
                hsl[f'lum_{color}'] = 0
                continue

            # Hue: weighted median with ×0.3 scale (RapidRAW: hsl[i].hue * 2.0, UI scale ×0.3)
            # Compute hue differences per pixel
            hue_diff = r_h - t_h
            hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
            hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)

            # Weighted median approximation: sort by value, find weight midpoint
            flat_diff = hue_diff.ravel()
            flat_w = combined_w.ravel()
            valid_mask = flat_w > 0.01
            if valid_mask.sum() > 50:
                vals = flat_diff[valid_mask]
                wts = flat_w[valid_mask]
                sorted_idx = np.argsort(vals)
                vals_sorted = vals[sorted_idx]
                wts_sorted = wts[sorted_idx]
                cum_weight = np.cumsum(wts_sorted)
                median_idx = np.searchsorted(cum_weight, cum_weight[-1] / 2)
                hue_median = float(vals_sorted[min(median_idx, len(vals_sorted) - 1)])
            else:
                hue_median = 0.0

            # RapidRAW: hue * 2.0 * normalized_influence * saturation_mask, UI hue ×0.3
            # Effective: hue_shift * 2.0 * 0.3 = hue_shift * 0.6
            hsl[f'hue_{color}'] = int(np.clip(hue_median * 0.6 * strength, -100, 100))

            # Saturation: ratio-based with normalized influence
            t_sat_w = float(np.average(t_s, weights=combined_w + 1e-10))
            r_sat_w = float(np.average(r_s, weights=r_w * sat_mask + 1e-10))
            if t_sat_w > 1.0:
                sat_ratio = (r_sat_w / t_sat_w) - 1.0
                hsl[f'sat_{color}'] = int(np.clip(sat_ratio * 100 * strength, -100, 100))
            else:
                hsl[f'sat_{color}'] = 0

            # Luminance: ratio-based with luminance_weight = smoothstep(0, 1, sat)
            lum_weight = t_w * t_sat_norm  # RapidRAW: luminance_weight = smoothstep(0, 1, sat)
            t_val_w = float(np.average(t_v, weights=lum_weight + 1e-10))
            r_val_w = float(np.average(r_v, weights=r_w * (r_s / 255.0) + 1e-10))
            if t_val_w > 1.0:
                lum_ratio = (r_val_w / t_val_w) - 1.0
                hsl[f'lum_{color}'] = int(np.clip(lum_ratio * 100 * strength, -100, 100))
            else:
                hsl[f'lum_{color}'] = 0

        return hsl

    @staticmethod
    def _extract_tone_curve_hermite(t_l, r_l, strength):
        """Tone curve via monotone cubic Hermite spline (Fritsch-Carlson) in sRGB space.

        RapidRAW: apply_curve uses Fritsch-Carlson monotone Hermite interpolation.
        We fit a monotone spline to (original_L → edited_L) mapping, then extract
        4 parametric zone values from the spline shape.
        """
        # Convert to sRGB-like perceptual space (0-1)
        t_perc = np.clip(t_l / 255.0, 0, 1)
        r_perc = np.clip(r_l / 255.0, 0, 1)

        # Build mapping by binning: for each input luminance bin, what's the output?
        n_bins = 64
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mapped_values = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (t_perc >= bin_edges[i]) & (t_perc < bin_edges[i + 1])
            if mask.sum() > 10:
                mapped_values[i] = float(np.median(r_perc[mask]))
            else:
                mapped_values[i] = bin_centers[i]  # identity

        # Enforce monotonicity (Fritsch-Carlson requirement)
        for i in range(1, n_bins):
            if mapped_values[i] < mapped_values[i - 1]:
                mapped_values[i] = mapped_values[i - 1]

        # Extract 4 parametric zone values from the spline shape
        zones = {
            'param_shadows':    (0, 16),      # bins 0-15 (darkest quarter)
            'param_darks':      (16, 32),     # bins 16-31
            'param_lights':     (32, 48),     # bins 32-47
            'param_highlights': (48, 64),     # bins 48-63
        }

        result = {}
        for name, (blo, bhi) in zones.items():
            zone_mapped = mapped_values[blo:bhi]
            zone_identity = bin_centers[blo:bhi]
            # How much the spline deviates from identity in this zone
            deviation = float(np.mean(zone_mapped - zone_identity))
            # Scale: ±0.5 deviation → ±100 slider
            result[name] = int(np.clip(deviation * 200 * strength, -100, 100))

        return result

    @staticmethod
    def _saturation_mix_inverse(target_image, reference_image, strength):
        """Inverse of RapidRAW saturation: mix(luma, rgb, 1+sat) in linear RGB.

        RapidRAW forward: rgb_out = lerp(vec3(luma), rgb_in, 1.0 + sat)
        Inverse: sat = (distance_edited / distance_original) - 1.0
        where distance = |rgb - luma|
        """
        LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722])

        def to_linear(c):
            c = c.astype(np.float32) / 255.0
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

        # BGR -> linear RGB
        t_lin = np.stack([to_linear(target_image[:, :, 2]),
                          to_linear(target_image[:, :, 1]),
                          to_linear(target_image[:, :, 0])], axis=2)
        r_lin = np.stack([to_linear(reference_image[:, :, 2]),
                          to_linear(reference_image[:, :, 1]),
                          to_linear(reference_image[:, :, 0])], axis=2)

        # Compute luma per pixel
        t_luma = np.sum(t_lin * LUMA_COEFF, axis=2, keepdims=True)
        r_luma = np.sum(r_lin * LUMA_COEFF, axis=2, keepdims=True)

        # Distance from gray (chroma in linear RGB)
        t_dist = np.sqrt(np.sum((t_lin - t_luma) ** 2, axis=2))
        r_dist = np.sqrt(np.sum((r_lin - r_luma) ** 2, axis=2))

        # Only measure where there's enough color
        valid = t_dist > 0.01
        if valid.sum() > 100:
            # sat = (r_dist / t_dist) - 1.0 per pixel, take median
            ratio = r_dist[valid] / t_dist[valid]
            sat = float(np.median(ratio)) - 1.0
            return int(np.clip(sat * 100 * strength, SATURATION_MIN, SATURATION_MAX))
        return 0

    def extract_params(self, target_image, reference_image, strength):
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        t_l = target_lab[:, :, 0]
        r_l = ref_lab[:, :, 0]
        t_a, t_b = target_lab[:, :, 1], target_lab[:, :, 2]
        r_a, r_b = ref_lab[:, :, 1], ref_lab[:, :, 2]

        # Convert to linear RGB for RapidRAW-style calculations
        target_rgb = target_image.astype(np.float32) / 255.0
        ref_rgb = reference_image.astype(np.float32) / 255.0
        # BGR -> RGB
        t_r, t_g, t_b_rgb = target_rgb[:, :, 2], target_rgb[:, :, 1], target_rgb[:, :, 0]
        r_r, r_g, r_b_rgb = ref_rgb[:, :, 2], ref_rgb[:, :, 1], ref_rgb[:, :, 0]

        # sRGB to linear
        def to_linear(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

        t_r_lin, t_g_lin, t_b_lin = to_linear(t_r), to_linear(t_g), to_linear(t_b_rgb)
        r_r_lin, r_g_lin, r_b_lin = to_linear(r_r), to_linear(r_g), to_linear(r_b_rgb)

        # --- Temperature: inverse of R×(1+t×0.2), B×(1-t×0.2) ---
        t_r_mean = float(np.mean(t_r_lin))
        r_r_mean = float(np.mean(r_r_lin))
        t_b_mean = float(np.mean(t_b_lin))
        r_b_mean = float(np.mean(r_b_lin))

        if t_r_mean > 0.01 and t_b_mean > 0.01:
            t_from_r = (r_r_mean / t_r_mean - 1.0) / 0.2
            t_from_b = (1.0 - r_b_mean / t_b_mean) / 0.2
            t_val = (t_from_r + t_from_b) / 2.0
            temperature = int(np.clip(TEMP_DEFAULT + t_val * 25 * strength, TEMP_MIN, TEMP_MAX))
        else:
            temperature = TEMP_DEFAULT

        # --- Tint: inverse of G×(1-tnt×0.25) ---
        t_g_mean = float(np.mean(t_g_lin))
        r_g_mean = float(np.mean(r_g_lin))
        if t_g_mean > 0.01:
            tnt = (1.0 - r_g_mean / t_g_mean) / 0.25
            tint = int(np.clip(tnt * 100 * strength, TINT_MIN, TINT_MAX))
        else:
            tint = 0

        # --- Exposure: log2 with BT.709 luma in linear RGB ---
        LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722])
        t_luma = t_r_lin * LUMA_COEFF[0] + t_g_lin * LUMA_COEFF[1] + t_b_lin * LUMA_COEFF[2]
        r_luma = r_r_lin * LUMA_COEFF[0] + r_g_lin * LUMA_COEFF[1] + r_b_lin * LUMA_COEFF[2]
        valid = (t_luma > 1e-4) & (r_luma > 1e-4)
        if valid.any():
            ev = np.log2(r_luma[valid] / t_luma[valid])
            exposure = round(float(np.clip(np.median(ev) * strength, EXPOSURE_MIN, EXPOSURE_MAX)), 2)
        else:
            exposure = 0.0

        # --- Contrast: perceptual S-curve inverse (gamma 2.2, strength=2^(con*1.25)) ---
        t_l_perc = (t_l / 255.0) ** (1 / 2.2)
        r_l_perc = (r_l / 255.0) ** (1 / 2.2)
        t_std_perc = float(np.std(t_l_perc))
        r_std_perc = float(np.std(r_l_perc))
        if t_std_perc > 0.01:
            contrast = int(np.clip(np.log2(r_std_perc / t_std_perc) * 100 / 1.25 * strength,
                                   CONTRAST_MIN, CONTRAST_MAX))
        else:
            contrast = 0

        # --- Highlights: inverse of tanh mask + gamma (RapidRAW apply_highlights_adjustment) ---
        t_hi = t_luma[t_luma >= np.percentile(t_luma, 75)]
        r_hi = r_luma[r_luma >= np.percentile(r_luma, 75)]
        if len(t_hi) > 0 and len(r_hi) > 0 and np.mean(t_hi) > 1e-4:
            ratio = float(np.mean(r_hi) / np.mean(t_hi))
            highlights = int(np.clip(np.log2(max(ratio, 1e-6)) / 1.75 * 120 * strength,
                                     HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        else:
            highlights = 0

        # --- Shadows: inverse of quadratic mask + 2^(sh*1.5) (RapidRAW get_shadow_mult) ---
        t_sh = t_luma[t_luma <= np.percentile(t_luma, 25)]
        r_sh = r_luma[r_luma <= np.percentile(r_luma, 25)]
        if len(t_sh) > 0 and len(r_sh) > 0 and np.mean(t_sh) > 1e-4:
            ratio = float(np.mean(r_sh) / np.mean(t_sh))
            shadows = int(np.clip(np.log2(max(ratio, 1e-6)) / 1.5 * 120 * strength,
                                  SHADOWS_MIN, SHADOWS_MAX))
        else:
            shadows = 0

        # --- Saturation: inverse of mix(luma, rgb, 1+sat) in linear RGB ---
        saturation_val = self._saturation_mix_inverse(target_image, reference_image, strength)

        # --- Vibrance: skin-aware, saturation-masked (RapidRAW apply_creative_color) ---
        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        t_hue_deg = target_hsv[:, :, 0] * 2  # 0-360
        t_sat_hsv = target_hsv[:, :, 1] / 255.0

        # RapidRAW: skin center 25°, smoothstep(35, 10, hue_dist), dampener 0.6
        hue_dist = np.abs(t_hue_deg - 25)
        hue_dist = np.minimum(hue_dist, 360 - hue_dist)
        is_skin = np.clip((35 - hue_dist) / 25, 0, 1)  # smoothstep(35, 10, dist)
        is_skin = is_skin * is_skin * (3 - 2 * is_skin)
        skin_dampener = 1.0 - is_skin * 0.4  # lerp(1.0, 0.6, is_skin)

        # RapidRAW: sat_mask = 1 - smoothstep(0.4, 0.9, current_sat)
        def np_smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0, 1)
            return t * t * (3 - 2 * t)

        sat_mask = 1.0 - np_smoothstep(0.4, 0.9, t_sat_hsv)
        # RapidRAW: amount = vib * sat_mask * skin_dampener * 3.0
        weight = sat_mask * skin_dampener

        t_chroma = np.sqrt((t_a - 128) ** 2 + (t_b - 128) ** 2)
        r_chroma = np.sqrt((r_a - 128) ** 2 + (r_b - 128) ** 2)

        if weight.sum() > 100:
            t_chroma_weighted = float(np.average(t_chroma, weights=weight + 1e-10))
            r_chroma_weighted = float(np.average(r_chroma, weights=weight + 1e-10))
            if t_chroma_weighted > 1:
                # Inverse of ×3.0 multiplier in RapidRAW
                vibrance = int(np.clip((r_chroma_weighted / t_chroma_weighted - 1.0) * 100 / 3.0 * strength,
                                       VIBRANCE_MIN, VIBRANCE_MAX))
            else:
                vibrance = 0
        else:
            vibrance = int(saturation_val * 0.6)

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        # HSL: RapidRAW exact Gaussian bins with ×0.3 hue scale and weighted median
        hsl = self._extract_hsl_rapidraw(target_hsv, ref_hsv, strength)
        # Tone curve: monotone cubic Hermite (Fritsch-Carlson) in sRGB space
        tone_curve = self._extract_tone_curve_hermite(t_l, r_l, strength)
        # Color grading: smoothstep masks (same as RapidRAW apply_color_grading)
        color_grading = _extract_color_grading_smoothstep(target_lab, ref_lab, strength)
        # Presence: log-domain clarity (RapidRAW apply_local_contrast)
        presence = _extract_presence_log(t_l, r_l, target_image, reference_image, strength)
        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# RapidRAW Exact Inverse Strategy
# ---------------------------------------------------------------------------

class RapidRAWExactInverseStrategy(XMPExtractionStrategy):
    """Exact mathematical inverse of RapidRAW WGSL shader formulas.

    Unlike RapidRAWStrategy which approximates via statistical measures (std ratios,
    mean ratios), this strategy inverts each RapidRAW formula per-pixel and takes
    the median, matching the actual shader math as closely as possible.

    RapidRAW processing order (from shader.wgsl apply_all_adjustments):
      1. Sharpness (local contrast mode 0)
      2. Clarity (local contrast mode 1)
      3. Structure (local contrast mode 1, different blur)
      4. Exposure (linear: rgb * 2^EV)
      5. Dehaze (dark channel prior)
      6. White Balance (temperature + tint multipliers)
      7. Whites (global multiplier)
      8. Shadows + Blacks (quadratic mask * 2^(sh*k))
      9. Contrast (perceptual gamma 2.2 S-curve)
     10. Highlights (tanh+smoothstep mask, gamma/boost)
     11. HSL panel (8 Gaussian bins)
     12. Saturation (mix(luma, rgb, 1+sat))
     13. Vibrance (skin-aware, sat-masked)
     14. Color Grading (3-way additive)
     15. Tone Curves (Fritsch-Carlson cubic Hermite, sRGB space)
    """

    @staticmethod
    def _srgb_to_linear(c):
        """Convert sRGB [0,1] to linear RGB."""
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def _linear_to_srgb(c):
        """Convert linear RGB to sRGB [0,1]."""
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 1e-10), 1.0 / 2.4) - 0.055)

    @staticmethod
    def _bt709_luma(r_lin, g_lin, b_lin):
        """BT.709 luminance from linear RGB."""
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0, 1)
        return t * t * (3 - 2 * t)

    @staticmethod
    def _inverse_scurve_per_pixel(out_perc, strength):
        """Exact inverse of RapidRAW's piecewise S-curve.

        Forward (shader.wgsl):
            if p < 0.5:  out = 0.5 * (2*p)^strength
            else:        out = 1 - 0.5 * (2*(1-p))^strength

        Inverse:
            if out < 0.5:  p = ((out/0.5)^(1/strength)) / 2
            else:          p = 1 - ((1-out)/0.5)^(1/strength) / 2
        """
        inv_s = 1.0 / np.maximum(strength, 1e-6)
        result = np.empty_like(out_perc)
        lo = out_perc < 0.5
        hi = ~lo

        # Low part inverse: p = (out/0.5)^(1/s) / 2
        out_lo = np.clip(out_perc[lo], 0, 0.5 - 1e-10)
        result[lo] = np.power(out_lo / 0.5, inv_s) / 2.0

        # High part inverse: p = 1 - ((1-out)/0.5)^(1/s) / 2
        out_hi = np.clip(out_perc[hi], 0.5 + 1e-10, 1.0)
        result[hi] = 1.0 - np.power((1.0 - out_hi) / 0.5, inv_s) / 2.0

        return result

    def _extract_temperature_tint_exact(self, t_lin_rgb, r_lin_rgb):
        """Exact inverse of apply_white_balance.

        Forward: rgb *= (1+temp*0.2, 1+temp*0.05, 1-temp*0.2) * (1+tnt*0.25, 1-tnt*0.25, 1+tnt*0.25)
        So: R_out = R_in * (1+t*0.2)*(1+tnt*0.25)
            G_out = G_in * (1+t*0.05)*(1-tnt*0.25)
            B_out = B_in * (1-t*0.2)*(1+tnt*0.25)

        Solve: ratio_R/ratio_B eliminates tint, ratio_G with known temp gives tint.
        """
        t_r, t_g, t_b = t_lin_rgb[:, :, 0], t_lin_rgb[:, :, 1], t_lin_rgb[:, :, 2]
        r_r, r_g, r_b = r_lin_rgb[:, :, 0], r_lin_rgb[:, :, 1], r_lin_rgb[:, :, 2]

        # Per-pixel ratios
        valid = (t_r > 0.01) & (t_g > 0.01) & (t_b > 0.01)
        if valid.sum() < 100:
            return TEMP_DEFAULT, 0

        ratio_r = r_r[valid] / t_r[valid]  # = (1+t*0.2)*(1+tnt*0.25)
        ratio_g = r_g[valid] / t_g[valid]  # = (1+t*0.05)*(1-tnt*0.25)
        ratio_b = r_b[valid] / t_b[valid]  # = (1-t*0.2)*(1+tnt*0.25)

        # ratio_r / ratio_b = (1+t*0.2) / (1-t*0.2) — tint cancels
        rb_ratio = ratio_r / np.maximum(ratio_b, 1e-6)
        # rb_ratio = (1+t*0.2) / (1-t*0.2) => t = (rb_ratio - 1) / (0.2 * (rb_ratio + 1))
        t_per_pixel = (rb_ratio - 1.0) / (0.2 * (rb_ratio + 1.0))
        temp_raw = float(np.median(t_per_pixel))

        # Now solve for tint using G channel: ratio_g = (1+t*0.05)*(1-tnt*0.25)
        temp_g_mult = 1.0 + temp_raw * 0.05
        if abs(temp_g_mult) > 0.01:
            tnt_from_g = (1.0 - ratio_g / temp_g_mult) / 0.25
            tint_raw = float(np.median(tnt_from_g))
        else:
            tint_raw = 0.0

        # Map to Lightroom scale
        temperature = int(np.clip(TEMP_DEFAULT + temp_raw * 25, TEMP_MIN, TEMP_MAX))
        tint = int(np.clip(tint_raw * 100, TINT_MIN, TINT_MAX))
        return temperature, tint

    def _extract_exposure_exact(self, t_lin_rgb, r_lin_rgb):
        """Exact inverse of apply_linear_exposure: rgb_out = rgb_in * 2^EV.

        Inverse: EV = log2(rgb_out / rgb_in) per pixel, take median.
        """
        valid = (t_lin_rgb > 0.001) & (r_lin_rgb > 0.001)
        if valid.sum() < 100:
            return 0.0
        ev_per_pixel = np.log2(r_lin_rgb[valid] / t_lin_rgb[valid])
        return round(float(np.clip(np.median(ev_per_pixel), EXPOSURE_MIN, EXPOSURE_MAX)), 2)

    def _extract_contrast_exact(self, t_lin_rgb, r_lin_rgb):
        """Exact inverse of RapidRAW S-curve contrast.

        Forward:
            perceptual = rgb^(1/2.2)
            strength = 2^(con * 1.25)
            if p < 0.5: out = 0.5 * (2p)^strength
            else:       out = 1 - 0.5 * (2(1-p))^strength
            result = out^2.2

        Inverse: For a range of candidate `con` values, apply the forward S-curve
        to target pixels and find which `con` minimizes error to reference pixels.
        """
        # Work in perceptual space (gamma 2.2)
        t_perc = np.clip(t_lin_rgb, 0, 1) ** (1 / 2.2)
        r_perc = np.clip(r_lin_rgb, 0, 1) ** (1 / 2.2)

        # Flatten and subsample for speed
        t_flat = t_perc.ravel()
        r_flat = r_perc.ravel()

        # Only use pixels in meaningful range (not clipped)
        valid = (t_flat > 0.02) & (t_flat < 0.98) & (r_flat > 0.02) & (r_flat < 0.98)
        if valid.sum() < 200:
            return 0

        t_v = t_flat[valid]
        r_v = r_flat[valid]

        # Subsample to max 50k pixels
        if len(t_v) > 50000:
            idx = np.random.default_rng(42).choice(len(t_v), 50000, replace=False)
            t_v = t_v[idx]
            r_v = r_v[idx]

        # Binary search for best `con` value
        best_con = 0
        best_err = float('inf')

        for con_candidate in np.linspace(-100, 100, 201):
            s = 2.0 ** (con_candidate * 1.25 / 100.0)
            # Apply forward S-curve to target
            predicted = np.empty_like(t_v)
            lo = t_v < 0.5
            hi = ~lo
            predicted[lo] = 0.5 * np.power(2.0 * t_v[lo], s)
            predicted[hi] = 1.0 - 0.5 * np.power(2.0 * (1.0 - t_v[hi]), s)

            err = float(np.mean(np.abs(predicted - r_v)))
            if err < best_err:
                best_err = err
                best_con = con_candidate

        # Refine with finer search around best
        for con_candidate in np.linspace(max(-100, best_con - 2), min(100, best_con + 2), 41):
            s = 2.0 ** (con_candidate * 1.25 / 100.0)
            predicted = np.empty_like(t_v)
            lo = t_v < 0.5
            hi = ~lo
            predicted[lo] = 0.5 * np.power(2.0 * t_v[lo], s)
            predicted[hi] = 1.0 - 0.5 * np.power(2.0 * (1.0 - t_v[hi]), s)

            err = float(np.mean(np.abs(predicted - r_v)))
            if err < best_err:
                best_err = err
                best_con = con_candidate

        return int(np.clip(best_con, CONTRAST_MIN, CONTRAST_MAX))

    def _extract_highlights_exact(self, t_luma, r_luma):
        """Exact inverse of apply_highlights_adjustment.

        Forward (positive highlights):
            mask = smoothstep(0.3, 0.95, tanh(luma * 1.5))
            adjusted = color * 2^(adj * 1.75)
            result = mix(color, adjusted, mask)

        Forward (negative highlights):
            mask = same
            new_luma = luma^(1 - adj*1.75)  (for luma <= 1)
            result = mix(color, adjusted, mask)

        Inverse: For pixels with high mask weight, solve for adj.
        """
        # Compute mask using target luma (RapidRAW uses input luma for mask)
        safe_luma = np.maximum(t_luma, 0.0001)
        mask_input = np.tanh(safe_luma * 1.5)
        highlight_mask = self._smoothstep(0.3, 0.95, mask_input)

        # Only use pixels with significant highlight mask
        strong_mask = highlight_mask > 0.5
        if strong_mask.sum() < 50:
            return 0

        t_hi = t_luma[strong_mask]
        r_hi = r_luma[strong_mask]
        mask_hi = highlight_mask[strong_mask]

        # Separate into likely positive (reference brighter) and negative (darker)
        ratio = np.median(r_hi / np.maximum(t_hi, 1e-6))

        if ratio >= 1.0:
            # Positive highlights: adjusted = color * 2^(adj*1.75)
            # result = mix(original, original * 2^(adj*1.75), mask)
            # result = original * (1 + mask * (2^(adj*1.75) - 1))
            # result/original = 1 + mask * (2^(adj*1.75) - 1)
            # For strong mask pixels: ratio ≈ 2^(adj*1.75)
            pixel_ratios = r_hi / np.maximum(t_hi, 1e-6)
            # Undo the mask blending: actual_factor = 1 + mask*(factor-1)
            # pixel_ratio = 1 + mask*(2^(adj*1.75) - 1)
            # (pixel_ratio - 1) / mask = 2^(adj*1.75) - 1
            unmasked = (pixel_ratios - 1.0) / np.maximum(mask_hi, 0.01) + 1.0
            unmasked = np.maximum(unmasked, 1e-6)
            adj_per_pixel = np.log2(unmasked) / 1.75
            adj = float(np.median(adj_per_pixel))
        else:
            # Negative highlights: new_luma = luma^gamma where gamma = 1 - adj*1.75
            # For mask-blended: result = mix(original, original^gamma, mask)
            # Approximate: for strong mask, result ≈ original^gamma
            valid_hi = (t_hi > 0.01) & (r_hi > 0.01)
            if valid_hi.sum() < 20:
                return 0
            # log(result) = gamma * log(original) => gamma = log(result)/log(original)
            gamma_per_pixel = np.log(r_hi[valid_hi]) / np.log(np.maximum(t_hi[valid_hi], 1e-6))
            gamma = float(np.median(gamma_per_pixel))
            # gamma = 1 - adj*1.75 => adj = (1 - gamma) / 1.75
            adj = (1.0 - gamma) / 1.75

        highlights = int(np.clip(adj * 120, HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))
        return highlights

    def _extract_shadows_exact(self, t_luma, r_luma):
        """Exact inverse of get_shadow_mult for shadows.

        Forward:
            limit = 0.1
            if luma < limit:
                x = luma / limit
                mask = (1-x)^2
                factor = min(2^(sh*1.5), 3.9)
                mult = mix(1, factor, mask) = 1 + mask*(factor-1)
            rgb *= mult

        Inverse: mult = r/t per pixel, solve for sh.
        """
        # Only pixels in shadow range (luma < 0.1 in linear, ~25 in 0-255 L)
        # We work in linear luma space
        limit = 0.1
        in_shadow = t_luma < limit
        if in_shadow.sum() < 50:
            return 0

        t_sh = t_luma[in_shadow]
        r_sh = r_luma[in_shadow]

        # Compute mask per pixel
        x = t_sh / limit
        mask = (1.0 - x) ** 2

        # mult = r/t per pixel
        mult_per_pixel = r_sh / np.maximum(t_sh, 1e-6)

        # mult = 1 + mask*(factor-1) => factor = 1 + (mult-1)/mask
        valid_mask = mask > 0.05
        if valid_mask.sum() < 20:
            return 0

        factor = 1.0 + (mult_per_pixel[valid_mask] - 1.0) / mask[valid_mask]
        factor = np.maximum(factor, 1e-6)

        # factor = min(2^(sh*1.5), 3.9) => sh = log2(factor) / 1.5
        # (ignore the min(3.9) clamp for now)
        sh_per_pixel = np.log2(factor) / 1.5
        sh = float(np.median(sh_per_pixel))
        return int(np.clip(sh * 120, SHADOWS_MIN, SHADOWS_MAX))

    def _extract_blacks_exact(self, t_luma, r_luma):
        """Exact inverse of get_shadow_mult for blacks.

        Same as shadows but limit=0.05 and factor=min(2^(bl*0.75), 3.9).
        """
        limit = 0.05
        in_black = t_luma < limit
        if in_black.sum() < 50:
            return 0

        t_bl = t_luma[in_black]
        r_bl = r_luma[in_black]

        x = t_bl / limit
        mask = (1.0 - x) ** 2
        mult_per_pixel = r_bl / np.maximum(t_bl, 1e-6)

        valid_mask = mask > 0.05
        if valid_mask.sum() < 20:
            return 0

        factor = 1.0 + (mult_per_pixel[valid_mask] - 1.0) / mask[valid_mask]
        factor = np.maximum(factor, 1e-6)
        bl_per_pixel = np.log2(factor) / 0.75
        bl = float(np.median(bl_per_pixel))
        return int(np.clip(bl * 100, -100, 100))

    def _extract_whites_exact(self, t_lin_rgb, r_lin_rgb):
        """Exact inverse of whites: rgb *= 1 / max(1 - wh*0.25, 0.01).

        This is a global multiplier, so: w_mult = mean(r) / mean(t)
        w_mult = 1 / (1 - wh*0.25) => wh = (1 - 1/w_mult) / 0.25
        """
        t_mean = np.mean(t_lin_rgb)
        r_mean = np.mean(r_lin_rgb)
        if t_mean < 0.001:
            return 0
        w_mult = r_mean / t_mean
        if w_mult < 0.01:
            return 0
        wh = (1.0 - 1.0 / w_mult) / 0.25
        return int(np.clip(wh * 100, -100, 100))

    def _extract_saturation_exact(self, t_lin_rgb, r_lin_rgb):
        """Exact inverse of: out = mix(luma, rgb, 1+sat) = luma + (rgb-luma)*(1+sat).

        Inverse: sat = |out-luma_out| / |rgb-luma_in| - 1 per pixel, take median.
        But more precisely, since luma_out changes too:
            out - luma_out = (rgb - luma_in) * (1+sat)  [luma_out = luma_in for pure sat change]
        So: (1+sat) = dist(out, luma_out) / dist(rgb, luma_in)
        """
        LUMA = np.array([0.2126, 0.7152, 0.0722])
        t_luma = np.sum(t_lin_rgb * LUMA, axis=2, keepdims=True)
        r_luma = np.sum(r_lin_rgb * LUMA, axis=2, keepdims=True)

        t_dist = np.sqrt(np.sum((t_lin_rgb - t_luma) ** 2, axis=2))
        r_dist = np.sqrt(np.sum((r_lin_rgb - r_luma) ** 2, axis=2))

        valid = t_dist > 0.01
        if valid.sum() < 100:
            return 0

        ratio = r_dist[valid] / t_dist[valid]
        sat = float(np.median(ratio)) - 1.0
        return int(np.clip(sat * 100, SATURATION_MIN, SATURATION_MAX))

    def _extract_vibrance_exact(self, t_lin_rgb, r_lin_rgb, t_hsv):
        """Exact inverse of RapidRAW vibrance.

        Forward (positive vib):
            sat_mask = 1 - smoothstep(0.4, 0.9, current_sat)
            skin: hue_dist from 25°, is_skin = smoothstep(35, 10, hue_dist)
            skin_dampener = mix(1.0, 0.6, is_skin)
            amount = vib * sat_mask * skin_dampener * 3.0
            out = mix(luma, rgb, 1 + amount)

        Inverse: per-pixel solve for vib from the saturation change,
        weighted by sat_mask * skin_dampener.
        """
        LUMA = np.array([0.2126, 0.7152, 0.0722])
        t_luma = np.sum(t_lin_rgb * LUMA, axis=2, keepdims=True)
        r_luma = np.sum(r_lin_rgb * LUMA, axis=2, keepdims=True)

        t_dist = np.sqrt(np.sum((t_lin_rgb - t_luma) ** 2, axis=2))
        r_dist = np.sqrt(np.sum((r_lin_rgb - r_luma) ** 2, axis=2))

        # Compute HSV-based masks (from target)
        t_hue_deg = t_hsv[:, :, 0] * 2.0  # OpenCV 0-179 -> 0-358
        t_sat_hsv = t_hsv[:, :, 1] / 255.0

        # Skin mask
        hue_dist = np.abs(t_hue_deg - 25.0)
        hue_dist = np.minimum(hue_dist, 360.0 - hue_dist)
        is_skin = self._smoothstep(35.0, 10.0, hue_dist)
        skin_dampener = 1.0 - is_skin * 0.4  # mix(1.0, 0.6, is_skin)

        # Saturation mask
        c_max = np.max(t_lin_rgb, axis=2)
        c_min = np.min(t_lin_rgb, axis=2)
        delta = c_max - c_min
        current_sat = delta / np.maximum(c_max, 0.001)
        sat_mask = 1.0 - self._smoothstep(0.4, 0.9, current_sat)

        # Per-pixel: amount = (r_dist/t_dist - 1)
        # vib = amount / (sat_mask * skin_dampener * 3.0)
        weight = sat_mask * skin_dampener * 3.0
        valid = (t_dist > 0.01) & (weight > 0.05)
        if valid.sum() < 100:
            return 0

        amount_per_pixel = r_dist[valid] / t_dist[valid] - 1.0
        vib_per_pixel = amount_per_pixel / weight[valid]
        vib = float(np.median(vib_per_pixel))
        return int(np.clip(vib * 100, VIBRANCE_MIN, VIBRANCE_MAX))

    def _extract_clarity_exact(self, target_image, reference_image, strength):
        """Exact inverse of apply_local_contrast (mode 1, positive).

        Forward (positive clarity):
            log_ratio = log2(pixel_luma / blurred_luma)
            contrast_factor = 2^(log_ratio * amount)
            final = pixel * contrast_factor
            result = mix(pixel, final, midtone_mask)

        Inverse: For each midtone pixel, contrast_factor = r/t,
            log_ratio = log2(t/blur_t), amount = log2(contrast_factor) / log_ratio
        """
        t_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        r_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Blur for local contrast (approximate RapidRAW's clarity blur)
        blur_size = max(3, min(t_gray.shape[:2]) // 10) | 1
        t_blur = cv2.GaussianBlur(t_gray, (blur_size, blur_size), 0)

        t_luma = np.maximum(t_gray, 1e-6)
        t_blur = np.maximum(t_blur, 1e-6)
        r_luma = np.maximum(r_gray, 1e-6)

        # Midtone mask (RapidRAW: shadow_protection * highlight_protection)
        shadow_prot = self._smoothstep(0.0, 0.1, t_luma)
        highlight_prot = 1.0 - self._smoothstep(0.9, 1.0, t_luma)
        midtone_mask = shadow_prot * highlight_prot

        # Only use midtone pixels with meaningful local contrast
        log_ratio = np.log2(t_luma / t_blur)
        valid = (midtone_mask > 0.3) & (np.abs(log_ratio) > 0.01)

        if valid.sum() < 100:
            return 0

        # contrast_factor = r/t per pixel (in midtones)
        cf = r_luma[valid] / t_luma[valid]
        lr = log_ratio[valid]

        # amount = log2(cf) / log_ratio per pixel
        amount_per_pixel = np.log2(cf) / lr
        amount = float(np.median(amount_per_pixel))
        return int(np.clip(amount * 100 * strength, -100, 100))

    def _extract_dehaze_exact(self, target_image, reference_image, strength):
        """Inverse of apply_dehaze using dark channel prior.

        Forward (positive dehaze):
            dark = min(R,G,B)
            mapped_haze = (dark-0.02) / (dark-0.02 + 0.2)
            t = max(1 - amount * mapped_haze * 0.85, 0.15)
            recovered = (color - A) / t + A   where A = (0.95, 0.97, 1.0)

        Inverse: given recovered and original, estimate amount.
        """
        t_rgb = target_image.astype(np.float32) / 255.0
        r_rgb = reference_image.astype(np.float32) / 255.0

        # Dark channel of target
        t_dark = np.min(t_rgb, axis=2)
        safe_dark = np.maximum(t_dark - 0.02, 0.0)
        mapped_haze = safe_dark / (safe_dark + 0.2)

        # A = atmospheric light
        A = np.array([0.95, 0.97, 1.0])  # BGR order matches? Actually shader is RGB

        # For pixels with haze: recovered = (color - A) / t + A
        # => t = (color - A) / (recovered - A)
        # But this is per-channel. Use the channel with largest signal.

        # Simplify: use luminance
        t_luma = np.mean(t_rgb, axis=2)
        r_luma = np.mean(r_rgb, axis=2)
        a_mean = np.mean(A)

        valid = (mapped_haze > 0.05) & (np.abs(r_luma - a_mean) > 0.01)
        if valid.sum() < 100:
            return 0

        # t_transmission = (color - A) / (recovered - A) per pixel
        t_val = (t_luma[valid] - a_mean) / (r_luma[valid] - a_mean + 1e-6)
        t_val = np.clip(t_val, 0.15, 1.0)

        # t = max(1 - amount * mapped_haze * 0.85, 0.15)
        # => amount = (1 - t) / (mapped_haze * 0.85)
        amount_per_pixel = (1.0 - t_val) / (mapped_haze[valid] * 0.85 + 1e-6)
        amount = float(np.median(amount_per_pixel))
        return int(np.clip(amount * 100 * strength, -100, 100))

    def _extract_color_grading_exact(self, target_image, reference_image, t_lin_rgb, r_lin_rgb, strength):
        """Exact inverse of apply_color_grading.

        Forward: additive per zone:
            tint_color = hsv_to_rgb(hue, 1, 1) - 0.5
            color += tint_color * saturation * zone_mask * scale_factor
            color += luminance * zone_mask * lum_scale

        Scale factors: shadow=0.3, midtone=0.6, highlight=0.8
        Lum scales: shadow=0.5, midtone=0.8, highlight=1.0
        """
        t_luma = self._bt709_luma(t_lin_rgb[:, :, 0], t_lin_rgb[:, :, 1], t_lin_rgb[:, :, 2])

        # Zone masks (using RapidRAW's exact parameters)
        base_shadow_crossover = 0.1
        base_highlight_crossover = 0.5
        feather = 0.2 * 0.5  # default blending = 50 => 0.2 * 0.5 = 0.1

        shadow_mask = 1.0 - self._smoothstep(
            base_shadow_crossover - feather, base_shadow_crossover + feather, t_luma)
        highlight_mask = self._smoothstep(
            base_highlight_crossover - feather, base_highlight_crossover + feather, t_luma)
        midtone_mask = np.maximum(0, 1.0 - shadow_mask - highlight_mask)

        # Color difference per zone
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_a, t_b = target_lab[:, :, 1] - 128.0, target_lab[:, :, 2] - 128.0
        r_a, r_b = ref_lab[:, :, 1] - 128.0, ref_lab[:, :, 2] - 128.0
        t_l_norm = target_lab[:, :, 0] / 255.0
        r_l_norm = ref_lab[:, :, 0] / 255.0

        scale_factors = {'shadow': 0.3, 'midtone': 0.6, 'highlight': 0.8}
        lum_scales = {'shadow': 0.5, 'midtone': 0.8, 'highlight': 1.0}

        result = {}
        for zone_name, mask in [('shadow', shadow_mask), ('midtone', midtone_mask),
                                 ('highlight', highlight_mask)]:
            w = mask.sum()
            if w > 100:
                a_diff = float(np.average(r_a - t_a, weights=mask + 1e-10))
                b_diff = float(np.average(r_b - t_b, weights=mask + 1e-10))

                # Undo scale factor to get raw saturation
                sf = scale_factors[zone_name]
                if abs(a_diff) > 0.3 or abs(b_diff) > 0.3:
                    hue_angle = np.degrees(np.arctan2(a_diff, b_diff)) % 360
                    raw_magnitude = np.sqrt(a_diff ** 2 + b_diff ** 2)
                    # The additive shift = tint_color * sat * scale_factor
                    # tint_color magnitude ≈ 0.5 (hsv_to_rgb(h,1,1) - 0.5)
                    # So: raw_magnitude ≈ 0.5 * sat/100 * sf * (LAB_scale)
                    sat_magnitude = min(100, int(raw_magnitude / (sf + 0.01) * 1.5 * strength))
                else:
                    hue_angle = 0
                    sat_magnitude = 0

                lum_diff = float(np.average(r_l_norm - t_l_norm, weights=mask + 1e-10)) * 255
                ls = lum_scales[zone_name]
                lum_val = int(np.clip(lum_diff / (ls + 0.01) * strength, -100, 100))

                result[f'cg_{zone_name}_hue'] = int(hue_angle)
                result[f'cg_{zone_name}_sat'] = sat_magnitude
                result[f'cg_{zone_name}_lum'] = lum_val
            else:
                result[f'cg_{zone_name}_hue'] = 0
                result[f'cg_{zone_name}_sat'] = 0
                result[f'cg_{zone_name}_lum'] = 0

        result['cg_blending'] = 50
        result['cg_balance'] = 0
        return result

    def extract_params(self, target_image, reference_image, strength):
        """Extract all parameters using exact inverse of RapidRAW shader formulas."""
        # Convert both images to linear RGB
        t_bgr_f = target_image.astype(np.float32) / 255.0
        r_bgr_f = reference_image.astype(np.float32) / 255.0

        t_lin_rgb = np.stack([
            self._srgb_to_linear(t_bgr_f[:, :, 2]),
            self._srgb_to_linear(t_bgr_f[:, :, 1]),
            self._srgb_to_linear(t_bgr_f[:, :, 0])
        ], axis=2)
        r_lin_rgb = np.stack([
            self._srgb_to_linear(r_bgr_f[:, :, 2]),
            self._srgb_to_linear(r_bgr_f[:, :, 1]),
            self._srgb_to_linear(r_bgr_f[:, :, 0])
        ], axis=2)

        # Luma in linear space
        t_luma = self._bt709_luma(t_lin_rgb[:, :, 0], t_lin_rgb[:, :, 1], t_lin_rgb[:, :, 2])
        r_luma = self._bt709_luma(r_lin_rgb[:, :, 0], r_lin_rgb[:, :, 1], r_lin_rgb[:, :, 2])

        # HSV for vibrance/HSL
        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # LAB for color grading
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_l = target_lab[:, :, 0]
        r_l = ref_lab[:, :, 0]

        # --- Extract all parameters using exact inverses ---

        # Temperature & Tint (joint solve)
        temperature, tint = self._extract_temperature_tint_exact(t_lin_rgb, r_lin_rgb)
        temperature = int(np.clip(
            TEMP_DEFAULT + (temperature - TEMP_DEFAULT) * strength, TEMP_MIN, TEMP_MAX))
        tint = int(np.clip(tint * strength, TINT_MIN, TINT_MAX))

        # Exposure
        exposure = self._extract_exposure_exact(t_lin_rgb, r_lin_rgb)
        exposure = round(float(np.clip(exposure * strength, EXPOSURE_MIN, EXPOSURE_MAX)), 2)

        # Contrast (per-pixel S-curve fitting)
        contrast = self._extract_contrast_exact(t_lin_rgb, r_lin_rgb)
        contrast = int(np.clip(contrast * strength, CONTRAST_MIN, CONTRAST_MAX))

        # Highlights (tanh + smoothstep mask)
        highlights = self._extract_highlights_exact(t_luma, r_luma)
        highlights = int(np.clip(highlights * strength, HIGHLIGHTS_MIN, HIGHLIGHTS_MAX))

        # Shadows (quadratic mask, limit=0.1)
        shadows = self._extract_shadows_exact(t_luma, r_luma)
        shadows = int(np.clip(shadows * strength, SHADOWS_MIN, SHADOWS_MAX))

        # Whites (global multiplier)
        whites = self._extract_whites_exact(t_lin_rgb, r_lin_rgb)
        whites = int(np.clip(whites * strength, -100, 100))

        # Blacks (quadratic mask, limit=0.05)
        blacks = self._extract_blacks_exact(t_luma, r_luma)
        blacks = int(np.clip(blacks * strength, -100, 100))

        # Saturation (exact mix inverse)
        saturation_val = self._extract_saturation_exact(t_lin_rgb, r_lin_rgb)
        saturation_val = int(np.clip(saturation_val * strength, SATURATION_MIN, SATURATION_MAX))

        # Vibrance (skin-aware, per-pixel)
        vibrance = self._extract_vibrance_exact(t_lin_rgb, r_lin_rgb, target_hsv)
        vibrance = int(np.clip(vibrance * strength, VIBRANCE_MIN, VIBRANCE_MAX))

        basic = {
            'temperature': temperature, 'tint': tint, 'exposure': exposure,
            'contrast': contrast, 'highlights': highlights, 'shadows': shadows,
            'saturation': saturation_val, 'vibrance': vibrance,
        }

        # HSL: reuse RapidRAW's Gaussian bin extraction (already matches shader bins)
        hsl = RapidRAWStrategy._extract_hsl_rapidraw(target_hsv, ref_hsv, strength)

        # Tone curve: reuse Hermite extraction (matches shader's Fritsch-Carlson)
        tone_curve = RapidRAWStrategy._extract_tone_curve_hermite(t_l, r_l, strength)

        # Color grading: exact inverse with proper scale factors
        color_grading = self._extract_color_grading_exact(
            target_image, reference_image, t_lin_rgb, r_lin_rgb, strength)

        # Presence: exact clarity + dehaze
        clarity = self._extract_clarity_exact(target_image, reference_image, strength)
        dehaze = self._extract_dehaze_exact(target_image, reference_image, strength)

        # Texture: use shared extractor (RapidRAW texture = same local contrast, different blur)
        t_gray_u8 = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        r_gray_u8 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        t_texture = cv2.Laplacian(t_gray_u8, cv2.CV_64F).var()
        r_texture = cv2.Laplacian(r_gray_u8, cv2.CV_64F).var()
        texture = int(np.clip(
            ((r_texture / max(t_texture, 1)) - 1.0) * 30 * strength, -100, 100)) if t_texture > 0 else 0

        presence = {
            'whites': whites, 'blacks': blacks, 'clarity': clarity,
            'texture': texture, 'dehaze': dehaze,
        }

        sharpening = _extract_sharpening(target_image, reference_image, strength)

        return _build_lightroom_params(basic, hsl, tone_curve, color_grading, presence, sharpening)


# ---------------------------------------------------------------------------
# Enhanced Forward Model (all 55 params, strategy-specific math)
# ---------------------------------------------------------------------------

# Full 55-parameter names matching LightroomParams fields
_FULL_PARAM_NAMES = [
    # Basic (8)
    'temperature', 'tint', 'exposure', 'contrast',
    'highlights', 'shadows', 'whites', 'blacks',
    # Presence (5)
    'texture', 'clarity', 'dehaze', 'vibrance', 'saturation',
    # HSL Hue (8)
    'hue_red', 'hue_orange', 'hue_yellow', 'hue_green',
    'hue_aqua', 'hue_blue', 'hue_purple', 'hue_magenta',
    # HSL Saturation (8)
    'sat_red', 'sat_orange', 'sat_yellow', 'sat_green',
    'sat_aqua', 'sat_blue', 'sat_purple', 'sat_magenta',
    # HSL Luminance (8)
    'lum_red', 'lum_orange', 'lum_yellow', 'lum_green',
    'lum_aqua', 'lum_blue', 'lum_purple', 'lum_magenta',
    # Tone Curve (4)
    'param_highlights', 'param_lights', 'param_darks', 'param_shadows',
    # Color Grading (11)
    'cg_shadow_hue', 'cg_shadow_sat', 'cg_shadow_lum',
    'cg_midtone_hue', 'cg_midtone_sat', 'cg_midtone_lum',
    'cg_highlight_hue', 'cg_highlight_sat', 'cg_highlight_lum',
    'cg_blending', 'cg_balance',
    # Sharpening (4)
    'sharpen_amount', 'sharpen_radius', 'sharpen_detail', 'sharpen_masking',
]

_FULL_BOUNDS = (
    # Basic (8)
    [(TEMP_MIN, TEMP_MAX), (TINT_MIN, TINT_MAX), (EXPOSURE_MIN, EXPOSURE_MAX),
     (CONTRAST_MIN, CONTRAST_MAX), (HIGHLIGHTS_MIN, HIGHLIGHTS_MAX),
     (SHADOWS_MIN, SHADOWS_MAX), (-100, 100), (-100, 100)]
    # Presence (5)
    + [(-100, 100)] * 5
    # HSL Hue (8)
    + [(-100, 100)] * 8
    # HSL Saturation (8)
    + [(-100, 100)] * 8
    # HSL Luminance (8)
    + [(-100, 100)] * 8
    # Tone Curve (4)
    + [(-100, 100)] * 4
    # Color Grading: hue (0-360), sat (0-100), lum (-100,100) × 3, blending (0-100), balance (-100,100)
    + [(0, 360), (0, 100), (-100, 100)] * 3
    + [(0, 100), (-100, 100)]
    # Sharpening: amount (0-150), radius (0.5-3), detail (0-100), masking (0-100)
    + [(0, 150), (0.5, 3.0), (0, 100), (0, 100)]
)


class EnhancedForwardModel:
    """Forward model simulating all 55 Lightroom sliders using strategy-specific math.

    Operates in LAB+HSV space on a downsampled source image.
    strategy_type: 'darktable', 'rawtherapee', or 'rapidraw'
    """

    def __init__(self, source_bgr: np.ndarray, strategy_type: str = 'darktable'):
        # Downsample for speed
        max_dim = max(source_bgr.shape[:2])
        scale = min(1.0, 96.0 / max_dim)
        if scale < 1.0:
            self.source_bgr = cv2.resize(source_bgr, None, fx=scale, fy=scale)
        else:
            self.source_bgr = source_bgr.copy()
        self.source_lab = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.source_hsv = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        self.strategy_type = strategy_type

        # Precompute linear RGB for rapidraw
        self._source_lin = None
        if strategy_type == 'rapidraw':
            bgr_f = self.source_bgr.astype(np.float32) / 255.0
            r, g, b = bgr_f[:, :, 2], bgr_f[:, :, 1], bgr_f[:, :, 0]
            to_lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            self._source_lin = np.stack([to_lin(r), to_lin(g), to_lin(b)], axis=2)

    @staticmethod
    def _l_to_Y(l_channel):
        l_star = l_channel * (100.0 / 255.0)
        return np.where(l_star > 7.9996, ((l_star + 16.0) / 116.0) ** 3, l_star / 903.3)

    @staticmethod
    def _Y_to_l(Y):
        l_star = np.where(Y > 0.008856, 116.0 * np.cbrt(Y) - 16.0, 903.3 * Y)
        return l_star * (255.0 / 100.0)

    def _param_index(self, name):
        return _FULL_PARAM_NAMES.index(name)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply all 55 params to source, return predicted LAB image."""
        lab = self.source_lab.copy()
        l = lab[:, :, 0].copy()
        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()
        hsv = self.source_hsv.copy()
        h_ch = hsv[:, :, 0]

        p = {name: x[i] for i, name in enumerate(_FULL_PARAM_NAMES)}

        # ---- EXPOSURE ----
        Y = self._l_to_Y(l)
        if self.strategy_type == 'rapidraw':
            # RapidRAW: multiply linear luma by 2^exposure
            Y = Y * (2.0 ** p['exposure'])
        elif self.strategy_type == 'darktable':
            # darktable: same EV-based
            Y = Y * (2.0 ** p['exposure'])
        else:  # rawtherapee
            Y = Y * (2.0 ** p['exposure'])
        Y = np.clip(Y, 0, 1)
        l = self._Y_to_l(Y)

        # ---- CONTRAST ----
        l_mid = np.mean(l)
        if self.strategy_type == 'darktable':
            # Sigmoid contrast (colisa.c): scale * k / sqrt(1 + csq * k^2)
            boost = 20.0
            contrast_frac = p['contrast'] / 200.0 + 1.0  # map -100..100 to 0.5..1.5
            csq = boost * (contrast_frac - 1.0) ** 2
            scale = np.sqrt(1.0 + csq)
            k = 2.0 * (l / 255.0 * 100.0) / 100.0 - 1.0
            k_out = scale * k / np.sqrt(1.0 + csq * k ** 2 + 1e-10)
            l = ((k_out + 1.0) / 2.0) * 255.0
        elif self.strategy_type == 'rawtherapee':
            # NURBS S-curve: pivot at mean
            contr = p['contrast'] / 250.0
            l_norm = l / 255.0
            avg = np.mean(l_norm)
            # Approximate NURBS S-curve
            below = l_norm < avg
            above = ~below
            if avg > 0.01:
                l_norm[below] = avg - (avg - l_norm[below]) * (1.0 + contr) / (1.0 - contr + 1e-6)
            if (1 - avg) > 0.01:
                l_norm[above] = avg + (l_norm[above] - avg) * (1.0 + contr) / (1.0 - contr + 1e-6)
            l = np.clip(l_norm, 0, 1) * 255.0
        else:  # rapidraw
            # Perceptual gamma 2.2 S-curve
            strength_factor = 2.0 ** (p['contrast'] * 1.25 / 100.0)
            l_perc = np.clip(l / 255.0, 1e-6, 1) ** (1 / 2.2)
            l_perc = l_mid / 255.0 + (l_perc - l_mid / 255.0) * strength_factor
            l = np.clip(l_perc ** 2.2, 0, 1) * 255.0

        # ---- HIGHLIGHTS ----
        if self.strategy_type == 'darktable':
            # Soft-light overlay (shadhi.c)
            hi_mask = np.clip((l - 191) / 64.0, 0, 1)
            l = l + hi_mask * p['highlights'] * 0.5
        elif self.strategy_type == 'rawtherapee':
            # Gamma: 4^(amount/100)
            hi_mask = np.clip((l - 191) / 64.0, 0, 1)
            gamma = 4.0 ** (p['highlights'] / 100.0)
            l = l + hi_mask * (l * (gamma - 1.0)) * 0.01
        else:  # rapidraw
            # tanh mask + gamma
            hi_mask = np.clip((l - 191) / 64.0, 0, 1)
            mult = 2.0 ** (p['highlights'] * 1.75 / 120.0)
            l = l + hi_mask * (l * (mult - 1.0)) * 0.01

        # ---- SHADOWS ----
        if self.strategy_type == 'darktable':
            sh_mask = np.clip((64 - l) / 64.0, 0, 1)
            l = l + sh_mask * p['shadows'] * 0.5
        elif self.strategy_type == 'rawtherapee':
            sh_mask = np.clip((64 - l) / 64.0, 0, 1)
            gamma = 4.0 ** (p['shadows'] / 100.0)
            l = l + sh_mask * (l * (gamma - 1.0)) * 0.01
        else:  # rapidraw
            sh_mask = np.clip((64 - l) / 64.0, 0, 1)
            mult = 2.0 ** (p['shadows'] * 1.5 / 120.0)
            l = l + sh_mask * (l * (mult - 1.0)) * 0.01

        # ---- WHITES ----
        w_mask = np.clip((l - 230) / 25.0, 0, 1)
        l = l + w_mask * p['whites'] * 0.5

        # ---- BLACKS ----
        bk_mask = np.clip((25 - l) / 25.0, 0, 1)
        l = l + bk_mask * p['blacks'] * 0.5

        # ---- TONE CURVE (parametric zones) ----
        tc_zones = [
            (p['param_shadows'], 0, 64),
            (p['param_darks'], 64, 128),
            (p['param_lights'], 128, 191),
            (p['param_highlights'], 191, 255),
        ]
        for val, lo, hi in tc_zones:
            zone_mask = np.clip(1.0 - np.abs(l - (lo + hi) / 2.0) / ((hi - lo) / 2.0), 0, 1)
            l = l + zone_mask * val * 0.3

        # ---- TEXTURE ----
        # Texture affects high-frequency detail — approximated as Laplacian boost
        # (small effect in forward model, mainly for cost function matching)

        # ---- CLARITY ----
        # Local contrast: approximate with unsharp mask on L
        if abs(p['clarity']) > 5:
            blur_size = max(3, min(l.shape[:2]) // 8) | 1
            l_blur = cv2.GaussianBlur(l, (blur_size, blur_size), 0)
            detail = l - l_blur
            l = l + detail * (p['clarity'] / 100.0)

        # ---- DEHAZE ----
        if abs(p['dehaze']) > 5:
            l = l + p['dehaze'] * 0.3

        # ---- TEMPERATURE ----
        if self.strategy_type == 'rapidraw':
            # RGB multiplier: R×(1+t×0.2), B×(1-t×0.2)
            t_normalized = (p['temperature'] - TEMP_DEFAULT) / 25.0
            b = b + t_normalized * 0.2 * -1  # B channel inversely related to temp
            a = a + t_normalized * 0.05  # small A shift
        else:
            # darktable/rawtherapee: Planck-based, approximated as B channel shift
            temp_shift = (p['temperature'] - TEMP_DEFAULT) / 100.0
            b = b + temp_shift * 0.5

        # ---- TINT ----
        a = a + p['tint'] * 0.3

        # ---- SATURATION ----
        a_centered = a - 128
        b_centered = b - 128
        if self.strategy_type == 'rapidraw':
            # mix(luma, rgb, 1+sat): scale chroma
            sat_factor = 1.0 + p['saturation'] / 100.0
        else:
            sat_factor = 1.0 + p['saturation'] / 100.0
        a = a_centered * sat_factor + 128
        b = b_centered * sat_factor + 128

        # ---- VIBRANCE ----
        chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
        max_chroma = np.percentile(chroma, 95) if chroma.max() > 0 else 1.0
        vib_weight = np.clip(1.0 - chroma / max(max_chroma, 1.0), 0, 1)
        if self.strategy_type == 'rapidraw':
            # RapidRAW: ×3.0 multiplier with skin dampening
            vib_factor = 1.0 + (p['vibrance'] * 3.0 / 100.0) * vib_weight
        else:
            vib_factor = 1.0 + (p['vibrance'] / 100.0) * vib_weight
        a = (a - 128) * vib_factor + 128
        b = (b - 128) * vib_factor + 128

        # ---- HSL adjustments ----
        hue_names = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']
        for color in hue_names:
            center, width = _GAUSSIAN_HUE_BINS[color]
            weight = _gaussian_hue_weight(h_ch, center, width)
            sig_pixels = weight > 0.1

            if sig_pixels.any():
                hue_val = p[f'hue_{color}']
                sat_val = p[f'sat_{color}']
                lum_val = p[f'lum_{color}']

                # Hue shift (convert from Lightroom scale to pixel hue)
                if abs(hue_val) > 0.5:
                    hue_shift = hue_val / 0.7  # inverse of the 0.7 extraction scale
                    h_ch = h_ch + weight * hue_shift * 0.5
                    h_ch = h_ch % 180

                # Saturation adjust per hue channel
                if abs(sat_val) > 0.5:
                    sat_scale = 1.0 + sat_val / 100.0
                    a_c = a - 128
                    b_c = b - 128
                    a = a + weight[:, :] * a_c * (sat_scale - 1.0)
                    b = b + weight[:, :] * b_c * (sat_scale - 1.0)

                # Luminance adjust per hue channel
                if abs(lum_val) > 0.5:
                    l = l + weight * lum_val * 0.3

        # ---- COLOR GRADING ----
        l_norm = np.clip(l / 255.0, 0, 1)
        feather = 0.15

        def smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0, 1)
            return t * t * (3 - 2 * t)

        cg_zones = {
            'shadow': 1.0 - smoothstep(0.1 - feather, 0.1 + feather, l_norm),
            'midtone': None,
            'highlight': smoothstep(0.5 - feather, 0.5 + feather, l_norm),
        }
        cg_zones['midtone'] = np.clip(1.0 - cg_zones['shadow'] - cg_zones['highlight'], 0, 1)

        blend = p['cg_blending'] / 100.0
        for zone_name, zone_mask in cg_zones.items():
            cg_hue = p[f'cg_{zone_name}_hue']
            cg_sat = p[f'cg_{zone_name}_sat']
            cg_lum = p[f'cg_{zone_name}_lum']

            if cg_sat > 0.5:
                # Apply color shift in a/b space
                hue_rad = np.radians(cg_hue)
                a_shift = np.sin(hue_rad) * cg_sat / 1.5 * blend
                b_shift = np.cos(hue_rad) * cg_sat / 1.5 * blend
                a = a + zone_mask * a_shift
                b = b + zone_mask * b_shift

            if abs(cg_lum) > 0.5:
                l = l + zone_mask * cg_lum * 0.5 * blend

        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        return np.stack([l, a, b], axis=-1)


def _ciede2000_cost(predicted_lab: np.ndarray, target_lab: np.ndarray) -> float:
    """Compute mean CIEDE2000 color distance between two LAB images.

    Uses the simplified CIEDE2000 formula for speed (no weighting factors).
    LAB values are in OpenCV scale: L=0-255, a,b=0-255 (centered at 128).
    """
    # Convert to standard LAB range: L=0-100, a,b=-128..+127
    p_l = predicted_lab[:, :, 0] * (100.0 / 255.0)
    p_a = predicted_lab[:, :, 1] - 128.0
    p_b = predicted_lab[:, :, 2] - 128.0
    t_l = target_lab[:, :, 0] * (100.0 / 255.0)
    t_a = target_lab[:, :, 1] - 128.0
    t_b = target_lab[:, :, 2] - 128.0

    # Chroma
    p_c = np.sqrt(p_a ** 2 + p_b ** 2)
    t_c = np.sqrt(t_a ** 2 + t_b ** 2)
    avg_c = (p_c + t_c) / 2.0

    # G factor (CIEDE2000 chroma correction)
    avg_c7 = avg_c ** 7
    G = 0.5 * (1.0 - np.sqrt(avg_c7 / (avg_c7 + 25.0 ** 7 + 1e-10)))

    # Adjusted a'
    p_a_prime = p_a * (1.0 + G)
    t_a_prime = t_a * (1.0 + G)

    # Adjusted chroma
    p_c_prime = np.sqrt(p_a_prime ** 2 + p_b ** 2)
    t_c_prime = np.sqrt(t_a_prime ** 2 + t_b ** 2)

    # Hue angle
    p_h = np.degrees(np.arctan2(p_b, p_a_prime)) % 360
    t_h = np.degrees(np.arctan2(t_b, t_a_prime)) % 360

    # Delta L', C', H'
    dl = t_l - p_l
    dc = t_c_prime - p_c_prime
    dh_angle = t_h - p_h
    dh_angle = np.where(dh_angle > 180, dh_angle - 360, dh_angle)
    dh_angle = np.where(dh_angle < -180, dh_angle + 360, dh_angle)
    dH = 2.0 * np.sqrt(p_c_prime * t_c_prime + 1e-10) * np.sin(np.radians(dh_angle / 2.0))

    # Weighting (simplified — SL=SC=SH=1 for speed)
    delta_e = np.sqrt(dl ** 2 + dc ** 2 + dH ** 2 + 1e-10)
    return float(np.mean(delta_e))


class _StrategyOptimizedBase(XMPExtractionStrategy):
    """Base class for strategy + optimizer combinations.

    Uses the strategy's extract_params as initial guess, then optimizes
    all 55 params with EnhancedForwardModel + CIEDE2000.
    """

    _base_strategy_cls = None  # Override in subclass
    _forward_model_type = None  # Override in subclass

    def _params_to_vector(self, params: LightroomParams) -> np.ndarray:
        d = params.to_dict()
        return np.array([float(d[name]) for name in _FULL_PARAM_NAMES])

    def _vector_to_params(self, x: np.ndarray) -> LightroomParams:
        d = {}
        for i, name in enumerate(_FULL_PARAM_NAMES):
            val = x[i]
            if name == 'exposure':
                d[name] = round(float(val), 2)
            elif name == 'sharpen_radius':
                d[name] = round(float(val), 1)
            elif name in ('cg_shadow_hue', 'cg_midtone_hue', 'cg_highlight_hue'):
                d[name] = int(val) % 360
            else:
                d[name] = int(np.clip(val, _FULL_BOUNDS[i][0], _FULL_BOUNDS[i][1]))
        return LightroomParams(**d)

    def extract_params(self, target_image, reference_image, strength):
        from scipy.optimize import minimize

        # Step 1: Get initial guess from base strategy
        base = self._base_strategy_cls()
        initial_params = base.extract_params(target_image, reference_image, strength)

        logger.info("[%s+opt] Base strategy done, starting optimizer (55 params)...",
                    self._forward_model_type)

        # Step 2: Prepare forward model (128px downsample for speed)
        model = EnhancedForwardModel(target_image, self._forward_model_type)

        # Step 3: Prepare reference in LAB (downsampled to match model)
        max_dim = max(target_image.shape[:2])
        scale = min(1.0, 96.0 / max_dim)
        ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        if scale < 1.0:
            ref_lab_small = cv2.resize(ref_lab, None, fx=scale, fy=scale)
        else:
            ref_lab_small = ref_lab

        # Step 4: Cost function (CIEDE2000)
        eval_count = [0]

        def cost(x):
            eval_count[0] += 1
            predicted = model.apply(x)
            return _ciede2000_cost(predicted, ref_lab_small)

        x0 = self._params_to_vector(initial_params)
        initial_cost = cost(x0)
        logger.info("[%s+opt] Initial CIEDE2000 cost: %.3f", self._forward_model_type, initial_cost)

        # Step 5: Optimize (50 iterations max for speed)
        try:
            result = minimize(
                cost, x0, method='L-BFGS-B', bounds=_FULL_BOUNDS,
                options={'maxiter': 50, 'ftol': 1e-5},
            )
            logger.info("[%s+opt] Done in %d evals, final cost: %.3f (was %.3f)",
                        self._forward_model_type, eval_count[0], result.fun, initial_cost)
            return self._vector_to_params(result.x)
        except Exception as e:
            logger.warning("Optimization failed for %s: %s, using base strategy results",
                           self._forward_model_type, e)
            return initial_params


class DarktableOptimizedStrategy(_StrategyOptimizedBase):
    """darktable inverse as initial guess + optimize all 55 params with darktable forward model + CIEDE2000."""
    _base_strategy_cls = DarktableStrategy
    _forward_model_type = 'darktable'


class RawTherapeeOptimizedStrategy(_StrategyOptimizedBase):
    """RawTherapee inverse as initial guess + optimize all 55 params with RawTherapee forward model + CIEDE2000."""
    _base_strategy_cls = RawTherapeeStrategy
    _forward_model_type = 'rawtherapee'


class RapidRAWOptimizedStrategy(_StrategyOptimizedBase):
    """RapidRAW inverse as initial guess + optimize all 55 params with RapidRAW forward model + CIEDE2000."""
    _base_strategy_cls = RapidRAWStrategy
    _forward_model_type = 'rapidraw'


class RapidRAWExactInverseOptimizedStrategy(_StrategyOptimizedBase):
    """RapidRAW exact inverse as initial guess + optimize all 55 params with CIEDE2000."""
    _base_strategy_cls = RapidRAWExactInverseStrategy
    _forward_model_type = 'rapidraw'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[XMPExtractionStrategy]] = {
    'basic': BasicStrategy,
    'color_science': ColorScienceStrategy,
    'optimization': OptimizationStrategy,
    'basic_optimized': BasicOptimizedStrategy,
    'deep_preset': DeepPresetStrategy,
    'darktable': DarktableStrategy,
    'rawtherapee': RawTherapeeStrategy,
    'rapidraw': RapidRAWStrategy,
    'darktable_optimized': DarktableOptimizedStrategy,
    'rawtherapee_optimized': RawTherapeeOptimizedStrategy,
    'rapidraw_optimized': RapidRAWOptimizedStrategy,
    'rapidraw_exact_inverse': RapidRAWExactInverseStrategy,
    'rapidraw_exact_inverse_optimized': RapidRAWExactInverseOptimizedStrategy,
}
