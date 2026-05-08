"""
All API routes for the LuminaFix Style Transfer web application.

Organized into sections:
- Dependencies (service factories)
- Pages
- References
- Uploads
- Processing
- Export (XMP/ZIP)
- Models
- NILUT
"""

import gc
import json
import os
import uuid
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.config import get_config
from src.batch_processor import BatchProcessor
from src.multi_method_processor import MultiMethodProcessor, NILUT_VARIANT_KEYS
from src.nilut_model_service import NILUTModelService
from src.reference_db import get_all_presets, get_categories, TRAINING_REF_DIR
from src.image_service import ImageService
from src.model_manager import get_model_manager
from src.device_manager import get_device_manager
from src.utils import parse_bool, load_image_as_cv2, IMAGE_EXTENSIONS
from src.xmp_generator import XMPPresetGenerator, LightroomParams
from src.xmp_parser import parse_xmp_to_params
from src.export_service import ExportService, ExportItem

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _log_memory(label: str):
    """Log current process memory usage in MB."""
    try:
        # Read from /proc on Linux (Render)
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    logger.info("MEMORY [%s]: %d MB", label, rss_kb // 1024)
                    return
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

@lru_cache()
def get_web_config():
    return get_config().web


def get_nilut_service() -> NILUTModelService:
    return NILUTModelService()


def get_image_service() -> ImageService:
    return ImageService()


@lru_cache(maxsize=8)
def get_processor(
    color_strength: float = 0.7,
    luminance_strength: float = 0.0,
    enable_skin_protection: bool = True,
    enable_neon_protection: bool = True,
    enable_lip_protection: bool = False,
    enable_eye_protection: bool = True,
) -> MultiMethodProcessor:
    return MultiMethodProcessor(
        color_strength=color_strength,
        luminance_strength=luminance_strength,
        enable_skin_protection=enable_skin_protection,
        enable_neon_protection=enable_neon_protection,
        enable_lip_protection=enable_lip_protection,
        enable_eye_protection=enable_eye_protection,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@router.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/results")
async def results_page(request: Request):
    return templates.TemplateResponse(request=request, name="results.html")


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

@router.get("/api/references")
async def get_references():
    references = []

    # Load preset references from database (10 per category)
    presets = get_all_presets()
    for p in presets:
        base = f"/reference/preset/{p['folder_name']}/{p['filename']}"
        references.append({
            "name": p["display_name"],
            "filename": p["filename"],
            "url": base,
            "thumb_url": f"{base}?thumb=1",
            "type": "preset",
            "category": p["category"],
            "deletable": False,
        })

    # User-uploaded references are tracked client-side per session only.
    # They are NOT listed here from the filesystem.

    categories = get_categories()
    return JSONResponse({"references": references, "categories": ["All"] + categories})


_IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=604800, immutable"}
_THUMBNAIL_SIZE = 400
_THUMBNAIL_QUALITY = 80


def _get_or_make_thumbnail(original_path: Path) -> Path:
    """Return a cached thumbnail next to the original. Generate on first call."""
    thumb_dir = original_path.parent / ".thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    thumb_path = thumb_dir / (original_path.stem + ".jpg")

    if thumb_path.exists() and thumb_path.stat().st_mtime >= original_path.stat().st_mtime:
        return thumb_path

    img = cv2.imread(str(original_path))
    if img is None:
        return original_path  # fall back to original on decode failure

    h, w = img.shape[:2]
    scale = _THUMBNAIL_SIZE / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, _THUMBNAIL_QUALITY])
    return thumb_path


@router.get("/reference/preset/{folder_name}/{filename}")
async def get_preset_reference_image(folder_name: str, filename: str, thumb: int = 0):
    file_path = TRAINING_REF_DIR / folder_name / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Preset reference image not found")
    if thumb:
        file_path = _get_or_make_thumbnail(file_path)
    return FileResponse(file_path, headers=_IMAGE_CACHE_HEADERS)


@router.get("/reference/{filename}")
async def get_reference_image(filename: str):
    config = get_web_config()
    file_path = config.reference_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Reference image not found")
    return FileResponse(file_path, headers=_IMAGE_CACHE_HEADERS)


@router.get("/reference/user/{filename}")
async def get_user_reference_image(filename: str):
    config = get_web_config()
    file_path = config.user_reference_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="User reference image not found")
    return FileResponse(file_path, headers=_IMAGE_CACHE_HEADERS)


@router.post("/api/references/upload")
async def upload_reference_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: jpg, jpeg, png, bmp, tiff, webp")

    config = get_web_config()
    unique_id = uuid.uuid4().hex[:8]
    safe_name = Path(file.filename).stem.replace(" ", "_")
    filename = f"{safe_name}_{unique_id}{suffix}"
    file_path = config.user_reference_dir / filename

    contents = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    return JSONResponse({
        "success": True, "name": Path(filename).stem, "filename": filename,
        "url": f"/reference/user/{filename}", "type": "user",
        "deletable": True, "original_name": file.filename
    })


@router.delete("/api/references/{filename}")
async def delete_reference_image(filename: str):
    config = get_web_config()
    file_path = config.user_reference_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Reference image not found")

    file_path.unlink()
    get_nilut_service().delete_model(Path(filename).stem)
    return JSONResponse({"success": True, "message": f"Reference image '{filename}' deleted"})


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

@router.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    filename = get_image_service().save_upload(file.filename, contents)
    return JSONResponse({"filename": filename, "url": f"/uploads/{filename}", "original_name": file.filename})


@router.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    file_path = get_image_service().resolve_upload_or_processed(filename)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@router.post("/api/save")
async def save_image(output_filename: str = Form(...), save_name: Optional[str] = Form(None)):
    source_path = get_web_config().processed_dir / output_filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(source_path, filename=save_name or output_filename, media_type="image/jpeg")


@router.delete("/api/cleanup/{filename}")
async def cleanup_file(filename: str):
    get_image_service().cleanup(filename)
    return JSONResponse({"success": True})


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

@router.get("/api/methods")
async def get_available_methods():
    return JSONResponse({"methods": get_processor().list_available_methods()})


@router.get("/api/segments")
async def get_segments():
    """List the segment names used for per-region NILUT tuning."""
    try:
        from src.segmentation import SEGMENT_NAMES, is_available as seg_available
        return JSONResponse({
            "segments": list(SEGMENT_NAMES),
            "available": bool(seg_available()),
        })
    except Exception as e:
        return JSONResponse({"segments": [], "available": False, "error": str(e)})


@router.post("/api/process")
async def process_image(
    target_filename: str = Form(...),
    reference_filename: str = Form(...),
    color_strength: float = Form(0.7),
    luminance_strength: float = Form(0.0),
    skin_protection: str = Form("true"),
    neon_protection: str = Form("true"),
    lip_protection: str = Form("false"),
    eye_protection: str = Form("true")
):
    skin_prot = parse_bool(skin_protection)
    neon_prot = parse_bool(neon_protection)
    lip_prot = parse_bool(lip_protection)
    eye_prot = parse_bool(eye_protection)

    config = get_web_config()
    image_service = get_image_service()
    target_path = config.upload_dir / target_filename
    reference_path = image_service.resolve_reference_path(reference_filename)

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found")
    if reference_path is None:
        raise HTTPException(status_code=404, detail="Reference image not found")

    try:
        processor = BatchProcessor(
            color_strength=color_strength, luminance_strength=luminance_strength,
            enable_skin_protection=skin_prot, enable_neon_protection=neon_prot,
            enable_lip_protection=lip_prot, enable_eye_protection=eye_prot, jpeg_quality=95
        )
        processor.load_reference(str(reference_path))

        target_image = cv2.imread(str(target_path))
        if target_image is None:
            raise HTTPException(status_code=500, detail="Could not load target image")

        result_image = processor.process_single(target_image)
        ref_name = Path(reference_filename).stem
        target_name = Path(target_filename).stem
        output_filename = image_service.save_processed(result_image, "reinhard", ref_name, target_name)

        return JSONResponse({
            "success": True, "output_filename": output_filename,
            "output_url": f"/uploads/{output_filename}",
            "input_url": f"/uploads/{target_filename}",
            "reference_url": f"/reference/{reference_filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/process-all")
async def process_image_all_methods(
    target_filename: str = Form(...),
    reference_filename: str = Form(...),
    color_strength: float = Form(0.7),
    luminance_strength: float = Form(0.0),
    skin_protection: str = Form("true"),
    neon_protection: str = Form("true"),
    lip_protection: str = Form("false"),
    eye_protection: str = Form("true"),
    methods: str = Form('["reinhard", "nilut"]'),
    nilut_mode: str = Form("per_reference"),
    nilut_models: str = Form('["latest"]'),
    per_segment_strengths: str = Form("")
):
    try:
        selected_methods = json.loads(methods)
    except (json.JSONDecodeError, TypeError):
        selected_methods = ["reinhard", "nilut"]
    try:
        selected_nilut_models = json.loads(nilut_models)
    except (json.JSONDecodeError, TypeError):
        selected_nilut_models = ["latest"]

    # Per-segment NILUT strengths: {"sky": 1.0, "grass": 0.2, "skin": 0.3, ...}
    parsed_segment_strengths = None
    if per_segment_strengths:
        try:
            raw = json.loads(per_segment_strengths)
            if isinstance(raw, dict):
                parsed_segment_strengths = {str(k): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_segment_strengths = None

    skin_prot = parse_bool(skin_protection)
    neon_prot = parse_bool(neon_protection)
    lip_prot = parse_bool(lip_protection)
    eye_prot = parse_bool(eye_protection)

    config = get_web_config()
    image_service = get_image_service()
    nilut_service = get_nilut_service()
    target_path = config.upload_dir / target_filename
    reference_path = image_service.resolve_reference_path(reference_filename)

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found")
    if reference_path is None:
        raise HTTPException(status_code=404, detail="Reference image not found")

    try:
        _log_memory("process-all: start")

        target_image = load_image_as_cv2(target_path)
        reference_image = load_image_as_cv2(reference_path)
        if target_image is None:
            raise HTTPException(status_code=500, detail="Could not load target image")
        if reference_image is None:
            raise HTTPException(status_code=500, detail="Could not load reference image")

        _log_memory("process-all: images loaded")

        processor = get_processor(
            color_strength=color_strength, luminance_strength=luminance_strength,
            enable_skin_protection=skin_prot, enable_neon_protection=neon_prot,
            enable_lip_protection=lip_prot, enable_eye_protection=eye_prot
        )

        ref_name = Path(reference_filename).stem
        target_name = Path(target_filename).stem
        nilut_methods = [m for m in selected_methods if m in NILUT_VARIANT_KEYS]
        non_nilut_methods = [m for m in selected_methods if m not in NILUT_VARIANT_KEYS]

        all_results = {}

        if non_nilut_methods:
            all_results.update(processor.process_all(
                target_image, reference_image,
                selected_methods=non_nilut_methods,
                reference_name=ref_name, nilut_mode=nilut_mode
            ))
            gc.collect()
            _log_memory("process-all: non-nilut done")

        if nilut_methods and selected_nilut_models:
            from src.transfers.base import TransferResult
            masks = processor.compute_masks(target_image)
            gc.collect()
            _log_memory("process-all: masks computed")

            for model_id in selected_nilut_models:
                model_path = nilut_service.get_universal_model_path(model_id)
                model_display_name = nilut_service.format_model_display_name(model_id)

                if not model_path.exists():
                    for method in nilut_methods:
                        rk = f"{method}_{model_id}"
                        all_results[rk] = TransferResult.error_result(
                            method_id=rk,
                            method_name=f"{method.replace('_', ' ').title()} ({model_display_name})",
                            error=f"Model not found: {model_path}"
                        )
                    continue

                all_results.update(processor.process_nilut_variants(
                    target_image=target_image, reference_image=reference_image,
                    masks=masks, model_path=str(model_path),
                    model_display_name=model_display_name, model_id=model_id,
                    requested_variants=nilut_methods, color_strength=color_strength,
                    per_segment_strengths=parsed_segment_strengths,
                ))
                gc.collect()
                _log_memory(f"process-all: nilut model {model_id} done")

            del masks
            gc.collect()

        _log_memory("process-all: all processing done")
        outputs, errors = {}, {}
        for method_id, result in all_results.items():
            if result.success and result.image is not None:
                output_filename = image_service.save_processed(result.image, method_id, ref_name, target_name)
                model_version = None
                if method_id.startswith("nilut"):
                    parts = method_id.rsplit("_", 1)
                    if len(parts) == 2 and (parts[1] == "latest" or parts[1].replace("_", "").isdigit()):
                        model_version = parts[1]
                outputs[method_id] = {
                    "filename": output_filename, "url": f"/uploads/{output_filename}",
                    "method_name": result.method_name,
                    "time_ms": round(result.processing_time_ms, 1),
                    "model_version": model_version
                }
            else:
                errors[method_id] = {"method_name": result.method_name, "error": result.error or "Unknown error"}

        # Free images and processing data from memory
        del all_results, target_image, reference_image
        gc.collect()

        return JSONResponse({
            "success": len(outputs) > 0, "outputs": outputs, "errors": errors,
            "input_url": f"/uploads/{target_filename}",
            "reference_url": f"/reference/{reference_filename}"
        })

    except Exception as e:
        logger.exception("Error in process_image_all_methods")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# XMP Preset Upload & Apply
# ---------------------------------------------------------------------------

@router.post("/api/xmp/upload")
async def upload_xmp_preset(file: UploadFile = File(...)):
    """Parse an uploaded XMP preset file and return its parameters."""
    if not file.filename.lower().endswith('.xmp'):
        raise HTTPException(status_code=400, detail="File must be an .xmp preset file")

    try:
        contents = await file.read()
        xmp_text = contents.decode('utf-8')
        params, preset_name = parse_xmp_to_params(xmp_text)
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "preset_name": preset_name or Path(file.filename).stem,
            "params": params.to_dict(),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error parsing XMP preset")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/process-xmp")
async def process_image_with_xmp(
    target_filename: str = Form(...),
    xmp_params: str = Form(...),
):
    """Apply XMP preset parameters to a target image."""
    config = get_web_config()
    image_service = get_image_service()
    target_path = config.upload_dir / target_filename

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    try:
        params_dict = json.loads(xmp_params)
        params = LightroomParams(**params_dict)
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid XMP params: {e}")

    try:
        import numpy as np
        from src.xmp_strategies import TEMP_DEFAULT, HSL_RANGES, _get_hue_mask

        target_image = load_image_as_cv2(target_path)
        if target_image is None:
            raise HTTPException(status_code=500, detail="Could not load target image")

        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = target_lab[:, :, 0].copy()
        a = target_lab[:, :, 1].copy()
        b = target_lab[:, :, 2].copy()

        # --- Basic Tone (exact inverse of BasicStrategy extraction) ---

        # Exposure: extract used (l_shift / 255.0) * 6.0 → inverse: l_shift = exposure * 255 / 6.0
        if params.exposure != 0:
            l_shift = params.exposure * 255.0 / 6.0
            l = l + l_shift

        # Contrast: extract used l_std_change * 2.0 → inverse: std change = contrast / 2.0
        if params.contrast != 0:
            l_std = float(np.std(l))
            if l_std > 1e-6:
                target_std = l_std + params.contrast / 2.0
                if target_std > 0:
                    l_mean = np.mean(l)
                    l = l_mean + (l - l_mean) * (target_std / l_std)

        # Highlights: extract used (r_upper - t_upper) * 1.0 → direct shift on top 25%
        if params.highlights != 0:
            p75 = np.percentile(l, 75)
            hi_mask = (l >= p75).astype(np.float32)
            l = l + hi_mask * params.highlights

        # Shadows: extract used (r_lower - t_lower) * 1.0 → direct shift on bottom 25%
        if params.shadows != 0:
            p25 = np.percentile(l, 25)
            sh_mask = (l <= p25).astype(np.float32)
            l = l + sh_mask * params.shadows

        # Whites: extract used presence whites logic on top 5%
        if params.whites != 0:
            p95 = np.percentile(l, 95)
            w_mask = (l >= p95).astype(np.float32)
            l = l + w_mask * params.whites

        # Blacks: extract used presence blacks logic on bottom 5%
        if params.blacks != 0:
            p5 = np.percentile(l, 5)
            bk_mask = (l <= p5).astype(np.float32)
            l = l + bk_mask * params.blacks

        # Temperature: extract used b_shift * 100 → inverse: b_shift = (temp - 5500) / 100
        if params.temperature != TEMP_DEFAULT:
            b_shift = (params.temperature - TEMP_DEFAULT) / 100.0
            b = b + b_shift

        # Tint: extract used a_shift * 3.5 → inverse: a_shift = tint / 3.5
        if params.tint != 0:
            a_shift = params.tint / 3.5
            a = a + a_shift

        # Saturation: extract used (sat_shift / 255.0) * 130.0 → inverse: sat_shift = sat * 255 / 130
        # Apply in HSV space to match extraction
        if params.saturation != 0:
            sat_shift = params.saturation * 255.0 / 130.0
            result_bgr_temp = cv2.cvtColor(
                np.stack([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8),
                cv2.COLOR_LAB2BGR
            )
            hsv = cv2.cvtColor(result_bgr_temp, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
            result_bgr_temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            result_lab_temp = cv2.cvtColor(result_bgr_temp, cv2.COLOR_BGR2LAB).astype(np.float32)
            a = result_lab_temp[:, :, 1]
            b = result_lab_temp[:, :, 2]

        # Vibrance: extract used saturation_val * 0.6 → similar but on low-chroma pixels
        if params.vibrance != 0:
            vib_sat_shift = params.vibrance * 255.0 / 130.0
            a_c = a - 128
            b_c = b - 128
            chroma = np.sqrt(a_c**2 + b_c**2)
            max_chroma = np.percentile(chroma, 95) if chroma.max() > 0 else 1.0
            vib_weight = np.clip(1.0 - chroma / max(float(max_chroma), 1.0), 0, 1)
            scale = 1.0 + (vib_sat_shift / 255.0) * vib_weight
            a = (a - 128) * scale + 128
            b = (b - 128) * scale + 128

        # --- Tone Curve: extract used direct L shift per zone ---
        tc_zones = [
            (params.param_shadows,    0,  25),
            (params.param_darks,     25,  50),
            (params.param_lights,    50,  75),
            (params.param_highlights, 75, 100),
        ]
        for val, plo, phi in tc_zones:
            if val == 0:
                continue
            lo_val = np.percentile(l, plo)
            hi_val = np.percentile(l, phi)
            zone_mask = ((l >= lo_val) & (l <= hi_val)).astype(np.float32)
            l = l + zone_mask * val

        # --- Clarity: extract measured std diff in midtones * 2.0 ---
        if params.clarity != 0:
            p25 = np.percentile(l, 25)
            p75 = np.percentile(l, 75)
            mid_mask = ((l >= p25) & (l <= p75)).astype(np.float32)
            mid_pixels = l[mid_mask > 0]
            if len(mid_pixels) > 0:
                mid_mean = np.mean(mid_pixels)
                mid_std = np.std(mid_pixels)
                if mid_std > 1e-6:
                    target_std = mid_std + params.clarity / 2.0
                    if target_std > 0:
                        l = np.where(mid_mask > 0,
                                     mid_mean + (l - mid_mean) * (target_std / mid_std),
                                     l)

        # --- Texture: extract measured Laplacian variance diff ---
        if params.texture != 0:
            l_blur = cv2.GaussianBlur(l, (0, 0), sigmaX=5)
            fine_detail = l - l_blur
            l = l + fine_detail * (params.texture / 100.0)

        # --- Dehaze: extract measured dark channel diff ---
        if params.dehaze != 0:
            dehaze_shift = params.dehaze * 255.0 / 120.0
            l_uint8 = np.clip(l, 0, 255).astype(np.uint8)
            bgr_temp = cv2.cvtColor(
                np.stack([l_uint8, np.clip(a, 0, 255).astype(np.uint8),
                          np.clip(b, 0, 255).astype(np.uint8)], axis=-1),
                cv2.COLOR_LAB2BGR
            )
            b_ch, g_ch, r_ch = cv2.split(bgr_temp)
            min_ch = np.minimum(np.minimum(b_ch, g_ch), r_ch).astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dark_ch = cv2.erode(min_ch, kernel)
            # Increase dark channel = reduce haze
            l = l + (dark_ch / 255.0) * dehaze_shift * 0.5

        # --- HSL Adjustments (inverse of _extract_hsl) ---
        target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_ch = target_hsv[:, :, 0]
        s_ch = target_hsv[:, :, 1]
        v_ch = target_hsv[:, :, 2]

        for color in HSL_RANGES:
            hue_adj = getattr(params, f'hue_{color}', 0)
            sat_adj = getattr(params, f'sat_{color}', 0)
            lum_adj = getattr(params, f'lum_{color}', 0)
            if hue_adj == 0 and sat_adj == 0 and lum_adj == 0:
                continue

            mask = _get_hue_mask(h_ch, color).astype(np.float32)
            if mask.sum() < 100:
                continue

            # Lum: extract used (val_diff / 255.0) * 130.0 → inverse
            if lum_adj != 0:
                val_shift = lum_adj * 255.0 / 130.0
                l = l + mask * val_shift

            # Sat: extract used (sat_diff / 255.0) * 130.0 → inverse
            if sat_adj != 0:
                sat_shift_ch = sat_adj * 255.0 / 130.0
                a_c = a - 128
                b_c = b - 128
                chroma = np.sqrt(a_c**2 + b_c**2)
                chroma_safe = np.maximum(chroma, 1e-6)
                scale = (chroma + mask * sat_shift_ch) / chroma_safe
                scale = np.clip(scale, 0, 5)
                a = (a - 128) * scale + 128
                b = (b - 128) * scale + 128

        # --- Color Grading (inverse of _extract_color_grading) ---
        cg_zones = [
            ('shadow',    0,  33),
            ('midtone',  33,  66),
            ('highlight', 66, 100),
        ]
        for zone, plo, phi in cg_zones:
            cg_hue = getattr(params, f'cg_{zone}_hue', 0)
            cg_sat = getattr(params, f'cg_{zone}_sat', 0)
            cg_lum = getattr(params, f'cg_{zone}_lum', 0)
            if cg_sat == 0 and cg_lum == 0:
                continue

            lo_val = np.percentile(l, plo)
            hi_val = np.percentile(l, phi)
            zone_mask = ((l >= lo_val) & (l <= hi_val)).astype(np.float32)

            # Extract used: hue = atan2(a_shift, b_shift), sat = sqrt(a^2+b^2)*1.5
            # Inverse: a_shift = sin(hue) * sat/1.5, b_shift = cos(hue) * sat/1.5
            if cg_sat > 0:
                hue_rad = np.radians(float(cg_hue))
                a_shift = np.sin(hue_rad) * cg_sat / 1.5
                b_shift = np.cos(hue_rad) * cg_sat / 1.5
                a = a + zone_mask * a_shift
                b = b + zone_mask * b_shift

            # Extract used: lum_shift * 0.5 → inverse
            if cg_lum != 0:
                l = l + zone_mask * cg_lum / 0.5

        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        result_lab = np.stack([l, a, b], axis=-1).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        target_name = Path(target_filename).stem
        output_filename = image_service.save_processed(result_bgr, "xmp_preset", "xmp", target_name)

        return JSONResponse({
            "success": True,
            "output_filename": output_filename,
            "output_url": f"/uploads/{output_filename}",
            "input_url": f"/uploads/{target_filename}",
            "method_name": "XMP Preset",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error applying XMP preset")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Export (XMP / ZIP)
# ---------------------------------------------------------------------------

@router.post("/api/export/params")
async def export_params(
    target_filename: str = Form(...),
    reference_filename: str = Form(...),
    color_strength: float = Form(0.7),
    output_filename: str = Form(""),
    xmp_strategy: str = Form("color_science"),
):
    """Extract Lightroom parameters by comparing target to its processed result."""
    config = get_web_config()
    image_service = get_image_service()
    target_path = config.upload_dir / target_filename

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    if not output_filename:
        logger.error("No output_filename provided for export params")
        raise HTTPException(status_code=400, detail="Processed result image filename is required")
    compare_path = image_service.resolve_upload_or_processed(output_filename)
    if compare_path is None:
        logger.error("Processed result image not found: %s", output_filename)
        raise HTTPException(status_code=404, detail=f"Processed result image not found: {output_filename}")

    try:
        target_image = load_image_as_cv2(target_path)
        compare_image = load_image_as_cv2(compare_path)
        if target_image is None or compare_image is None:
            raise HTTPException(status_code=500, detail="Could not load images")

        generator = XMPPresetGenerator()
        # Result image already embodies the desired strength, so use 1.0
        # to capture the exact delta without dampening or inflating.
        xmp_method = xmp_strategy if xmp_strategy in ("basic", "color_science", "basic_optimized", "darktable", "rawtherapee", "rapidraw", "darktable_optimized", "rawtherapee_optimized", "rapidraw_optimized", "rapidraw_exact_inverse", "rapidraw_exact_inverse_optimized") else "color_science"
        params = generator.extract_params(target_image, compare_image, strength=1.0, method=xmp_method)
        return JSONResponse({"params": params.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error extracting export params")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/export/xmp")
async def export_xmp(
    target_filename: str = Form(...),
    reference_filename: str = Form(...),
    method_id: str = Form(...),
    color_strength: float = Form(0.7),
    output_filename: str = Form(""),
    xmp_strategy: str = Form("color_science"),
):
    """Download a single XMP preset file."""
    logger.info("=== XMP EXPORT REQUEST === strategy=%s target=%s ref=%s method=%s output=%s strength=%s",
                xmp_strategy, target_filename, reference_filename, method_id, output_filename, color_strength)
    config = get_web_config()
    image_service = get_image_service()
    target_path = config.upload_dir / target_filename

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found")

    if not output_filename:
        logger.error("No output_filename provided for XMP export")
        raise HTTPException(status_code=400, detail="Processed result image filename is required")
    compare_path = image_service.resolve_upload_or_processed(output_filename)
    if compare_path is None:
        logger.error("Processed result image not found: %s", output_filename)
        raise HTTPException(status_code=404, detail=f"Processed result image not found: {output_filename}")

    try:
        target_image = load_image_as_cv2(target_path)
        compare_image = load_image_as_cv2(compare_path)
        if target_image is None or compare_image is None:
            raise HTTPException(status_code=500, detail="Could not load images")

        generator = XMPPresetGenerator()
        xmp_method = xmp_strategy if xmp_strategy in ("basic", "color_science", "basic_optimized", "darktable", "rawtherapee", "rapidraw", "darktable_optimized", "rawtherapee_optimized", "rapidraw_optimized", "rapidraw_exact_inverse", "rapidraw_exact_inverse_optimized") else "color_science"
        params = generator.extract_params(target_image, compare_image, strength=1.0, method=xmp_method)

        logger.info("XMP params (%s): temp=%s tint=%s exp=%s con=%s hi=%s sh=%s sat=%s vib=%s", xmp_method,
                     params.temperature, params.tint, params.exposure, params.contrast,
                     params.highlights, params.shadows, params.saturation, params.vibrance)

        ref_name = Path(reference_filename).stem
        strength_pct = int(color_strength * 100)
        strategy_labels = {
            'color_science': 'colorsci',
            'basic': 'basic',
            'basic_optimized': 'basic+optimizer',
            'darktable': 'darktable',
            'rawtherapee': 'rawtherapee',
            'rapidraw': 'rapidraw',
            'darktable_optimized': 'darktable+optimizer',
            'rawtherapee_optimized': 'rawtherapee+optimizer',
            'rapidraw_optimized': 'rapidraw+optimizer',
            'rapidraw_exact_inverse': 'rapidraw-exact',
            'rapidraw_exact_inverse_optimized': 'rapidraw-exact+optimizer',
        }
        strategy_label = strategy_labels.get(xmp_method, xmp_method)
        preset_name = f"{ref_name}_{strategy_label}"
        xmp_content = generator.generate_xmp(params, preset_name)

        xmp_filename = f"{preset_name}.xmp"
        tmp_path = config.processed_dir / xmp_filename
        tmp_path.write_text(xmp_content, encoding='utf-8')

        return FileResponse(
            tmp_path,
            filename=xmp_filename,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating XMP preset")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/export/zip")
async def export_zip(request: Request):
    """Download ZIP bundle of images + XMP presets."""
    try:
        body = await request.json()
        items_data = body.get("items", [])
        xmp_strategy = body.get("xmp_strategy", "color_science")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not items_data:
        raise HTTPException(status_code=400, detail="No items to export")

    xmp_method = xmp_strategy if xmp_strategy in ("basic", "color_science", "basic_optimized", "darktable", "rawtherapee", "rapidraw", "darktable_optimized", "rawtherapee_optimized", "rapidraw_optimized", "rapidraw_exact_inverse", "rapidraw_exact_inverse_optimized") else "color_science"
    config = get_web_config()
    image_service = get_image_service()
    generator = XMPPresetGenerator()
    export_items = []

    for item in items_data:
        output_filename = item.get("output_filename", "")
        method_id = item.get("method_id", "unknown")
        reference_name = item.get("reference_name", "ref")
        strength = int(item.get("strength", 70))
        target_filename = item.get("target_filename", "")
        reference_filename = item.get("reference_filename", "")

        # Resolve image path
        image_path = image_service.resolve_upload_or_processed(output_filename)
        if image_path is None:
            continue

        # Extract params by comparing target to its processed result
        params = LightroomParams()
        if target_filename:
            target_path = config.upload_dir / target_filename
            compare_path = image_path  # already resolved output image
            if compare_path is None:
                logger.warning("Processed result image not found for %s, skipping params", output_filename)
            if target_path.exists() and compare_path is not None:
                try:
                    target_image = load_image_as_cv2(target_path)
                    compare_image = load_image_as_cv2(compare_path)
                    if target_image is not None and compare_image is not None:
                        params = generator.extract_params(
                            target_image, compare_image,
                            strength=1.0,
                            method=xmp_method,
                        )
                except Exception:
                    logger.warning("Could not extract params for %s", output_filename)

        export_items.append(ExportItem(
            image_path=image_path,
            method_name=method_id,
            reference_name=reference_name,
            strength=strength,
            params=params,
            target_name=Path(target_filename).stem if target_filename else "",
        ))

    if not export_items:
        raise HTTPException(status_code=400, detail="No valid items to export")

    try:
        service = ExportService()
        zip_buffer = service.create_export_zip(export_items, xmp_strategy=xmp_method)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=LuminaFix_Export.zip"},
        )
    except Exception as e:
        logger.exception("Error creating export ZIP")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/api/models/status")
async def get_models_status():
    try:
        mm = get_model_manager()
        dm = get_device_manager()
        return JSONResponse({"models": mm.list_available_models(), "device": dm.get_device_info()})
    except Exception as e:
        return JSONResponse({"models": {}, "device": {"error": str(e)}, "error": str(e)})


@router.post("/api/models/download/{model_id}")
async def download_model(model_id: str):
    try:
        path = get_model_manager().ensure_model(model_id)
        return JSONResponse({"success": True, "model_id": model_id, "path": str(path)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# NILUT
# ---------------------------------------------------------------------------

@router.get("/api/nilut/status")
async def get_nilut_status():
    config = get_web_config()
    nilut_service = get_nilut_service()
    meta = nilut_service.get_meta()
    references = []

    for ref_dir, ref_type in [(config.reference_dir, "preset"), (config.user_reference_dir, "user")]:
        if not ref_dir.exists():
            continue
        for f in sorted(ref_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            ref_name = f.stem
            model_path = nilut_service.get_model_path(ref_name)
            model_info = meta.get(ref_name, {})
            references.append({
                "name": ref_name, "filename": f.name, "type": ref_type,
                "has_model": model_path.exists(),
                "last_trained": model_info.get("last_trained"),
                "training_samples": model_info.get("training_samples", 0),
                "epochs": model_info.get("epochs", 0)
            })

    return JSONResponse({"references": references, "models_dir": str(config.nilut_models_dir)})


@router.post("/api/nilut/train")
async def train_nilut_model(
    reference_filename: str = Form(...),
    use_all_references_as_samples: str = Form("true")
):
    config = get_web_config()
    nilut_service = get_nilut_service()

    image_service = get_image_service()
    reference_path = image_service.resolve_reference_path(reference_filename)
    if reference_path is None:
        raise HTTPException(status_code=404, detail="Reference image not found")

    try:
        reference_image = cv2.imread(str(reference_path))
        if reference_image is None:
            raise HTTPException(status_code=500, detail="Could not load reference image")

        sample_images = []
        if use_all_references_as_samples.lower() in ('true', '1', 'yes', 'on'):
            for ref_dir in (config.reference_dir, config.user_reference_dir):
                if ref_dir.exists():
                    for f in ref_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.name != reference_filename:
                            img = cv2.imread(str(f))
                            if img is not None:
                                sample_images.append(img)

        if config.upload_dir.exists():
            for f in list(config.upload_dir.iterdir())[:10]:
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    img = cv2.imread(str(f))
                    if img is not None:
                        sample_images.append(img)

        if len(sample_images) < 2:
            h, w = reference_image.shape[:2]
            sample_images.extend([
                reference_image[:h//2, :w//2],
                reference_image[h//2:, w//2:],
                reference_image[:h//2, w//2:]
            ])

        from src.transfers.nilut_transfer import NILUTTransfer
        nilut = NILUTTransfer()
        ref_name = reference_path.stem
        model_path = nilut_service.get_model_path(ref_name)

        nilut.pretrain_on_reference(
            reference_image=reference_image, sample_images=sample_images,
            save_path=str(model_path)
        )
        nilut_service.update_training_meta(
            name=ref_name, training_samples=len(sample_images),
            epochs=200, model_path=str(model_path)
        )

        return JSONResponse({
            "success": True, "reference": ref_name, "model_path": str(model_path),
            "training_samples": len(sample_images),
            "message": f"NILUT model trained successfully for {ref_name}"
        })
    except Exception as e:
        logger.exception("Error training NILUT model")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/nilut/model/{reference_name}")
async def delete_nilut_model(reference_name: str):
    get_nilut_service().delete_model(reference_name)
    return JSONResponse({"success": True, "message": f"NILUT model for {reference_name} deleted"})


@router.get("/api/nilut/universal/status")
async def get_universal_nilut_status():
    from src.transfers.nilut_transfer import NILUTTransfer
    nilut_service = get_nilut_service()
    universal_path = NILUTTransfer.UNIVERSAL_MODEL_PATH
    meta = nilut_service.get_meta()
    universal_meta = meta.get("__universal__", {})
    return JSONResponse({
        "available": universal_path.exists(), "path": str(universal_path),
        "last_trained": universal_meta.get("last_trained"),
        "training_references": universal_meta.get("training_references", 0)
    })


@router.get("/api/nilut/universal/versions")
async def get_universal_model_versions():
    config = get_web_config()
    nilut_service = get_nilut_service()
    models = nilut_service.list_universal_versions()
    for model in models:
        try:
            model["path"] = str(Path(model["path"]).relative_to(config.base_dir))
        except ValueError:
            pass
    return JSONResponse({"models": models, "count": len(models)})


@router.post("/api/nilut/universal/train")
async def train_universal_nilut_model():
    config = get_web_config()
    nilut_service = get_nilut_service()

    try:
        reference_images, sample_images = [], []
        for ref_dir in (config.reference_dir, config.user_reference_dir):
            if ref_dir.exists():
                for f in ref_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        img = cv2.imread(str(f))
                        if img is not None:
                            reference_images.append(img)
                            sample_images.append(img)

        if config.upload_dir.exists():
            for f in list(config.upload_dir.iterdir())[:10]:
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    img = cv2.imread(str(f))
                    if img is not None:
                        sample_images.append(img)

        if len(reference_images) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 reference images to train universal model")

        from src.transfers.nilut_transfer import NILUTTransfer
        nilut = NILUTTransfer()
        nilut.train_universal_model(reference_images=reference_images, sample_images=sample_images, epochs=500)

        nilut_service.update_training_meta(
            name="__universal__", training_samples=len(sample_images),
            epochs=500, training_references=len(reference_images)
        )

        return JSONResponse({
            "success": True,
            "message": f"Universal NILUT model trained on {len(reference_images)} references",
            "training_references": len(reference_images),
            "training_samples": len(sample_images)
        })
    except Exception as e:
        logger.exception("Error training universal NILUT model")
        raise HTTPException(status_code=500, detail=str(e))
