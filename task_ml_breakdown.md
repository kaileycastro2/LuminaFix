# LuminaFix - Task & ML Breakdown

## Milestone 1 - Functional Baseline ($400)

| Task | ML Needed? | Approach |
|------|------------|----------|
| Style extraction (tone curve) | No | Histogram analysis, percentile mapping |
| Style extraction (contrast) | No | Standard deviation calculation |
| Style extraction (HSL/color grading) | No | LAB color space mean/std transfer |
| Single image processing | No | Apply extracted parameters |
| Batch processing (20-50 images) | No | Loop through images |
| Deterministic results | No | Fixed seed, no randomness |
| Portrait-safe (skin tones) | No (M1) | HSV/YCbCr color thresholds (~60-70% accuracy, sufficient for M1) |
| Neon-safe (saturated colors) | No | Saturation clamping, gamut mapping |

---

## Milestone 2 - Aesthetic Improvement ($500)

| Task | ML Needed? | Approach |
|------|------------|----------|
| Improved tone curve | No | Better curve fitting algorithms |
| Better contrast behavior | No | Adaptive contrast, CLAHE |
| Color separation/depth | No | LAB channel manipulation |
| Skin tone preservation | Recommended (M2) | Upgrade to MediaPipe/BiSeNet if M1 quality insufficient (~95% accuracy) |
| Highlight/shadow detail | No | Tone mapping, shadow/highlight recovery |
| Lighting-aware adaptation | No | Histogram heuristics (brightness, contrast stats) |

---

## Milestone 3 - Productization ($450)

| Task | ML Needed? | Approach |
|------|------------|----------|
| Parameter abstraction (Temp, Tint, Contrast) | No | Map internal values to Lightroom scale |
| XMP preset generation | No | XML template with extracted values |
| ZIP export | No | Python zipfile module |
| Preview alignment with XMP | No | Ensure parameter mapping is accurate |

---

## Milestone 4 - Polish & Handoff ($400)

| Task | ML Needed? | Approach |
|------|------------|----------|
| UI/UX (upload, progress, preview, download) | No | React/Vue + FastAPI |
| Performance improvements | No | Image resizing, caching, async processing |
| Bug fixes & QA | No | Testing |
| Documentation | No | Markdown/comments |

---

## Summary

| Category | Count |
|----------|-------|
| Total Tasks | ~20 |
| No ML Required | 18 |
| ML Optional/Recommended | 2 |

### Where ML Actually Helps:
1. **Skin detection/segmentation** - For protecting skin tones (Milestone 1-2)
2. **Face detection** - To identify portrait areas (Milestone 1-2)

### Recommended Pre-trained Models (No Training Required):
- **MediaPipe Face Mesh** - Face detection & landmarks
- **BiSeNet** - Fast semantic segmentation
- **OpenCV DNN Face Detector** - Lightweight face detection

### NOT Needed:
- Neural style transfer
- Custom model training
- GANs or diffusion models
- Large language models
