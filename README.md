# LuminaFix Style Transfer - Milestone 1

Extracts color/tone characteristics from a reference image and applies them to batch images.

## Features

- **Style Extraction**: LAB color statistics, histogram, saturation profile
- **Color Transfer**: Reinhard method with histogram matching
- **Skin Protection**: YCbCr-based skin detection to preserve skin tones
- **Neon Protection**: Saturation clamping to prevent oversaturation
- **Batch Processing**: Process 20-50+ images with consistent results
- **Deterministic**: Same inputs always produce same outputs

## Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your reference image in `input/` folder (name it `reference.jpg`)
2. Place target images in `input/` folder
3. Run the script:

```bash
python main.py
```

4. Find processed images in `output/` folder

## Web Application

A browser-based interface for interactive style transfer.

### Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web server
cd web && uvicorn app:app --host 0.0.0.0 --port 8000
# OR
python web/app.py
```

3. Open http://localhost:8000 in your browser

### Web App Features
- Upload target images via drag-and-drop
- Select from available reference images
- Adjust color/luminance strength with sliders
- Toggle skin and neon protection
- Download processed results

### Requirements
- Reference images in `test_images/reference/` folder
- `uploads/` directory (created automatically)

## Configuration

Edit `main.py` to adjust settings:

```python
# Paths
INPUT_DIR = "input"
REFERENCE_IMAGE = "input/reference.jpg"
OUTPUT_DIR = "output"

# Processing options
COLOR_STRENGTH = 0.8          # 0-1, higher = more color transfer
LUMINANCE_STRENGTH = 0.5      # 0-1, higher = more brightness matching
ENABLE_SKIN_PROTECTION = True # Protect skin tones
ENABLE_NEON_PROTECTION = True # Prevent oversaturation
JPEG_QUALITY = 95             # Output quality
```

## Project Structure

```
style_transfer/
├── src/
│   ├── __init__.py
│   ├── style_extractor.py   # Extract style from reference
│   ├── color_transfer.py    # Apply style to targets
│   ├── skin_protection.py   # Skin tone detection
│   ├── neon_protection.py   # Saturation protection
│   └── batch_processor.py   # Batch processing logic
├── web/
│   ├── app.py               # FastAPI web server
│   ├── static/              # CSS/JS assets
│   └── templates/           # HTML templates
├── main.py                  # CLI entry point
├── requirements.txt
├── input/                   # Place images here
│   ├── reference.jpg
│   └── (target images)
├── test_images/reference/   # Reference images for web app
└── output/                  # Results saved here
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Algorithm

1. **Style Extraction**: Extract LAB mean/std, histogram, saturation from reference
2. **Skin Detection**: Identify skin regions using YCbCr color thresholds
3. **Color Transfer**: Apply Reinhard color transfer with reduced strength on skin
4. **Neon Protection**: Clamp saturation and blend back original in neon regions
5. **Gamut Mapping**: Ensure output colors are within valid range
