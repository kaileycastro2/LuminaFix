"""
FastAPI Backend for LuminaFix Style Transfer.

Thin application shell — all routes live in web/routes.py.
"""

import sys
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for imports (needed when running directly)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from web.routes import router

# Let each worker use multiple threads for PyTorch inference (NILUT)
try:
    import os
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
except Exception:
    pass

# Import transfer methods to register them via decorators
from src.transfers.reinhard_transfer import ReinhardTransfer  # noqa: F401
try:
    from src.transfers.nilut_transfer import NILUTTransfer  # noqa: F401
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LuminaFix Style Transfer")


def _log_startup_memory():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    logger.info("MEMORY [startup]: %d MB", rss_kb // 1024)
                    return
    except Exception:
        pass


_log_startup_memory()

# Pre-load MediaPipe FaceLandmarker in master process so --preload shares
# the model across forked workers via copy-on-write (saves ~300 MB per worker).
try:
    from src.face_landmarker import get_face_landmarker
    if get_face_landmarker() is not None:
        logger.info("MediaPipe FaceLandmarker pre-loaded at startup")
    else:
        logger.warning("MediaPipe FaceLandmarker pre-load returned None")
except Exception:
    logger.exception("MediaPipe FaceLandmarker pre-load failed")

_log_startup_memory()

# Ensure directories exist
config = get_config()
config.web.ensure_directories()

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Register all routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
