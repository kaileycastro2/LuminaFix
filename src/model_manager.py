"""
Model Manager - Handles downloading, caching, and verification of neural network weights.

Provides thread-safe model downloads with progress reporting.
"""

from pathlib import Path
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass
import hashlib
import urllib.request
import threading
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str
    url: str
    filename: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class DownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelManager:
    """
    Singleton manager for neural network model weights.

    Features:
    - Lazy downloading on first use
    - SHA256 verification
    - Thread-safe downloads
    - Progress callbacks
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    # Registry of known models
    MODELS: Dict[str, ModelInfo] = {
        "msgnet": ModelInfo(
            name="MSG-Net 21 Styles",
            url="https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer/raw/master/experiments/models/21styles.model",
            filename="msgnet_21styles.model",
            size_bytes=None  # Will be detected during download
        ),
        "photowct2_encoder": ModelInfo(
            name="PhotoWCT2 Encoder",
            url="https://github.com/chiutaiyin/PhotoWCT2/raw/main/ckpts/ckpts-conv/encoder.pkl",
            filename="photowct2_encoder.pkl",
            size_bytes=None
        ),
        "photowct2_decoder": ModelInfo(
            name="PhotoWCT2 Decoder",
            url="https://github.com/chiutaiyin/PhotoWCT2/raw/main/ckpts/ckpts-conv/decoder.pkl",
            filename="photowct2_decoder.pkl",
            size_bytes=None
        ),
    }

    def __new__(cls, models_dir: Optional[Path] = None) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, models_dir: Optional[Path] = None):
        if ModelManager._initialized and models_dir is None:
            return

        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"

        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._download_locks: Dict[str, threading.Lock] = {}
        self._download_status: Dict[str, str] = {}

        ModelManager._initialized = True

    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        return self._models_dir

    def get_model_path(self, model_id: str) -> Path:
        """
        Get path to a model file, downloading if necessary.

        Args:
            model_id: Model identifier (e.g., "msgnet", "photowct2_encoder")

        Returns:
            Path to the model file

        Raises:
            DownloadError: If download fails
            ValueError: If model_id is unknown
        """
        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        model_info = self.MODELS[model_id]
        model_path = self._models_dir / model_info.filename

        if model_path.exists():
            return model_path

        # Download the model
        self._download_model(model_id, model_info)
        return model_path

    def ensure_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """
        Ensure a model is downloaded and return its path.

        Args:
            model_id: Model identifier
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Path to the model file
        """
        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = self.MODELS[model_id]
        model_path = self._models_dir / model_info.filename

        if model_path.exists():
            logger.info(f"Model {model_id} already downloaded: {model_path}")
            return model_path

        return self._download_model(model_id, model_info, progress_callback)

    def _download_model(
        self,
        model_id: str,
        model_info: ModelInfo,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """Download a model with thread safety."""
        # Get or create lock for this model
        if model_id not in self._download_locks:
            self._download_locks[model_id] = threading.Lock()

        lock = self._download_locks[model_id]

        with lock:
            # Check again in case another thread downloaded it
            model_path = self._models_dir / model_info.filename
            if model_path.exists():
                return model_path

            self._download_status[model_id] = "downloading"
            logger.info(f"Downloading {model_info.name} from {model_info.url}")

            try:
                self._download_file(
                    model_info.url,
                    model_path,
                    progress_callback
                )

                # Verify hash if provided
                if model_info.sha256:
                    if not self._verify_hash(model_path, model_info.sha256):
                        model_path.unlink()
                        raise DownloadError(f"Hash verification failed for {model_id}")

                self._download_status[model_id] = "completed"
                logger.info(f"Successfully downloaded {model_info.name}")
                return model_path

            except Exception as e:
                self._download_status[model_id] = f"failed: {str(e)}"
                if model_path.exists():
                    model_path.unlink()
                raise DownloadError(f"Failed to download {model_id}: {e}") from e

    def _download_file(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download a file with progress reporting."""
        temp_path = destination.with_suffix(".tmp")

        try:
            # Open URL and get content length
            with urllib.request.urlopen(url, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)

            # Move temp file to final destination
            temp_path.rename(destination)

        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_hash

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        if model_id not in self.MODELS:
            return False
        model_path = self._models_dir / self.MODELS[model_id].filename
        return model_path.exists()

    def get_download_status(self, model_id: str) -> str:
        """Get download status for a model."""
        if model_id not in self.MODELS:
            return "unknown"
        if self.is_model_available(model_id):
            return "available"
        return self._download_status.get(model_id, "not_downloaded")

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all models and their status."""
        result = {}
        for model_id, model_info in self.MODELS.items():
            result[model_id] = {
                "name": model_info.name,
                "filename": model_info.filename,
                "available": self.is_model_available(model_id),
                "status": self.get_download_status(model_id),
                "path": str(self._models_dir / model_info.filename)
            }
        return result

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        if model_id not in self.MODELS:
            return False
        model_path = self._models_dir / self.MODELS[model_id].filename
        if model_path.exists():
            model_path.unlink()
            return True
        return False


# Module-level singleton accessor
_model_manager: Optional[ModelManager] = None


def get_model_manager(models_dir: Optional[Path] = None) -> ModelManager:
    """Get the global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(models_dir)
    return _model_manager
