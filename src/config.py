"""
Configuration module for style transfer application.

Provides centralized configuration management with environment variable support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""

    color_strength: float = 0.7
    luminance_strength: float = 0.0
    enable_skin_protection: bool = True
    enable_neon_protection: bool = True
    jpeg_quality: int = 95

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.color_strength <= 2:
            raise ValueError("color_strength must be between 0 and 2")
        if not 0 <= self.luminance_strength <= 1:
            raise ValueError("luminance_strength must be between 0 and 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass
class ModelConfig:
    """Configuration for neural network models."""

    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    download_timeout: int = 300  # seconds

    # Model URLs (official sources)
    msgnet_url: str = "https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer/raw/master/experiments/models/21styles.model"
    photowct2_encoder_url: str = "https://github.com/chiutaiyin/PhotoWCT2/raw/main/ckpts/ckpts-conv/encoder.pkl"
    photowct2_decoder_url: str = "https://github.com/chiutaiyin/PhotoWCT2/raw/main/ckpts/ckpts-conv/decoder.pkl"

    def __post_init__(self):
        """Ensure models directory exists."""
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class WebConfig:
    """Configuration for web application paths."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def upload_dir(self) -> Path:
        return self.base_dir / "uploads"

    @property
    def reference_dir(self) -> Path:
        return self.base_dir / "test_images" / "reference"

    @property
    def user_reference_dir(self) -> Path:
        return self.base_dir / "uploads" / "references"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "uploads" / "processed"

    @property
    def nilut_models_dir(self) -> Path:
        return self.base_dir / "models" / "nilut"

    @property
    def nilut_meta_file(self) -> Path:
        return self.nilut_models_dir / "meta.json"

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.user_reference_dir.mkdir(exist_ok=True)
        self.nilut_models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration."""

    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    web: WebConfig = field(default_factory=WebConfig)

    # Directories
    upload_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "uploads")
    reference_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "test_images" / "reference")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "uploads" / "processed")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        processing = ProcessingConfig(
            color_strength=float(os.getenv("COLOR_STRENGTH", "0.7")),
            luminance_strength=float(os.getenv("LUMINANCE_STRENGTH", "0.0")),
            enable_skin_protection=os.getenv("ENABLE_SKIN_PROTECTION", "true").lower() == "true",
            enable_neon_protection=os.getenv("ENABLE_NEON_PROTECTION", "true").lower() == "true",
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "95"))
        )

        models = ModelConfig(
            models_dir=Path(os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "models"))),
            device=os.getenv("DEVICE", "auto"),
            download_timeout=int(os.getenv("DOWNLOAD_TIMEOUT", "300"))
        )

        return cls(
            processing=processing,
            models=models,
            upload_dir=Path(os.getenv("UPLOAD_DIR", str(Path(__file__).parent.parent / "uploads"))),
            reference_dir=Path(os.getenv("REFERENCE_DIR", str(Path(__file__).parent.parent / "test_images" / "reference"))),
            output_dir=Path(os.getenv("OUTPUT_DIR", str(Path(__file__).parent.parent / "uploads" / "processed")))
        )


# Global config instance (can be overridden)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
