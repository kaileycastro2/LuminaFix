"""
NILUT Model Service - Manages NILUT model metadata, paths, and versioning.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config

logger = logging.getLogger(__name__)


class NILUTModelService:
    """
    Service for managing NILUT model files, metadata, and versions.

    Responsibilities:
    - Model path resolution (per-reference and universal)
    - Metadata persistence (meta.json)
    - Model version discovery and listing
    - Model lifecycle (create metadata, delete)
    """

    def __init__(self, models_dir: Optional[Path] = None, meta_file: Optional[Path] = None):
        config = get_config()
        self._models_dir = models_dir or config.web.nilut_models_dir
        self._meta_file = meta_file or config.web.nilut_meta_file

    def get_meta(self) -> Dict[str, Any]:
        """Load NILUT model metadata from disk."""
        if self._meta_file.exists():
            with open(self._meta_file, 'r') as f:
                return json.load(f)
        return {}

    def save_meta(self, meta: Dict[str, Any]) -> None:
        """Persist NILUT model metadata to disk."""
        with open(self._meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

    def get_model_path(self, reference_name: str) -> Path:
        """Get path for a per-reference NILUT model file."""
        safe_name = reference_name.replace(" ", "_").replace("/", "_")
        return self._models_dir / f"{safe_name}.pt"

    def get_universal_model_path(self, model_id: str = "latest") -> Path:
        """Get path for a universal model version."""
        if model_id == "latest":
            return self._models_dir / "latest" / "universal.pt"
        return self._models_dir / "universal" / model_id / "universal.pt"

    def format_model_display_name(self, model_id: str) -> str:
        """Format a model_id into a human-readable display name."""
        if model_id == "latest":
            return "Latest"
        try:
            dt = datetime.strptime(model_id, "%Y%m%d_%H%M%S")
            return dt.strftime("%b %d, %I:%M %p")
        except ValueError:
            return model_id

    def list_universal_versions(self) -> List[Dict[str, Any]]:
        """List all available universal model versions."""
        models = []

        # Latest model
        latest_path = self._models_dir / "latest" / "universal.pt"
        if latest_path.exists():
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_path))
            models.append({
                "id": "latest",
                "name": "Latest Model",
                "path": str(latest_path),
                "timestamp": mod_time.strftime("%b %d, %I:%M %p"),
                "timestamp_raw": mod_time.isoformat(),
                "is_latest": True
            })

        # Timestamped backups
        backup_dir = self._models_dir / "universal"
        if backup_dir.exists():
            for ts_dir in sorted(backup_dir.iterdir(), reverse=True):
                if ts_dir.is_dir():
                    model_path = ts_dir / "universal.pt"
                    if model_path.exists():
                        try:
                            dt = datetime.strptime(ts_dir.name, "%Y%m%d_%H%M%S")
                            models.append({
                                "id": ts_dir.name,
                                "name": dt.strftime("%b %d, %I:%M %p"),
                                "path": str(model_path),
                                "timestamp": dt.strftime("%b %d, %I:%M %p"),
                                "timestamp_raw": dt.isoformat(),
                                "is_latest": False
                            })
                        except ValueError:
                            continue

        return models

    def delete_model(self, reference_name: str) -> None:
        """Delete a per-reference model and its metadata."""
        model_path = self.get_model_path(reference_name)
        if model_path.exists():
            model_path.unlink()
        meta = self.get_meta()
        if reference_name in meta:
            del meta[reference_name]
            self.save_meta(meta)

    def update_training_meta(
        self,
        name: str,
        training_samples: int,
        epochs: int,
        model_path: Optional[str] = None,
        training_references: Optional[int] = None,
    ) -> None:
        """Update metadata after a successful training run."""
        meta = self.get_meta()
        entry: Dict[str, Any] = {
            "last_trained": datetime.now().isoformat(),
            "training_samples": training_samples,
            "epochs": epochs,
        }
        if model_path:
            entry["model_path"] = model_path
        if training_references is not None:
            entry["training_references"] = training_references
        meta[name] = entry
        self.save_meta(meta)
