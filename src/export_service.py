"""
Export Service Module

Bundles processed JPGs and XMP presets into downloadable ZIP archives.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .xmp_generator import LightroomParams, XMPPresetGenerator


@dataclass
class ExportItem:
    """A single item to include in the export ZIP."""
    image_path: Path
    method_name: str
    reference_name: str
    strength: int
    params: LightroomParams
    target_name: str = ""


class ExportService:
    """Creates ZIP exports bundling processed images and XMP presets."""

    def __init__(self):
        self._xmp_generator = XMPPresetGenerator()

    STRATEGY_LABELS = {
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

    def create_export_zip(self, items: List[ExportItem], xmp_strategy: str = "color_science") -> io.BytesIO:
        """
        Create a ZIP archive containing images and XMP presets.

        Args:
            items: List of ExportItem to bundle

        Returns:
            BytesIO containing the ZIP file data

        ZIP structure:
            LuminaFix_Export.zip
            ├── images/
            │   ├── reinhard_ref1_target1_70pct.jpg
            │   └── nilut_ref1_target1_70pct.jpg
            └── presets/
                ├── reinhard_ref1_70pct.xmp
                └── nilut_ref1_70pct.xmp
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            seen_presets = set()

            for item in items:
                # Add image
                image_path = Path(item.image_path)
                if image_path.exists():
                    safe_method = self._safe_filename(item.method_name)
                    safe_ref = self._safe_filename(item.reference_name)
                    target_stem = image_path.stem
                    image_name = f"{safe_method}_{safe_ref}_{target_stem}_{item.strength}pct{image_path.suffix}"
                    zf.write(str(image_path), f"images/{image_name}")

                # Add XMP preset (one per target+method+reference+strength combo)
                safe_target = self._safe_filename(item.target_name) if item.target_name else target_stem
                strategy_label = self.STRATEGY_LABELS.get(xmp_strategy, xmp_strategy)
                preset_key = f"{safe_target}_{item.method_name}_{item.reference_name}_{item.strength}"
                if preset_key not in seen_presets:
                    seen_presets.add(preset_key)
                    preset_name = f"{safe_ref}_{strategy_label}"
                    xmp_content = self._xmp_generator.generate_xmp(
                        item.params, preset_name
                    )
                    xmp_filename = f"{safe_ref}_{strategy_label}.xmp"
                    zf.writestr(f"presets/{xmp_filename}", xmp_content)

        buffer.seek(0)
        return buffer

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitize a string for use in filenames."""
        return name.replace(' ', '_').replace('/', '_').replace('\\', '_')
