"""HardSampleIngester — locates hard-sample images and copies them to a staging dir.

Reads the hard_samples_manifest.json written by MonitoringAgent (a JSON array of
HardSample dicts, each with an "image_id" field), then resolves each image_id
against an *images_source_dir* using three fallback strategies:

  1. Treat image_id as an absolute path — if it exists, use it directly.
  2. Treat image_id as a filename or relative path; look for
     <images_source_dir>/<basename>.
  3. Try <images_source_dir>/<stem><ext> for every extension in _IMAGE_EXTS.

Images that cannot be resolved are skipped with a warning.  The staging
directory is created by the caller before this function is invoked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..observability.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _resolve_image(source_dir: Path, image_id: str) -> Path | None:
    candidate = Path(image_id)
    # 1. Absolute path that exists on disk
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # 2. Basename lookup in source_dir
    by_name = source_dir / candidate.name
    if by_name.exists():
        return by_name
    # 3. Stem + common image extensions
    for ext in _IMAGE_EXTS:
        p = source_dir / (candidate.stem + ext)
        if p.exists():
            return p
    return None


class HardSampleIngester:
    """Reads a hard-samples manifest and copies found images to a staging dir."""

    def stage(
        self,
        manifest_path: Path,
        source_dir: Path,
        staging_dir: Path,
    ) -> tuple[int, int, int, list[str]]:
        """Copy hard-sample images from *source_dir* into *staging_dir*.

        Returns:
            (num_in_manifest, num_found, num_not_found, warnings)
        """
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        hard_samples: list[dict] = raw if isinstance(raw, list) else raw.get("hard_samples", [])

        staging_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        found = 0
        not_found = 0

        for sample in hard_samples:
            image_id = sample.get("image_id", "")
            if not image_id:
                warnings.append("Hard sample entry missing 'image_id' — skipped.")
                not_found += 1
                continue
            img_path = _resolve_image(source_dir, image_id)
            if img_path is None:
                warnings.append(f"Hard sample '{image_id}' not found under {source_dir} — skipped.")
                not_found += 1
                continue
            dest = staging_dir / img_path.name
            if dest.exists():
                # Avoid overwriting a file from an earlier hard sample with the same basename
                dest = staging_dir / f"{img_path.stem}_{found}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            found += 1
            logger.debug("Staged hard sample", extra={"src": str(img_path), "dest": str(dest)})

        return len(hard_samples), found, not_found, warnings
