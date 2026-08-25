"""Data intake scanner — the core tool used by Data Intake Agent.

Scans a raw image directory, checks format/corruption, hashes files to find
duplicates, and produces a dataset_manifest. No Azure/cloud storage client is
used here — raw_data_path is a local directory (consistent with how
DatasetValidator operates on a local dataset_path).

Corruption detection uses Pillow when available (soft dependency — install
with `pip install -e ".[vision]"`); without it, files are only sanity-checked
for a non-zero size, since real corruption detection needs image decoding.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from agentic_mlops.contracts.data_intake import (
    DataIntakeInput,
    DataIntakeOutput,
    DataIntakeStatus,
    ImageFileInfo,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class DataIntakeScanner:
    """Scans a raw data directory and produces a dataset manifest."""

    def scan(self, inp: DataIntakeInput) -> DataIntakeOutput:
        raw_dir = Path(inp.raw_data_path).resolve()
        logger.info("Starting data intake scan", extra={"raw_data_path": str(raw_dir)})

        if not raw_dir.is_dir():
            return _failed(f"raw_data_path not found or not a directory: {raw_dir}", inp)

        all_files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
        if not all_files:
            return _failed(f"No files found under {raw_dir}", inp)

        expected_exts = {f".{fmt.lower().lstrip('.')}" for fmt in inp.expected_formats}
        pillow_available = _pillow_available()

        image_records: list[ImageFileInfo] = []
        corrupted: list[str] = []
        unexpected_format: list[str] = []
        hash_to_files: dict[str, list[str]] = defaultdict(list)

        for f in all_files:
            rel = str(f.relative_to(raw_dir))
            if f.suffix.lower() not in expected_exts:
                unexpected_format.append(rel)
                continue

            info = _inspect_image(f, rel)
            if info is None:
                corrupted.append(rel)
                continue

            hash_to_files[info.sha256].append(rel)
            image_records.append(info)

        duplicate_groups = [group for group in hash_to_files.values() if len(group) > 1]
        num_duplicate_extra_copies = sum(len(g) - 1 for g in duplicate_groups)

        num_files = len(all_files)
        corrupted_ratio = len(corrupted) / num_files if num_files else 0.0
        duplicate_ratio = num_duplicate_extra_copies / num_files if num_files else 0.0

        blocking: list[str] = []
        if num_files < inp.min_files:
            blocking.append(
                f"Only {num_files} file(s) found under {raw_dir}, expected at least "
                f"{inp.min_files}."
            )
        if corrupted_ratio > inp.corrupted_ratio_threshold:
            blocking.append(
                f"Corrupted image ratio {corrupted_ratio:.1%} exceeds threshold "
                f"{inp.corrupted_ratio_threshold:.1%} ({len(corrupted)}/{num_files})."
            )

        if blocking:
            logger.warning("Data intake failed", extra={"reasons": blocking})
            return DataIntakeOutput(
                success=False,
                message="; ".join(blocking),
                status=DataIntakeStatus.FAILED,
                errors=blocking,
                dataset_name=inp.dataset_name,
                source=inp.source,
                num_files=num_files,
                valid_images=len(image_records),
                corrupted_images=corrupted,
                duplicate_groups=duplicate_groups,
                unexpected_format_files=unexpected_format,
                pillow_available=pillow_available,
            )

        review_reasons: list[str] = []
        if not inp.source:
            review_reasons.append(
                "Source is not specified — a human must confirm data provenance/license "
                "before this dataset proceeds."
            )
        if duplicate_ratio > inp.duplicate_ratio_threshold:
            review_reasons.append(
                f"Duplicate ratio {duplicate_ratio:.1%} exceeds threshold "
                f"{inp.duplicate_ratio_threshold:.1%} ({num_duplicate_extra_copies}/{num_files})."
            )
        if unexpected_format:
            review_reasons.append(
                f"{len(unexpected_format)} file(s) have an unexpected format "
                f"(expected: {sorted(expected_exts)})."
            )

        status = (
            DataIntakeStatus.NEEDS_HUMAN_SOURCE_APPROVAL
            if review_reasons
            else DataIntakeStatus.PASSED
        )

        logger.info(
            "Data intake scan complete",
            extra={
                "status": status,
                "num_files": num_files,
                "valid_images": len(image_records),
                "corrupted": len(corrupted),
                "duplicate_groups": len(duplicate_groups),
            },
        )

        return DataIntakeOutput(
            success=True,
            message=(
                f"Data intake {status}: {len(image_records)}/{num_files} valid image(s), "
                f"{len(corrupted)} corrupted, {len(duplicate_groups)} duplicate group(s)."
            ),
            status=status,
            dataset_name=inp.dataset_name,
            source=inp.source,
            num_files=num_files,
            valid_images=len(image_records),
            corrupted_images=corrupted,
            duplicate_groups=duplicate_groups,
            unexpected_format_files=unexpected_format,
            image_records=image_records,
            pillow_available=pillow_available,
            warnings=review_reasons,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _sha256(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_image(path: Path, rel_path: str) -> ImageFileInfo | None:
    """Return ImageFileInfo, or None if the file is empty/unreadable/corrupted."""
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return None
    if size_bytes == 0:
        return None

    try:
        sha256 = _sha256(path)
    except OSError:
        return None

    width: int | None = None
    height: int | None = None
    fmt = path.suffix.lower().lstrip(".")

    try:
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:  # verify() invalidates the handle — reopen
                width, height = img.size
                fmt = (img.format or fmt).lower()
        except (UnidentifiedImageError, OSError, ValueError):
            return None
    except ImportError:
        pass  # Pillow not installed — accept based on size/extension alone

    return ImageFileInfo(
        filename=path.name,
        path=rel_path,
        size_bytes=size_bytes,
        format=fmt,
        width=width,
        height=height,
        sha256=sha256,
    )


def _failed(message: str, inp: DataIntakeInput) -> DataIntakeOutput:
    return DataIntakeOutput(
        success=False,
        message=message,
        status=DataIntakeStatus.FAILED,
        errors=[message],
        dataset_name=inp.dataset_name,
        source=inp.source,
    )
