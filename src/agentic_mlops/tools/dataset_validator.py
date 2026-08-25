"""YOLO dataset validator — the core tool used by Dataset Validation Agent.

Validates a YOLO-format dataset directory and returns a structured report.
No Azure or MLflow calls are made here.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import yaml

from agentic_mlops.contracts.datasets import (
    DatasetValidationInput,
    DatasetValidationOutput,
    IssueSeverity,
    LabelIssue,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

# Supported image extensions (lower-case)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Required YOLO directory structure inside a dataset root
_REQUIRED_SPLITS = ["train", "val"]
_OPTIONAL_SPLITS = ["test"]
_ALL_SPLITS = _REQUIRED_SPLITS + _OPTIONAL_SPLITS


class DatasetValidator:
    """Validates a YOLO object-detection dataset.

    Usage::
        validator = DatasetValidator()
        result = validator.validate(DatasetValidationInput(dataset_path="/data/my_dataset"))
    """

    def validate(self, inp: DatasetValidationInput) -> DatasetValidationOutput:
        dataset_path = Path(inp.dataset_path).resolve()
        logger.info("Starting dataset validation", extra={"dataset_path": str(dataset_path)})

        blocking_issues: list[str] = []
        warnings: list[str] = []
        label_issues: list[LabelIssue] = []
        class_distribution: dict[str, int] = defaultdict(int)
        num_images = 0
        num_labels = 0

        # ── 1. data.yaml ──────────────────────────────────────────────────────
        yaml_path = Path(inp.data_yaml_path) if inp.data_yaml_path else dataset_path / "data.yaml"
        class_names: dict[int, str] = {}

        if not yaml_path.exists():
            blocking_issues.append(f"data.yaml not found at {yaml_path}")
        else:
            try:
                data_yaml = _load_yaml(yaml_path)
                class_names = _parse_class_names(data_yaml)
                if not class_names:
                    blocking_issues.append("data.yaml: 'names' section is empty or missing")
            except Exception as exc:
                blocking_issues.append(f"data.yaml parse error: {exc}")

        num_classes = len(class_names)

        # ── 2. Required split directories ─────────────────────────────────────
        for split in _REQUIRED_SPLITS:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split
            if not img_dir.is_dir():
                blocking_issues.append(f"Missing directory: images/{split}")
            if not lbl_dir.is_dir():
                blocking_issues.append(f"Missing directory: labels/{split}")

        # Stop here — structural issues make further checks meaningless
        if blocking_issues:
            return _make_output(
                success=False,
                status="failed",
                blocking_issues=blocking_issues,
                warnings=warnings,
                label_issues=label_issues,
                class_distribution=dict(class_distribution),
                num_images=num_images,
                num_labels=num_labels,
            )

        # ── 3. Per-split validation ────────────────────────────────────────────
        # Collect image checksums per split to detect cross-split duplicates.
        split_image_hashes: dict[str, set[str]] = {}

        for split in _ALL_SPLITS:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split

            if not img_dir.is_dir():
                continue  # optional splits may be absent

            image_files = _list_images(img_dir)
            if not image_files:
                warnings.append(f"No images found in images/{split}")
                continue

            num_images += len(image_files)
            split_hashes: set[str] = set()

            for img_path in image_files:
                # Fast hash on first 64 KB — sufficient to flag exact duplicates
                img_hash = _file_hash_prefix(img_path)
                split_hashes.add(img_hash)

                # Check matching label file
                label_path = lbl_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    warnings.append(f"Missing label for image: {img_path.name} in {split}")
                    continue

                issues, dist = _validate_label_file(
                    label_path=label_path,
                    num_classes=num_classes,
                    class_names=class_names,
                    split=split,
                )
                label_issues.extend(issues)
                num_labels += 1
                for cls_name, count in dist.items():
                    class_distribution[cls_name] += count

            split_image_hashes[split] = split_hashes

        # ── 4. Cross-split duplicate detection ────────────────────────────────
        _detect_cross_split_duplicates(split_image_hashes, blocking_issues)

        # ── 5. Class distribution warnings ────────────────────────────────────
        if class_distribution:
            total = sum(class_distribution.values())
            for cls_name, count in class_distribution.items():
                ratio = count / total if total > 0 else 0
                if ratio < 0.15:
                    warnings.append(
                        f"Class '{cls_name}' has only {ratio:.1%} of total annotations "
                        f"({count}/{total}) — consider augmentation or more samples."
                    )

        # ── 6. Blocking label issues ───────────────────────────────────────────
        blocking_label_msgs = [
            issue.message for issue in label_issues if issue.severity == IssueSeverity.BLOCKING
        ]
        blocking_issues.extend(blocking_label_msgs)

        warning_label_msgs = [
            f"{issue.file}: {issue.message}"
            for issue in label_issues
            if issue.severity == IssueSeverity.WARNING
        ]
        warnings.extend(warning_label_msgs)

        # ── 7. Determine final status ──────────────────────────────────────────
        has_blocking = bool(blocking_issues)
        has_warnings = bool(warnings)

        if has_blocking:
            status = "failed"
            success = False
        elif has_warnings and inp.fail_on_warnings:
            status = "failed"
            success = False
        elif has_warnings:
            status = "warning"
            success = True
        else:
            status = "passed"
            success = True

        logger.info(
            "Dataset validation complete",
            extra={
                "status": status,
                "num_images": num_images,
                "blocking_issues": len(blocking_issues),
                "warnings": len(warnings),
            },
        )

        return _make_output(
            success=success,
            status=status,
            blocking_issues=blocking_issues,
            warnings=warnings,
            label_issues=label_issues,
            class_distribution=dict(class_distribution),
            num_images=num_images,
            num_labels=num_labels,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_class_names(data_yaml: dict) -> dict[int, str]:
    """Return {class_id: name} from data.yaml 'names' section."""
    names = data_yaml.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def _list_images(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in _IMAGE_EXTS]


def _file_hash_prefix(path: Path, chunk_size: int = 65536) -> str:
    """Return a hex digest of the first chunk_size bytes of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(chunk_size))
    return h.hexdigest()


def _validate_label_file(
    label_path: Path,
    num_classes: int,
    class_names: dict[int, str],
    split: str,
) -> tuple[list[LabelIssue], dict[str, int]]:
    """Validate a single YOLO label .txt file.

    Returns:
        (list of issues, class distribution dict for this file)
    """
    issues: list[LabelIssue] = []
    dist: dict[str, int] = defaultdict(int)

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        issues.append(
            LabelIssue(
                severity=IssueSeverity.BLOCKING,
                file=str(label_path),
                message=f"Cannot read label file: {exc}",
            )
        )
        return issues, dict(dist)

    if not lines:
        return issues, dict(dist)  # empty label = background image, allowed

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.BLOCKING,
                    file=label_path.name,
                    line=line_num,
                    message=(
                        f"[{split}/{label_path.name}:{line_num}] "
                        f"Expected 5 columns, got {len(parts)}: '{line}'"
                    ),
                )
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.BLOCKING,
                    file=label_path.name,
                    line=line_num,
                    message=(
                        f"[{split}/{label_path.name}:{line_num}] " f"Non-numeric values: '{line}'"
                    ),
                )
            )
            continue

        # class_id range check
        if num_classes > 0 and (class_id < 0 or class_id >= num_classes):
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.BLOCKING,
                    file=label_path.name,
                    line=line_num,
                    message=(
                        f"[{split}/{label_path.name}:{line_num}] "
                        f"Unknown class_id {class_id} "
                        f"(valid range: 0..{num_classes - 1})"
                    ),
                )
            )
            continue

        # bbox coordinate range check
        bad_coords = []
        for name, val in [
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", width),
            ("height", height),
        ]:
            if not (0.0 <= val <= 1.0):
                bad_coords.append(f"{name}={val:.4f}")

        if bad_coords:
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.BLOCKING,
                    file=label_path.name,
                    line=line_num,
                    message=(
                        f"[{split}/{label_path.name}:{line_num}] "
                        f"Bbox coordinates out of [0,1]: {', '.join(bad_coords)}"
                    ),
                )
            )
            continue

        # bbox degenerate size check
        if width <= 0 or height <= 0:
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.BLOCKING,
                    file=label_path.name,
                    line=line_num,
                    message=(
                        f"[{split}/{label_path.name}:{line_num}] "
                        f"Degenerate bbox: width={width}, height={height}"
                    ),
                )
            )
            continue

        # Very small objects — warning only
        if width < 0.01 or height < 0.01:
            issues.append(
                LabelIssue(
                    severity=IssueSeverity.WARNING,
                    file=label_path.name,
                    line=line_num,
                    message=(f"Very small object: width={width:.4f}, height={height:.4f}"),
                )
            )

        # Count class distribution
        cls_name = class_names.get(class_id, f"class_{class_id}")
        dist[cls_name] += 1

    return issues, dict(dist)


def _detect_cross_split_duplicates(
    split_hashes: dict[str, set[str]],
    blocking_issues: list[str],
) -> None:
    """Add blocking issues for any image that appears in more than one split."""
    splits = list(split_hashes.keys())
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            overlap = split_hashes[split_a] & split_hashes[split_b]
            if overlap:
                blocking_issues.append(
                    f"{len(overlap)} duplicate image(s) found between "
                    f"'{split_a}' and '{split_b}' splits."
                )


def _make_output(
    *,
    success: bool,
    status: str,
    blocking_issues: list[str],
    warnings: list[str],
    label_issues: list[LabelIssue],
    class_distribution: dict[str, int],
    num_images: int,
    num_labels: int,
) -> DatasetValidationOutput:
    return DatasetValidationOutput(
        success=success,
        status=status,
        message=f"Validation {status}: {len(blocking_issues)} blocking issue(s), "
        f"{len(warnings)} warning(s).",
        blocking_issues=blocking_issues,
        warnings=warnings,
        label_issues=label_issues,
        class_distribution=class_distribution,
        num_images=num_images,
        num_labels=num_labels,
        errors=blocking_issues,
    )
