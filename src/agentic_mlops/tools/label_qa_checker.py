"""Label QA checker — the core tool used by Label QA Agent.

Runs deterministic geometric/statistical checks on an already-valid YOLO dataset
(assumes DatasetValidationAgent has already passed it — this tool does not repeat
structural checks like malformed columns or out-of-range coordinates) and,
optionally, compares human labels against a reference YOLO model's predictions.

No labels are ever modified — this tool only reports suspicious samples.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from agentic_mlops.contracts.label_qa import (
    LabelQAInput,
    LabelQAOutput,
    LabelQAStatus,
    QAIssueType,
    SuspiciousSample,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SPLITS = ("train", "val", "test")


class LabelQAChecker:
    """Checks label quality on a YOLO dataset that has already passed validation."""

    def check(self, inp: LabelQAInput) -> LabelQAOutput:
        dataset_path = Path(inp.dataset_path).resolve()
        logger.info("Starting label QA", extra={"dataset_path": str(dataset_path)})

        yaml_path = Path(inp.data_yaml_path) if inp.data_yaml_path else dataset_path / "data.yaml"
        if not yaml_path.exists():
            return _failed(f"data.yaml not found at {yaml_path}")

        try:
            class_names = _parse_class_names(_load_yaml(yaml_path))
        except Exception as exc:
            return _failed(f"data.yaml parse error: {exc}")
        if not class_names:
            return _failed("data.yaml: 'names' section is empty or missing")

        reference_model = None
        if inp.reference_model_path:
            try:
                reference_model = _load_reference_model(inp.reference_model_path)
            except Exception as exc:
                return _failed(f"Cannot load reference model: {exc}")

        suspicious: list[SuspiciousSample] = []
        class_distribution: dict[str, int] = defaultdict(int)
        num_images = 0
        num_labels = 0
        splits_found = False

        for split in _SPLITS:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split
            if not img_dir.is_dir():
                continue
            splits_found = True

            for img_path in sorted(_list_images(img_dir)):
                num_images += 1
                label_path = lbl_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    suspicious.append(
                        SuspiciousSample(
                            image=img_path.name,
                            split=split,
                            issue_type=QAIssueType.MISSING_LABEL_FILE,
                            message=f"No label file found for image '{img_path.name}'.",
                        )
                    )
                    continue

                boxes = _parse_label_file(label_path)
                num_labels += 1

                for class_id, xc, yc, w, h, line_num in boxes:
                    cls_name = class_names.get(class_id, f"class_{class_id}")
                    class_distribution[cls_name] += 1
                    suspicious.extend(
                        _check_geometry(
                            img_path.name,
                            split,
                            line_num,
                            class_id,
                            cls_name,
                            xc,
                            yc,
                            w,
                            h,
                            inp,
                        )
                    )

                if reference_model is not None:
                    suspicious.extend(
                        _compare_with_reference(
                            reference_model, img_path, boxes, class_names, inp, split
                        )
                    )

        if not splits_found:
            return _failed(
                f"No images/{{{','.join(_SPLITS)}}} directories found under {dataset_path}"
            )

        suspicious.extend(_check_class_imbalance(class_distribution, inp))

        num_suspicious = len(suspicious)
        score = max(0.0, 1.0 - num_suspicious / max(num_labels, 1))
        status = (
            LabelQAStatus.REVIEW_REQUIRED
            if num_suspicious >= inp.review_required_threshold
            else LabelQAStatus.PASSED
        )

        logger.info(
            "Label QA complete",
            extra={
                "status": status,
                "num_images": num_images,
                "num_suspicious": num_suspicious,
                "score": score,
            },
        )

        return LabelQAOutput(
            success=True,
            message=(
                f"Label QA {status}: {num_suspicious} suspicious sample(s) "
                f"across {num_labels} label file(s)."
            ),
            status=status,
            label_quality_score=score,
            suspicious_samples=suspicious,
            class_distribution=dict(class_distribution),
            num_images_checked=num_images,
            num_labels_checked=num_labels,
            reference_model_used=reference_model is not None,
            warnings=[s.message for s in suspicious],
        )


# ── Geometric checks ──────────────────────────────────────────────────────────


def _check_geometry(
    image: str,
    split: str,
    line_num: int,
    class_id: int,
    cls_name: str,
    xc: float,
    yc: float,
    w: float,
    h: float,
    inp: LabelQAInput,
) -> list[SuspiciousSample]:
    issues: list[SuspiciousSample] = []
    bbox = [xc, yc, w, h]

    if w < inp.too_small_threshold or h < inp.too_small_threshold:
        issues.append(
            SuspiciousSample(
                image=image,
                split=split,
                line=line_num,
                class_id=class_id,
                class_name=cls_name,
                bbox=bbox,
                issue_type=QAIssueType.BBOX_TOO_SMALL,
                message=f"{image}:{line_num} bbox too small: width={w:.4f}, height={h:.4f}",
            )
        )

    if w > inp.too_large_threshold or h > inp.too_large_threshold:
        issues.append(
            SuspiciousSample(
                image=image,
                split=split,
                line=line_num,
                class_id=class_id,
                class_name=cls_name,
                bbox=bbox,
                issue_type=QAIssueType.BBOX_TOO_LARGE,
                message=f"{image}:{line_num} bbox too large: width={w:.4f}, height={h:.4f}",
            )
        )

    margin = inp.boundary_margin
    if (
        (xc - w / 2) <= margin
        or (xc + w / 2) >= (1 - margin)
        or (yc - h / 2) <= margin
        or (yc + h / 2) >= (1 - margin)
    ):
        issues.append(
            SuspiciousSample(
                image=image,
                split=split,
                line=line_num,
                class_id=class_id,
                class_name=cls_name,
                bbox=bbox,
                issue_type=QAIssueType.BBOX_NEAR_BOUNDARY,
                message=f"{image}:{line_num} bbox touches image boundary.",
            )
        )

    ratio = max(w, h) / max(min(w, h), 1e-6)
    if ratio > inp.max_aspect_ratio:
        issues.append(
            SuspiciousSample(
                image=image,
                split=split,
                line=line_num,
                class_id=class_id,
                class_name=cls_name,
                bbox=bbox,
                issue_type=QAIssueType.SUSPICIOUS_ASPECT_RATIO,
                message=f"{image}:{line_num} suspicious aspect ratio: {ratio:.1f}:1",
            )
        )

    return issues


def _check_class_imbalance(
    class_distribution: dict[str, int], inp: LabelQAInput
) -> list[SuspiciousSample]:
    issues: list[SuspiciousSample] = []
    total = sum(class_distribution.values())
    if total == 0:
        return issues
    for cls_name, count in class_distribution.items():
        ratio = count / total
        if ratio < inp.class_imbalance_ratio:
            issues.append(
                SuspiciousSample(
                    image="",
                    split="",
                    class_name=cls_name,
                    issue_type=QAIssueType.CLASS_IMBALANCE,
                    message=(
                        f"Class '{cls_name}' has only {ratio:.1%} of total annotations "
                        f"({count}/{total})."
                    ),
                )
            )
    return issues


# ── Reference model disagreement ──────────────────────────────────────────────


def _load_reference_model():
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc
    return YOLO


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU between two normalised (xc, yc, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_predictions(result) -> list[tuple[int, float, float, float, float, float]]:
    """Parse an Ultralytics predict() Results object into (class_id, xc, yc, w, h, conf)."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn
    cls = boxes.cls
    conf = boxes.conf
    preds = []
    for i in range(len(boxes)):
        xc, yc, w, h = (float(v) for v in xywhn[i])
        preds.append((int(cls[i]), xc, yc, w, h, float(conf[i])))
    return preds


def _compare_with_reference(
    model_cls,
    img_path: Path,
    human_boxes: list[tuple[int, float, float, float, float, int]],
    class_names: dict[int, str],
    inp: LabelQAInput,
    split: str,
) -> list[SuspiciousSample]:
    model = model_cls(inp.reference_model_path)
    results = model.predict(
        source=str(img_path), conf=inp.reference_model_confidence, verbose=False
    )
    predictions = _extract_predictions(results[0])

    matched_pred: set[int] = set()
    matched_human: set[int] = set()
    for pi, (_pcls, pxc, pyc, pw, ph, _pconf) in enumerate(predictions):
        best_iou, best_hi = 0.0, None
        for hi, (_hcls, hxc, hyc, hw, hh, _line) in enumerate(human_boxes):
            if hi in matched_human:
                continue
            iou = _bbox_iou((pxc, pyc, pw, ph), (hxc, hyc, hw, hh))
            if iou > best_iou:
                best_iou, best_hi = iou, hi
        if best_iou >= inp.reference_model_iou_threshold:
            matched_human.add(best_hi)  # type: ignore[arg-type]
            matched_pred.add(pi)

    issues: list[SuspiciousSample] = []
    for pi, (pcls, pxc, pyc, pw, ph, pconf) in enumerate(predictions):
        if pi in matched_pred:
            continue
        cls_name = class_names.get(pcls, f"class_{pcls}")
        issues.append(
            SuspiciousSample(
                image=img_path.name,
                split=split,
                class_id=pcls,
                class_name=cls_name,
                bbox=[pxc, pyc, pw, ph],
                issue_type=QAIssueType.REFERENCE_MODEL_DISAGREEMENT,
                message=(
                    f"{img_path.name}: reference model detected '{cls_name}' "
                    f"(conf={pconf:.2f}) with no matching human label — possible missing label."
                ),
            )
        )
    return issues


# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_class_names(data_yaml: dict) -> dict[int, str]:
    names = data_yaml.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def _list_images(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in _IMAGE_EXTS]


def _parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float, int]]:
    """Parse a YOLO label file, silently skipping malformed lines.

    Structural validity (correct column count, in-range values) is
    DatasetValidationAgent's job, not this tool's — it assumes a passing dataset.
    """
    boxes: list[tuple[int, float, float, float, float, int]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return boxes

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append((class_id, xc, yc, w, h, line_num))
    return boxes


def _failed(message: str) -> LabelQAOutput:
    return LabelQAOutput(
        success=False,
        message=message,
        status=LabelQAStatus.FAILED,
        errors=[message],
    )
