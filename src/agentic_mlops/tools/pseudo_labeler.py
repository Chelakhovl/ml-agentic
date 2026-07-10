"""Pseudo-labeler — the core tool used by the Annotation / Pseudo-label Agent.

Runs an approved/pre-trained YOLO model over unlabeled (or partially labeled)
images, writes candidate YOLO labels, and routes each image into a confidence
bucket (high / medium / low) so uncertain and hard cases go to human review.

Safety rule: never overwrites existing human labels, and pseudo-labels are
always written to a separate pseudo_labels/ directory — never merged into an
existing dataset's labels/ automatically. Promoting them to final labels is a
deliberate separate step outside this tool.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_mlops.contracts.annotation import (
    AnnotationInput,
    AnnotationOutput,
    ConfidenceBucket,
    PseudoLabelRecord,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class PseudoLabeler:
    """Pre-labels images with a YOLO model and routes them by confidence."""

    def run(self, inp: AnnotationInput, artifacts_dir: Path) -> AnnotationOutput:
        images_dir = Path(inp.images_path).resolve()
        if not images_dir.is_dir():
            return _failed(f"images_path not found or not a directory: {images_dir}")

        images = sorted(
            p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            return _failed(f"No images found under {images_dir}")

        try:
            YOLO = _import_yolo()
            model = YOLO(inp.model_path)
        except Exception as exc:
            return _failed(f"Cannot load model '{inp.model_path}': {exc}")

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_dir = artifacts_dir / "pseudo_labels"
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

        existing_labels_dir = (
            Path(inp.existing_labels_path).resolve() if inp.existing_labels_path else None
        )

        records: list[PseudoLabelRecord] = []
        counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        num_skipped = 0

        for img in images:
            if inp.skip_existing_labels and existing_labels_dir is not None:
                existing_label = existing_labels_dir / (img.stem + ".txt")
                if existing_label.exists():
                    num_skipped += 1
                    records.append(
                        PseudoLabelRecord(image=img.name, skipped_existing_label=True)
                    )
                    continue

            try:
                results = model.predict(
                    source=str(img),
                    conf=inp.confidence_thresholds.human_review,
                    imgsz=inp.imgsz,
                    device=inp.device,
                    verbose=False,
                )
                predictions = _extract_predictions(results[0])
            except Exception as exc:
                logger.error(
                    "Prediction failed for image", extra={"image": img.name, "error": str(exc)}
                )
                return _failed(f"Prediction failed on '{img.name}': {exc}")

            bucket, mean_conf, min_conf = _bucket_for(predictions, inp.confidence_thresholds)
            counts[bucket] += 1

            label_lines = [
                f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                for cls, xc, yc, w, h, _conf in predictions
            ]
            label_path = pseudo_labels_dir / (img.stem + ".txt")
            content = "\n".join(label_lines)
            label_path.write_text(content + ("\n" if content else ""), encoding="utf-8")

            records.append(
                PseudoLabelRecord(
                    image=img.name,
                    bucket=ConfidenceBucket(bucket),
                    num_detections=len(predictions),
                    mean_confidence=mean_conf,
                    min_confidence=min_conf,
                    label_path=str(label_path),
                )
            )

        review_queue = [
            r.model_dump()
            for r in records
            if r.bucket in (ConfidenceBucket.MEDIUM, ConfidenceBucket.LOW)
        ]
        review_queue_path = artifacts_dir / "review_queue.json"
        review_queue_path.write_text(
            json.dumps(review_queue, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        num_processed = len(images) - num_skipped
        logger.info(
            "Pseudo-labeling complete",
            extra={
                "num_processed": num_processed,
                "num_skipped": num_skipped,
                "counts": counts,
            },
        )

        return AnnotationOutput(
            success=True,
            message=(
                f"Pseudo-labeled {num_processed} image(s) "
                f"({counts['high']} high / {counts['medium']} medium / {counts['low']} low "
                f"confidence); {num_skipped} skipped (already labeled)."
            ),
            pseudo_labels_path=str(pseudo_labels_dir),
            review_queue_path=str(review_queue_path),
            high_confidence_count=counts["high"],
            medium_confidence_count=counts["medium"],
            low_confidence_count=counts["low"],
            num_images_processed=num_processed,
            num_images_skipped_existing=num_skipped,
            records=records,
        )


# ── Confidence routing ─────────────────────────────────────────────────────────


def _bucket_for(
    predictions: list[tuple[int, float, float, float, float, float]],
    thresholds,  # ConfidenceThresholds
) -> tuple[str, float | None, float | None]:
    if not predictions:
        # No detections above the human_review floor at all: could be a genuinely
        # empty/background image, or a missed object — can't tell without a human,
        # so route to standard review rather than auto-accepting an empty label.
        return "medium", None, None

    confs = [p[5] for p in predictions]
    mean_conf = sum(confs) / len(confs)
    min_conf = min(confs)

    if min_conf >= thresholds.auto_candidate:
        return "high", mean_conf, min_conf
    if min_conf >= thresholds.human_review:
        return "medium", mean_conf, min_conf
    return "low", mean_conf, min_conf


# ── Ultralytics glue ────────────────────────────────────────────────────────────


def _import_yolo():
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc


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


def _failed(message: str) -> AnnotationOutput:
    return AnnotationOutput(success=False, message=message, errors=[message])
