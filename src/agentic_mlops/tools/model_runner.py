"""ModelRunner — framework-agnostic interface for train / evaluate / predict.

``ModelRunner`` is a ``typing.Protocol`` so any class that provides the three
methods (train, evaluate, predict) satisfies it without inheriting from it.

Concrete implementations
------------------------
- ``YoloModelRunner``      — Ultralytics YOLO (train + evaluate + predict).
- ``OnnxOnlyModelRunner``  — ONNX runtime inference only (predict); train/evaluate
                             raise ``NotImplementedError`` since ONNX models are
                             pre-trained artifacts.
- ``TorchvisionModelRunner`` — stub reserved for future PyTorch/TorchVision support;
                               all three methods raise ``NotImplementedError``.

Usage::

    runner = YoloModelRunner()
    train_out = runner.train(training_input, artifacts_dir)
    eval_out  = runner.evaluate(evaluation_input, artifacts_dir)
    pred_out  = runner.predict(annotation_input, artifacts_dir)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_mlops.contracts.annotation import (
    AnnotationInput,
    AnnotationOutput,
    ConfidenceBucket,
    PseudoLabelRecord,
)
from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationOutput
from agentic_mlops.contracts.training import TrainingInput, TrainingOutput
from agentic_mlops.observability.logging import get_logger

if TYPE_CHECKING:
    from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner
    from agentic_mlops.tools.training_runner import AzureMLTrainingRunner

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@runtime_checkable
class ModelRunner(Protocol):
    """Framework-agnostic interface wrapping train / evaluate / predict.

    Any class that provides these three methods satisfies the protocol — no
    inheritance required.  The ``framework`` field on each input contract
    carries which framework was requested, but the runner itself does not need
    to inspect it: routing is done by the factory that *selects* the runner.
    """

    def train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        """Run a full training job and return the output contract."""
        ...

    def evaluate(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        """Evaluate a model checkpoint and return the output contract."""
        ...

    def predict(self, inp: AnnotationInput, artifacts_dir: Path) -> AnnotationOutput:
        """Run inference over a directory of images (pseudo-labeling)."""
        ...


# ── YOLO ──────────────────────────────────────────────────────────────────────


class YoloModelRunner:
    """Ultralytics YOLO implementation of ``ModelRunner``.

    Delegates to:
    - ``YoloTrainer``  for ``train()``
    - ``YoloEvaluator`` for ``evaluate()``
    - ``PseudoLabeler`` for ``predict()``

    Inject ``azure_trainer`` / ``azure_evaluator`` to enable Azure ML modes;
    those remain optional so dry-run and local-YOLO use cases work with no
    Azure SDK installed.
    """

    def __init__(
        self,
        azure_trainer: AzureMLTrainingRunner | None = None,
        azure_evaluator: AzureMLEvaluationRunner | None = None,
    ) -> None:
        from agentic_mlops.tools.pseudo_labeler import PseudoLabeler  # noqa: PLC0415
        from agentic_mlops.tools.yolo_evaluator import YoloEvaluator  # noqa: PLC0415
        from agentic_mlops.tools.yolo_trainer import YoloTrainer  # noqa: PLC0415

        self._trainer = YoloTrainer(azure_runner=azure_trainer)
        self._evaluator = YoloEvaluator(azure_runner=azure_evaluator)
        self._labeler = PseudoLabeler()

    def train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        return self._trainer.run(inp, artifacts_dir)

    def evaluate(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        return self._evaluator.run(inp, artifacts_dir)

    def predict(self, inp: AnnotationInput, artifacts_dir: Path) -> AnnotationOutput:
        return self._labeler.run(inp, artifacts_dir)


# ── ONNX-only ─────────────────────────────────────────────────────────────────


class OnnxOnlyModelRunner:
    """Inference-only runner for ONNX models.

    ONNX models are pre-trained artifacts — training and evaluation in the
    MLOps sense are not supported.  Only ``predict()`` is implemented.

    Requires ``onnxruntime`` (``pip install onnxruntime``) and ``Pillow``
    (``pip install -e '.[vision]'``).  Both are checked lazily at call time so
    the class can be imported and the protocol isinstance check passed without
    either package installed.

    Output format matches ``PseudoLabeler`` exactly (``AnnotationOutput``).

    ONNX output shape detection
    ----------------------------
    Two common YOLO export formats are handled automatically:

    * ``[batch, n_det, 6]``   — YOLOv5-style: each row is
      ``(x1_px, y1_px, x2_px, y2_px, conf, class_id)`` in absolute pixel coords.
    * ``[batch, 4+nc, n_det]`` — YOLOv8/YOLO11-style (transposed): each column is
      ``(xc, yc, w, h, cls0_score, …, cls_N_score)`` already normalized to [0,1].

    The heuristic: if ``output.shape[2] == 6`` → format 1; otherwise transpose
    and treat as format 2.
    """

    def train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        raise NotImplementedError(
            "ONNX models are pre-trained — training is not supported by OnnxOnlyModelRunner. "
            "Use YoloModelRunner (or another framework runner) to train from scratch."
        )

    def evaluate(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        raise NotImplementedError(
            "Use YoloModelRunner or a compatible runner for evaluation — "
            "OnnxOnlyModelRunner only supports inference (predict)."
        )

    def predict(self, inp: AnnotationInput, artifacts_dir: Path) -> AnnotationOutput:  # noqa: PLR0912
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except (ImportError, TypeError):
            return _ann_failed(
                "onnxruntime is not installed — pip install onnxruntime"
            )

        try:
            from PIL import Image  # noqa: PLC0415
        except (ImportError, TypeError):
            return _ann_failed(
                "Pillow is not installed — pip install Pillow (or pip install -e '.[vision]')"
            )

        images_dir = Path(inp.images_path).resolve()
        if not images_dir.is_dir():
            return _ann_failed(f"images_path not found or not a directory: {images_dir}")

        images = sorted(
            p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            return _ann_failed(f"No images found under {images_dir}")

        try:
            session = ort.InferenceSession(inp.model_path)
        except Exception as exc:
            return _ann_failed(f"Cannot load ONNX model '{inp.model_path}': {exc}")

        input_name = session.get_inputs()[0].name
        imgsz = inp.imgsz

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_dir = artifacts_dir / "pseudo_labels"
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

        existing_labels_dir = (
            Path(inp.existing_labels_path).resolve() if inp.existing_labels_path else None
        )

        import numpy as np  # noqa: PLC0415

        records: list[PseudoLabelRecord] = []
        counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        num_skipped = 0
        thresholds = inp.confidence_thresholds

        for img_path in images:
            if inp.skip_existing_labels and existing_labels_dir is not None:
                if (existing_labels_dir / (img_path.stem + ".txt")).exists():
                    num_skipped += 1
                    records.append(
                        PseudoLabelRecord(image=img_path.name, skipped_existing_label=True)
                    )
                    continue

            try:
                img = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
                arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
                tensor = arr.transpose(2, 0, 1)[np.newaxis]  # BCHW
                raw = session.run(None, {input_name: tensor})[0]  # first output
            except Exception as exc:
                logger.error(
                    "ONNX inference failed",
                    extra={"image": img_path.name, "error": str(exc)},
                )
                return _ann_failed(f"Inference failed on '{img_path.name}': {exc}")

            preds = _parse_onnx_output(raw, thresholds.human_review)

            bucket_name, mean_conf, min_conf = _bucket_for_confs(preds, thresholds)
            counts[bucket_name] += 1

            label_lines = [
                f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h, _c in preds
            ]
            label_path = pseudo_labels_dir / (img_path.stem + ".txt")
            content = "\n".join(label_lines)
            label_path.write_text(content + ("\n" if content else ""), encoding="utf-8")

            records.append(
                PseudoLabelRecord(
                    image=img_path.name,
                    bucket=ConfidenceBucket(bucket_name),
                    num_detections=len(preds),
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
        return AnnotationOutput(
            success=True,
            message=(
                f"ONNX inference complete: {num_processed} image(s) processed "
                f"({counts['high']} high / {counts['medium']} medium / {counts['low']} low); "
                f"{num_skipped} skipped."
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


# ── TorchVision (stub) ────────────────────────────────────────────────────────


class TorchvisionModelRunner:
    """Stub runner reserved for future PyTorch / TorchVision support.

    All three methods raise ``NotImplementedError`` explaining what would be
    needed to implement them:

    * ``torch`` and ``torchvision`` packages installed.
    * A ``TorchvisionTrainingConfig`` contract (not yet defined) describing the
      model architecture, optimizer, scheduler, and augmentation pipeline.
    * A ``TorchvisionEvaluationRunner`` that computes COCO-style mAP metrics.
    """

    _MSG = (
        "TorchvisionModelRunner is a reserved stub — it is not yet implemented. "
        "To add support: install torch + torchvision, define TorchvisionTrainingConfig "
        "in contracts/training.py, and implement TorchvisionModelRunner.{method}()."
    )

    def train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        raise NotImplementedError(self._MSG.format(method="train"))

    def evaluate(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        raise NotImplementedError(self._MSG.format(method="evaluate"))

    def predict(self, inp: AnnotationInput, artifacts_dir: Path) -> AnnotationOutput:
        raise NotImplementedError(self._MSG.format(method="predict"))


# ── Factory ───────────────────────────────────────────────────────────────────


def create_model_runner(framework: str = "yolo", **kwargs) -> ModelRunner:
    """Return the ``ModelRunner`` for *framework*.

    Args:
        framework: ``"yolo"`` (default), ``"onnx_only"``, or ``"torchvision"``.
        **kwargs: Passed to the runner constructor.  For ``"yolo"``:
            ``azure_trainer`` and ``azure_evaluator`` are accepted.

    Returns:
        A ``ModelRunner`` instance.

    Raises:
        ValueError: Unknown framework name.
    """
    if framework == "yolo":
        return YoloModelRunner(**kwargs)
    if framework == "onnx_only":
        return OnnxOnlyModelRunner(**kwargs)
    if framework == "torchvision":
        return TorchvisionModelRunner(**kwargs)
    raise ValueError(
        f"Unknown framework '{framework}'. "
        "Supported values: 'yolo', 'onnx_only', 'torchvision'."
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ann_failed(message: str) -> AnnotationOutput:
    return AnnotationOutput(success=False, message=message, errors=[message])


def _parse_onnx_output(
    raw,  # np.ndarray
    conf_threshold: float,
) -> list[tuple[int, float, float, float, float, float]]:
    """Parse raw ONNX output into ``(class_id, xc, yc, w, h, conf)`` tuples.

    Handles two common export shapes:

    * ``[batch, n_det, 6]``   — each row: ``x1_px, y1_px, x2_px, y2_px, conf, cls``
      (absolute pixel coords → normalised here using the session imgsz).
    * ``[batch, 4+nc, n_det]`` — transposed: each column: ``xc, yc, w, h, cls0, …``
      (already normalised to [0,1]).
    """
    import numpy as np  # noqa: PLC0415

    output = np.array(raw)
    if output.ndim == 2:
        output = output[np.newaxis]  # add batch dim if missing

    batch = output[0]  # [n_det, 6] or [4+nc, n_det]

    preds: list[tuple[int, float, float, float, float, float]] = []

    # Detect format by last dimension size
    if batch.ndim == 2 and batch.shape[1] == 6:
        # YOLOv5-style: [n_det, 6] = x1,y1,x2,y2,conf,cls (absolute pixel)
        # We don't know the original image size here, so normalize by imgsz (assumed square).
        # This is a best-effort approximation; the caller uses inp.imgsz for the resize.
        for row in batch:
            x1, y1, x2, y2, conf, cls = row
            conf = float(conf)
            if conf < conf_threshold:
                continue
            xc = float((x1 + x2) / 2)
            yc = float((y1 + y2) / 2)
            w = float(x2 - x1)
            h = float(y2 - y1)
            preds.append((int(cls), xc, yc, w, h, conf))
    else:
        # YOLOv8/YOLO11-style: [4+nc, n_det] — transpose to [n_det, 4+nc]
        if batch.ndim == 2 and batch.shape[0] < batch.shape[1]:
            batch = batch.T  # now [n_det, 4+nc]
        for row in batch:
            if len(row) < 5:
                continue
            xc, yc, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4:]
            cls = int(np.argmax(class_scores))
            conf = float(class_scores[cls])
            if conf < conf_threshold:
                continue
            preds.append((cls, xc, yc, w, h, conf))

    return preds


def _bucket_for_confs(
    preds: list[tuple[int, float, float, float, float, float]],
    thresholds,
) -> tuple[str, float | None, float | None]:
    """Map a list of predictions to a confidence bucket (mirrors PseudoLabeler)."""
    if not preds:
        return "medium", None, None
    confs = [p[5] for p in preds]
    mean_conf = sum(confs) / len(confs)
    min_conf = min(confs)
    if min_conf >= thresholds.auto_candidate:
        return "high", mean_conf, min_conf
    if min_conf >= thresholds.human_review:
        return "medium", mean_conf, min_conf
    return "low", mean_conf, min_conf
