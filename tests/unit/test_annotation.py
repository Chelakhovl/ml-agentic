"""Unit tests for the Annotation / Pseudo-label Agent, tool, and CLI command.

Coverage matrix:
    1.  images_path not found -> failed
    2.  No images found -> failed
    3.  Model load failure -> failed
    4.  All detections >= auto_candidate -> high confidence bucket
    5.  min confidence between human_review and auto_candidate -> medium bucket
    6.  min confidence below human_review -> low bucket
    7.  Zero detections -> medium bucket, empty label file written
    8.  Existing label present -> image skipped (not re-predicted), recorded
    9.  skip_existing_labels=False -> image still predicted even if labeled
   10.  review_queue.json contains only medium/low bucket entries
   11.  Pseudo-labels never written into existing_labels_path (separate directory)
   12.  ConfidenceThresholds: human_review > auto_candidate rejected
   13.  AnnotationAgent writes pseudo_label_report.json / .md
   14.  AnnotationAgent logs to MLflow when enabled
   15.  CLI: pseudo-label command exists and succeeds (exit 0)
   16.  CLI: pseudo-label exits 1 on missing images_path
   17.  CLI: pseudo-label exits 1 when model loading fails
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from agentic_mlops.agents.annotation import AnnotationAgent
from agentic_mlops.contracts.annotation import (
    AnnotationInput,
    ConfidenceBucket,
    ConfidenceThresholds,
)
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.pseudo_labeler import PseudoLabeler

# ── Fake YOLO model ────────────────────────────────────────────────────────────


class _FakeBoxes:
    def __init__(self, dets: list[tuple[int, float, float, float, float, float]]) -> None:
        self.xywhn = [(xc, yc, w, h) for (_c, xc, yc, w, h, _conf) in dets]
        self.cls = [c for (c, *_rest) in dets]
        self.conf = [conf for (*_rest, conf) in dets]

    def __len__(self) -> int:
        return len(self.xywhn)


class _FakeResult:
    def __init__(self, dets: list[tuple[int, float, float, float, float, float]]) -> None:
        self.boxes = _FakeBoxes(dets)


class _FakeModel:
    def __init__(self, dets_by_stem: dict[str, list] | None = None, default_dets=None) -> None:
        self._dets_by_stem = dets_by_stem or {}
        self._default = default_dets or []

    def predict(self, source, conf, imgsz, device, verbose=False):  # noqa: ANN001
        stem = Path(source).stem
        dets = self._dets_by_stem.get(stem, self._default)
        return [_FakeResult(dets)]


def _make_images(dir_: Path, names: list[str]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (16, 16), color=(i * 17 % 256, i * 41 % 256, i * 71 % 256)).save(
            dir_ / name
        )


def _run_with_fake_model(inp: AnnotationInput, artifacts_dir: Path, model: _FakeModel):
    with patch(
        "agentic_mlops.tools.pseudo_labeler._import_yolo",
        return_value=lambda path: model,
    ):
        return PseudoLabeler().run(inp, artifacts_dir)


# ── 1-3. Structural failures ──────────────────────────────────────────────────


def test_images_path_not_found_fails(tmp_path: Path) -> None:
    result = PseudoLabeler().run(
        AnnotationInput(images_path=str(tmp_path / "nope"), model_path="fake.pt"),
        tmp_path / "artifacts",
    )
    assert result.success is False
    assert "not found" in result.message.lower()


def test_no_images_fails(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = PseudoLabeler().run(
        AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
        tmp_path / "artifacts",
    )
    assert result.success is False
    assert "no images" in result.message.lower()


def test_model_load_failure_fails(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])

    def _broken_import():
        raise RuntimeError("ultralytics is not installed. Install it with: pip install ultralytics")

    with patch("agentic_mlops.tools.pseudo_labeler._import_yolo", side_effect=_broken_import):
        result = PseudoLabeler().run(
            AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
            tmp_path / "artifacts",
        )

    assert result.success is False
    assert "cannot load model" in result.message.lower()


# ── 4-7. Confidence routing ────────────────────────────────────────────────────


def test_high_confidence_bucket(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95), (0, 0.3, 0.3, 0.1, 0.1, 0.92)])

    result = _run_with_fake_model(
        AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
        tmp_path / "artifacts",
        model,
    )

    assert result.success is True
    assert result.high_confidence_count == 1
    assert result.records[0].bucket == ConfidenceBucket.HIGH


def test_medium_confidence_bucket(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.7)])

    result = _run_with_fake_model(
        AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
        tmp_path / "artifacts",
        model,
    )

    assert result.medium_confidence_count == 1
    assert result.records[0].bucket == ConfidenceBucket.MEDIUM


def test_low_confidence_bucket(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    # one high-conf and one low-conf detection -> min drives the bucket -> low
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95), (0, 0.2, 0.2, 0.1, 0.1, 0.3)])

    result = _run_with_fake_model(
        AnnotationInput(
            images_path=str(images_dir),
            model_path="fake.pt",
            confidence_thresholds=ConfidenceThresholds(auto_candidate=0.9, human_review=0.5),
        ),
        tmp_path / "artifacts",
        model,
    )

    assert result.low_confidence_count == 1
    assert result.records[0].bucket == ConfidenceBucket.LOW


def test_zero_detections_is_medium_and_empty_label(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[])
    artifacts_dir = tmp_path / "artifacts"

    result = _run_with_fake_model(
        AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
        artifacts_dir,
        model,
    )

    assert result.medium_confidence_count == 1
    assert result.records[0].num_detections == 0
    label_path = Path(result.records[0].label_path)
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8") == ""


# ── 8-11. Existing-label safety rule ──────────────────────────────────────────


def test_existing_label_skips_prediction(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg", "b.jpg"])
    existing_labels = tmp_path / "existing_labels"
    existing_labels.mkdir()
    (existing_labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])
    result = _run_with_fake_model(
        AnnotationInput(
            images_path=str(images_dir),
            model_path="fake.pt",
            existing_labels_path=str(existing_labels),
        ),
        tmp_path / "artifacts",
        model,
    )

    assert result.num_images_skipped_existing == 1
    assert result.num_images_processed == 1
    skipped = next(r for r in result.records if r.image == "a.jpg")
    assert skipped.skipped_existing_label is True
    assert skipped.bucket is None


def test_skip_existing_labels_false_still_predicts(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    existing_labels = tmp_path / "existing_labels"
    existing_labels.mkdir()
    (existing_labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])
    result = _run_with_fake_model(
        AnnotationInput(
            images_path=str(images_dir),
            model_path="fake.pt",
            existing_labels_path=str(existing_labels),
            skip_existing_labels=False,
        ),
        tmp_path / "artifacts",
        model,
    )

    assert result.num_images_skipped_existing == 0
    assert result.num_images_processed == 1


def test_review_queue_contains_only_medium_and_low(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["high.jpg", "medium.jpg", "low.jpg"])
    model = _FakeModel(
        dets_by_stem={
            "high": [(0, 0.5, 0.5, 0.2, 0.2, 0.95)],
            "medium": [(0, 0.5, 0.5, 0.2, 0.2, 0.7)],
            "low": [(0, 0.5, 0.5, 0.2, 0.2, 0.3)],
        }
    )
    artifacts_dir = tmp_path / "artifacts"

    result = _run_with_fake_model(
        AnnotationInput(images_path=str(images_dir), model_path="fake.pt"),
        artifacts_dir,
        model,
    )

    queue = json.loads(Path(result.review_queue_path).read_text(encoding="utf-8"))
    images_in_queue = {item["image"] for item in queue}
    assert images_in_queue == {"medium.jpg", "low.jpg"}


def test_pseudo_labels_never_written_to_existing_labels_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    existing_labels = tmp_path / "existing_labels"
    existing_labels.mkdir()

    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])
    result = _run_with_fake_model(
        AnnotationInput(
            images_path=str(images_dir),
            model_path="fake.pt",
            existing_labels_path=str(existing_labels),
        ),
        tmp_path / "artifacts",
        model,
    )

    assert not (existing_labels / "a.txt").exists()
    assert Path(result.pseudo_labels_path) != existing_labels
    assert (Path(result.pseudo_labels_path) / "a.txt").exists()


# ── 12. Contract validation ────────────────────────────────────────────────────


def test_thresholds_ordering_validated() -> None:
    with pytest.raises(ValueError, match="must be <="):
        ConfidenceThresholds(auto_candidate=0.5, human_review=0.9)


# ── 13-14. AnnotationAgent ──────────────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])
    artifacts_dir = tmp_path / "artifacts"

    with patch("agentic_mlops.tools.pseudo_labeler._import_yolo", return_value=lambda path: model):
        agent = AnnotationAgent(artifacts_dir=artifacts_dir)
        result = agent.run(AnnotationInput(images_path=str(images_dir), model_path="fake.pt"))

    assert (artifacts_dir / "pseudo_label_report.json").exists()
    assert (artifacts_dir / "pseudo_label_report.md").exists()
    assert result.report_path == str(artifacts_dir / "pseudo_label_report.json")

    data = json.loads((artifacts_dir / "pseudo_label_report.json").read_text(encoding="utf-8"))
    assert data["success"] is True
    assert data["high_confidence_count"] == 1


def test_agent_logs_to_mlflow_when_enabled(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    with patch("agentic_mlops.tools.pseudo_labeler._import_yolo", return_value=lambda path: model):
        agent = AnnotationAgent(
            artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
        )
        agent.run(AnnotationInput(images_path=str(images_dir), model_path="fake.pt"))

    metrics = client.runs[run_id]["metrics"]
    assert "annotation.high_confidence_count" in metrics
    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "annotation"
    assert len(client.runs[run_id]["artifacts"]) > 0


# ── 15-17. CLI ─────────────────────────────────────────────────────────────────


def test_cli_pseudo_label_succeeds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])
    model = _FakeModel(default_dets=[(0, 0.5, 0.5, 0.2, 0.2, 0.95)])

    with patch("agentic_mlops.tools.pseudo_labeler._import_yolo", return_value=lambda path: model):
        result = CliRunner().invoke(
            app, ["pseudo-label", str(images_dir), "--model-path", "fake.pt"]
        )

    assert result.exit_code == 0, result.output
    assert "SUCCESS" in result.output


def test_cli_pseudo_label_exits_1_on_missing_images_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app, ["pseudo-label", str(tmp_path / "nope"), "--model-path", "fake.pt"]
    )

    assert result.exit_code == 1


def test_cli_pseudo_label_exits_1_on_model_load_failure(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    images_dir = tmp_path / "images"
    _make_images(images_dir, ["a.jpg"])

    def _broken_import():
        raise RuntimeError("ultralytics is not installed.")

    with patch("agentic_mlops.tools.pseudo_labeler._import_yolo", side_effect=_broken_import):
        result = CliRunner().invoke(
            app, ["pseudo-label", str(images_dir), "--model-path", "fake.pt"]
        )

    assert result.exit_code == 1
