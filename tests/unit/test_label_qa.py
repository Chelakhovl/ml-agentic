"""Unit tests for the Label QA Agent, checker tool, and CLI command.

Coverage matrix:
    1.  Clean dataset -> status passed, score 1.0, no suspicious samples
    2.  Missing data.yaml -> status failed
    3.  Empty 'names' section -> status failed
    4.  No images/{train,val,test} directories -> status failed
    5.  bbox too small flagged
    6.  bbox too large flagged
    7.  bbox near boundary flagged
    8.  suspicious aspect ratio flagged
    9.  missing label file flagged
   10.  class imbalance flagged
   11.  status flips to review_required once suspicious count hits threshold
   12.  label_quality_score decreases with suspicious samples
   13.  reference model flags an unmatched prediction as disagreement
   14.  reference model match (high IoU) does NOT get flagged
   15.  reference_model_used is False when no reference_model_path given
   16.  LabelQAAgent writes label_quality_report.json / .md
   17.  LabelQAAgent logs to MLflow when enabled
   18.  CLI: label-qa command exists and runs on a clean dataset (exit 0)
   19.  CLI: label-qa exits 1 when review_required
   20.  CLI: label-qa exits 1 when dataset missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_mlops.agents.label_qa import LabelQAAgent
from agentic_mlops.contracts.label_qa import LabelQAInput, LabelQAStatus, QAIssueType
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.label_qa_checker import LabelQAChecker
from tests.conftest import make_data_yaml, make_image, make_label, make_valid_dataset

# ── Fake reference model ──────────────────────────────────────────────────────


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
    def __init__(self, dets: list[tuple[int, float, float, float, float, float]]) -> None:
        self._dets = dets

    def predict(self, source, conf, verbose=False):  # noqa: ANN001
        return [_FakeResult(self._dets)]


def _fake_model_cls(dets: list[tuple[int, float, float, float, float, float]]):
    def _cls(path):  # noqa: ANN001
        return _FakeModel(dets)

    return _cls


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_single_image_dataset(
    tmp_path: Path,
    label_content: str,
    class_names: list[str] | None = None,
    split: str = "train",
) -> Path:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, class_names or ["scratch", "dent"])
    img = root / "images" / split / "img_000.jpg"
    lbl = root / "labels" / split / "img_000.txt"
    make_image(img)
    make_label(lbl, label_content)
    return root


# ── 1-4. Structural checks ────────────────────────────────────────────────────


def test_clean_dataset_passes(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert result.success is True
    assert result.status == LabelQAStatus.PASSED
    assert result.label_quality_score == pytest.approx(1.0)
    assert result.suspicious_samples == []
    assert result.num_images_checked == 6
    assert result.num_labels_checked == 6


def test_missing_data_yaml_fails(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert result.success is False
    assert result.status == LabelQAStatus.FAILED
    assert "data.yaml" in result.message


def test_empty_names_fails(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert result.success is False
    assert result.status == LabelQAStatus.FAILED


def test_no_split_dirs_fails(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, ["scratch"])

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert result.success is False
    assert result.status == LabelQAStatus.FAILED
    assert "images/" in result.message


# ── 5-10. Geometric / statistical checks ──────────────────────────────────────


def test_bbox_too_small_flagged(tmp_path: Path) -> None:
    root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.005 0.05\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.BBOX_TOO_SMALL in types


def test_bbox_too_large_flagged(tmp_path: Path) -> None:
    root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.95 0.3\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.BBOX_TOO_LARGE in types


def test_bbox_near_boundary_flagged(tmp_path: Path) -> None:
    # xc - w/2 = 0.005 - 0.005 = 0.0 <= default margin 0.01
    root = _make_single_image_dataset(tmp_path, "0 0.05 0.5 0.09 0.2\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.BBOX_NEAR_BOUNDARY in types


def test_suspicious_aspect_ratio_flagged(tmp_path: Path) -> None:
    # ratio = 0.5 / 0.02 = 25 > default max_ratio 10
    root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.5 0.02\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.SUSPICIOUS_ASPECT_RATIO in types


def test_missing_label_file_flagged(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, ["scratch"])
    make_image(root / "images" / "train" / "orphan.jpg")
    # no matching labels/train/orphan.txt

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.MISSING_LABEL_FILE in types
    assert result.num_images_checked == 1
    assert result.num_labels_checked == 0


def test_class_imbalance_flagged(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, ["common", "rare"])
    # 18 'common' boxes across many images, 1 'rare' box -> rare ratio ~5% < 15%
    lines = ["0 0.5 0.5 0.2 0.2\n"] * 18
    for i, line in enumerate(lines):
        make_image(root / "images" / "train" / f"img_{i:03d}.jpg")
        make_label(root / "labels" / "train" / f"img_{i:03d}.txt", line)
    make_image(root / "images" / "train" / "img_rare.jpg")
    make_label(root / "labels" / "train" / "img_rare.txt", "1 0.5 0.5 0.2 0.2\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    imbalance = [
        s for s in result.suspicious_samples if s.issue_type == QAIssueType.CLASS_IMBALANCE
    ]
    assert any(s.class_name == "rare" for s in imbalance)


# ── 11-12. Status / score derivation ──────────────────────────────────────────


def test_status_flips_to_review_required(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, ["scratch"])
    # 6 images, each with one too-small bbox -> 6 suspicious samples >= default threshold 5
    for i in range(6):
        make_image(root / "images" / "train" / f"img_{i:03d}.jpg")
        make_label(root / "labels" / "train" / f"img_{i:03d}.txt", "0 0.5 0.5 0.005 0.005\n")

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert len(result.suspicious_samples) >= 5
    assert result.status == LabelQAStatus.REVIEW_REQUIRED


def test_quality_score_decreases_with_suspicious_samples(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(clean_root)
    clean_result = LabelQAChecker().check(LabelQAInput(dataset_path=str(clean_root)))

    bad_root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.005 0.005\n")
    bad_result = LabelQAChecker().check(LabelQAInput(dataset_path=str(bad_root)))

    assert bad_result.label_quality_score < clean_result.label_quality_score


# ── 13-15. Reference model disagreement ───────────────────────────────────────


def test_reference_model_flags_unmatched_prediction(tmp_path: Path) -> None:
    root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.2 0.2\n")
    # Prediction at a totally different location -> IoU 0 with the human box
    fake_cls = _fake_model_cls([(0, 0.1, 0.1, 0.1, 0.1, 0.9)])

    with patch("agentic_mlops.tools.label_qa_checker._load_reference_model", return_value=fake_cls):
        result = LabelQAChecker().check(
            LabelQAInput(dataset_path=str(root), reference_model_path="fake.pt")
        )

    assert result.reference_model_used is True
    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.REFERENCE_MODEL_DISAGREEMENT in types


def test_reference_model_match_not_flagged(tmp_path: Path) -> None:
    root = _make_single_image_dataset(tmp_path, "0 0.5 0.5 0.2 0.2\n")
    # Prediction matches the human box almost exactly -> high IoU, no disagreement
    fake_cls = _fake_model_cls([(0, 0.5, 0.5, 0.2, 0.2, 0.9)])

    with patch("agentic_mlops.tools.label_qa_checker._load_reference_model", return_value=fake_cls):
        result = LabelQAChecker().check(
            LabelQAInput(dataset_path=str(root), reference_model_path="fake.pt")
        )

    types = [s.issue_type for s in result.suspicious_samples]
    assert QAIssueType.REFERENCE_MODEL_DISAGREEMENT not in types


def test_reference_model_not_used_by_default(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)

    result = LabelQAChecker().check(LabelQAInput(dataset_path=str(root)))

    assert result.reference_model_used is False


# ── 16-17. LabelQAAgent ────────────────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)
    artifacts_dir = tmp_path / "artifacts"

    agent = LabelQAAgent(artifacts_dir=artifacts_dir)
    result = agent.run(LabelQAInput(dataset_path=str(root)))

    assert (artifacts_dir / "label_quality_report.json").exists()
    assert (artifacts_dir / "label_quality_report.md").exists()
    assert result.report_path == str(artifacts_dir / "label_quality_report.json")

    data = json.loads((artifacts_dir / "label_quality_report.json").read_text(encoding="utf-8"))
    assert data["status"] == "passed"


def test_agent_logs_to_mlflow_when_enabled(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = LabelQAAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(LabelQAInput(dataset_path=str(root)))

    metrics = client.runs[run_id]["metrics"]
    assert "label_qa.quality_score" in metrics
    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "label_qa"
    assert len(client.runs[run_id]["artifacts"]) > 0


def test_agent_does_not_log_to_mlflow_when_disabled(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)

    agent = LabelQAAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(LabelQAInput(dataset_path=str(root)))

    assert result.status == LabelQAStatus.PASSED  # just proves it runs without a client


# ── 18-20. CLI ─────────────────────────────────────────────────────────────────


def test_cli_label_qa_passes_on_clean_dataset(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(root)

    result = CliRunner().invoke(app, ["label-qa", str(root)])

    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_cli_label_qa_exits_1_when_review_required(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    make_data_yaml(root, ["scratch"])
    for i in range(6):
        make_image(root / "images" / "train" / f"img_{i:03d}.jpg")
        make_label(root / "labels" / "train" / f"img_{i:03d}.txt", "0 0.5 0.5 0.005 0.005\n")

    result = CliRunner().invoke(app, ["label-qa", str(root)])

    assert result.exit_code == 1
    assert "REVIEW_REQUIRED" in result.output


def test_cli_label_qa_exits_1_when_dataset_missing(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(app, ["label-qa", str(tmp_path / "nonexistent")])

    assert result.exit_code == 1
