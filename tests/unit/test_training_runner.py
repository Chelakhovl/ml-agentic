"""Unit tests for LocalYOLOTrainingRunner.

Coverage matrix:
    1.  YOLO class instantiated with the configured model name
    2.  model.train() called with all correct keyword args
    3.  best.pt copied to artifacts_dir and listed in training_artifacts
    4.  last.pt copied to artifacts_dir and listed in training_artifacts
    5.  results.csv copied to artifacts_dir when present
    6.  Missing best.pt — no crash, best_weights_path is None
    7.  Missing results.csv — no crash, artifact list excludes it
    8.  training_output.json written to artifacts_dir
    9.  training_output.json contains job_id, mode, job_status
   10.  best_weights_path set correctly in TrainingOutput
   11.  job_status == COMPLETED on success
   12.  mode == LOCAL_TRAIN in output
   13.  FAILED status returned when model.train() raises an exception
   14.  Azure ML not called (LocalYOLOTrainingRunner accepts no azure/mlflow clients)
   15.  YoloTrainer.run() delegates LOCAL_TRAIN to LocalYOLOTrainingRunner
   16.  CLI --runner fake → LOCAL_DRY_RUN (dry-run plan produced)
   17.  CLI --runner local-yolo → LOCAL_TRAIN mode
   18.  CLI --runner unknown → exit code 1
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_mlops.contracts.training import (
    TrainingConfig,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
)
from agentic_mlops.tools.training_runner import LocalYOLOTrainingRunner
from agentic_mlops.tools.yolo_trainer import YoloTrainer

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_save_dir(tmp_path: Path) -> Path:
    """Fake YOLO save_dir with weights/best.pt, weights/last.pt, and results.csv."""
    save_dir = tmp_path / "runs" / "train" / "exp"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"fake_best_weights")
    (weights_dir / "last.pt").write_bytes(b"fake_last_weights")
    (save_dir / "results.csv").write_text("epoch,train/loss\n1,0.5\n2,0.3\n", encoding="utf-8")
    return save_dir


@pytest.fixture
def mock_yolo_cls(fake_save_dir: Path):
    """Mock YOLO class. model.trainer.save_dir points to fake_save_dir."""
    mock_model = MagicMock()
    mock_model.trainer.save_dir = str(fake_save_dir)
    mock_model.train.return_value = {}
    MockYOLO = MagicMock(return_value=mock_model)
    return MockYOLO, mock_model


def _make_input(tmp_path: Path, mode: TrainingMode = TrainingMode.LOCAL_TRAIN) -> TrainingInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text("names:\n  0: scratch\n", encoding="utf-8")
    cfg = TrainingConfig(
        model="yolo11m.pt",
        epochs=5,
        imgsz=320,
        batch=4,
        optimizer="auto",
        patience=3,
        project="test_proj",
        name="test_run",
        mode=mode,
    )
    return TrainingInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        training_config=cfg,
        workflow_id="wf_runner_test",
    )


# ── 1-2. YOLO class instantiation and train() args ────────────────────────────


def test_yolo_class_instantiated_with_model_name(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, mock_model = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        runner = LocalYOLOTrainingRunner()
        inp = _make_input(tmp_path)
        runner.run(inp, tmp_path / "artifacts")

    MockYOLO.assert_called_once_with("yolo11m.pt")


def test_model_train_called_with_correct_args(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, mock_model = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        runner = LocalYOLOTrainingRunner()
        inp = _make_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        runner.run(inp, artifacts_dir)

    call_kwargs = mock_model.train.call_args.kwargs
    assert call_kwargs["data"] == str(inp.data_yaml_path)
    assert call_kwargs["epochs"] == 5
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["batch"] == 4
    assert call_kwargs["optimizer"] == "auto"
    assert call_kwargs["patience"] == 3
    assert "test_proj" in call_kwargs["project"]
    assert call_kwargs["name"] == "test_run"


# ── 3. best.pt collected ──────────────────────────────────────────────────────


def test_best_pt_copied_to_artifacts_dir(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert (tmp_path / "artifacts" / "best.pt").exists()
    names = [a.name for a in result.training_artifacts]
    assert "best.pt" in names


# ── 4. last.pt collected ──────────────────────────────────────────────────────


def test_last_pt_copied_to_artifacts_dir(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert (tmp_path / "artifacts" / "last.pt").exists()
    names = [a.name for a in result.training_artifacts]
    assert "last.pt" in names


# ── 5. results.csv collected ──────────────────────────────────────────────────


def test_results_csv_copied_to_artifacts_dir(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert (tmp_path / "artifacts" / "results.csv").exists()
    names = [a.name for a in result.training_artifacts]
    assert "results.csv" in names


# ── 6. Missing best.pt — graceful handling ────────────────────────────────────


def test_missing_best_pt_does_not_crash(tmp_path, fake_save_dir):
    (fake_save_dir / "weights" / "best.pt").unlink()
    mock_model = MagicMock()
    mock_model.trainer.save_dir = str(fake_save_dir)
    mock_model.train.return_value = {}
    MockYOLO = MagicMock(return_value=mock_model)

    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.success is True
    assert result.best_weights_path is None
    names = [a.name for a in result.training_artifacts]
    assert "best.pt" not in names
    assert "last.pt" in names


# ── 7. Missing results.csv — graceful handling ────────────────────────────────


def test_missing_results_csv_does_not_crash(tmp_path, fake_save_dir):
    (fake_save_dir / "results.csv").unlink()
    mock_model = MagicMock()
    mock_model.trainer.save_dir = str(fake_save_dir)
    mock_model.train.return_value = {}
    MockYOLO = MagicMock(return_value=mock_model)

    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.success is True
    names = [a.name for a in result.training_artifacts]
    assert "results.csv" not in names


# ── 8-9. training_output.json written and valid ───────────────────────────────


def test_training_output_json_written(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    artifacts_dir = tmp_path / "artifacts"
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        LocalYOLOTrainingRunner().run(_make_input(tmp_path), artifacts_dir)

    json_path = artifacts_dir / "training_output.json"
    assert json_path.exists(), "training_output.json must be written"


def test_training_output_json_contains_required_fields(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    artifacts_dir = tmp_path / "artifacts"
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), artifacts_dir)

    data = json.loads((artifacts_dir / "training_output.json").read_text(encoding="utf-8"))
    assert data["job_id"] == result.job_id
    assert data["mode"] == "local_train"
    assert data["runner"] == "local-yolo"
    assert data["job_status"] == "completed"


# ── 10-12. Output contract correctness ───────────────────────────────────────


def test_best_weights_path_set_in_output(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.best_weights_path is not None
    assert result.best_weights_path.endswith("best.pt")


def test_job_status_completed_on_success(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.job_status == TrainingJobStatus.COMPLETED
    assert result.success is True


def test_mode_is_local_train(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.mode == TrainingMode.LOCAL_TRAIN


# ── 13. Training failure path ─────────────────────────────────────────────────


def test_training_failure_returns_failed_status(tmp_path):
    mock_model = MagicMock()
    mock_model.train.side_effect = RuntimeError("GPU OOM")
    MockYOLO = MagicMock(return_value=mock_model)

    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    assert result.success is False
    assert result.job_status == TrainingJobStatus.FAILED
    assert result.mode == TrainingMode.LOCAL_TRAIN
    assert any("GPU OOM" in e for e in result.errors)


# ── 14. Azure ML / MLflow not used ────────────────────────────────────────────


def test_azure_ml_not_called(tmp_path, mock_yolo_cls, fake_save_dir):
    """LocalYOLOTrainingRunner has no azure/mlflow clients — Azure cannot be called."""
    MockYOLO, _ = mock_yolo_cls
    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    # No azure_client attribute — the runner doesn't accept one
    assert not hasattr(LocalYOLOTrainingRunner(), "_azure")
    assert result.mlflow_run_id is None


# ── 15. YoloTrainer delegates LOCAL_TRAIN to LocalYOLOTrainingRunner ──────────


def test_yolo_trainer_local_train_uses_runner(tmp_path, mock_yolo_cls, fake_save_dir):
    MockYOLO, _ = mock_yolo_cls
    trainer = YoloTrainer()
    inp = _make_input(tmp_path, mode=TrainingMode.LOCAL_TRAIN)
    artifacts_dir = tmp_path / "artifacts"

    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        result = trainer.run(inp, artifacts_dir)

    assert result.mode == TrainingMode.LOCAL_TRAIN
    assert result.job_status == TrainingJobStatus.COMPLETED


# ── 16-18. CLI --runner flag ──────────────────────────────────────────────────


@pytest.fixture
def cli_dataset(tmp_path: Path):
    """Minimal dataset + config YAML for CLI tests."""
    ds = tmp_path / "ds"
    data_yaml = ds / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("names:\n  0: scratch\n", encoding="utf-8")

    cfg_path = tmp_path / "training.yaml"
    cfg_path.write_text(
        "model: yolo11m.pt\nepochs: 1\nimgsz: 320\nbatch: 2\nmode: local_dry_run\n",
        encoding="utf-8",
    )
    return ds, data_yaml, cfg_path


def test_cli_runner_fake_produces_dry_run(tmp_path, cli_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, cfg_path = cli_dataset
    out_dir = tmp_path / "out"
    r = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-path",
            str(ds),
            "--data-yaml",
            str(data_yaml),
            "--training-config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--runner",
            "fake",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out_dir / "training_request.json").exists()


def test_cli_runner_local_yolo_sets_local_train_mode(
    tmp_path, cli_dataset, mock_yolo_cls, fake_save_dir
):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    MockYOLO, _ = mock_yolo_cls
    ds, data_yaml, cfg_path = cli_dataset
    out_dir = tmp_path / "out"

    with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
        r = CliRunner().invoke(
            app,
            [
                "train",
                "--dataset-path",
                str(ds),
                "--data-yaml",
                str(data_yaml),
                "--training-config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
                "--runner",
                "local-yolo",
            ],
        )

    assert r.exit_code == 0, r.output
    assert (out_dir / "training_output.json").exists()
    data = json.loads((out_dir / "training_output.json").read_text())
    assert data["mode"] == "local_train"
    assert data["runner"] == "local-yolo"


def test_ultralytics_mlflow_autolog_disabled_during_training(tmp_path, fake_save_dir):
    """_suppress_ultralytics_mlflow must call settings.update({'mlflow': False}) before
    training and restore the previous value in a finally block."""
    from unittest.mock import MagicMock, call, patch

    mock_settings = MagicMock()
    mock_settings.get.return_value = True

    mock_ult = MagicMock()
    mock_ult.settings = mock_settings

    mock_model = MagicMock()
    mock_model.trainer.save_dir = str(fake_save_dir)
    mock_model.train.return_value = {}
    MockYOLO = MagicMock(return_value=mock_model)

    with patch.dict("sys.modules", {"ultralytics": mock_ult}):
        with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=MockYOLO):
            LocalYOLOTrainingRunner().run(_make_input(tmp_path), tmp_path / "artifacts")

    update_calls = mock_settings.update.call_args_list
    assert call({"mlflow": False}) in update_calls, "Must disable MLflow before training"
    assert call({"mlflow": True}) in update_calls, "Must restore MLflow after training"


def test_cli_runner_unknown_exits_nonzero(tmp_path, cli_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, cfg_path = cli_dataset
    r = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-path",
            str(ds),
            "--data-yaml",
            str(data_yaml),
            "--training-config",
            str(cfg_path),
            "--runner",
            "azure",
        ],
    )
    assert r.exit_code != 0
