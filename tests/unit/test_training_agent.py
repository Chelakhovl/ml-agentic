"""Unit tests for TrainingAgent.

Coverage matrix:
    1. Training is blocked when dataset_validation_status == 'failed'
    2. Dry-run creates training_request.json with correct structure
    3. Dry-run creates training_report.md
    4. TrainingConfig loads correctly from a YAML file
    5. TrainingAgent returns a structured TrainingOutput in dry-run
    6. FakeAzureMLTrainingClient is NOT called in dry-run mode
    7. Training is NOT blocked when validation status is 'passed'
    8. Training is NOT blocked when validation status is 'warning'
    9. Training is NOT blocked when validation status is None (not provided)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agentic_mlops.agents.training import TrainingAgent
from agentic_mlops.contracts.training import (
    TrainingConfig,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
)
from agentic_mlops.integrations.azure_ml_client import FakeAzureMLTrainingClient
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_azure() -> FakeAzureMLTrainingClient:
    return FakeAzureMLTrainingClient()


@pytest.fixture
def fake_mlflow() -> FakeMLflowClient:
    return FakeMLflowClient()


def _make_agent(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> TrainingAgent:
    return TrainingAgent(
        artifacts_dir=tmp_path / "artifacts",
        azure_client=fake_azure,
        mlflow_client=fake_mlflow,
    )


def _dry_run_input(
    tmp_path: Path,
    validation_status: str | None = "passed",
) -> TrainingInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text("names:\n  0: scratch\n", encoding="utf-8")

    cfg = TrainingConfig(
        model="yolo11m.pt",
        epochs=10,
        imgsz=640,
        batch=8,
        project="test_project",
        name="test_run",
        mode=TrainingMode.LOCAL_DRY_RUN,
    )
    return TrainingInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        training_config=cfg,
        workflow_id="wf_test_001",
        dataset_validation_status=validation_status,  # type: ignore[arg-type]
    )


# ── 1. Blocked on failed validation ───────────────────────────────────────────


def test_training_blocked_on_failed_validation(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path, validation_status="failed")

    result = agent.run(inp)

    assert result.success is False
    assert result.job_status == TrainingJobStatus.CANCELLED
    assert "blocked" in result.message.lower() or "failed" in result.message.lower()
    # Azure client must not have been called
    assert fake_azure.submitted_jobs == []


# ── 2. Dry-run creates training_request.json ──────────────────────────────────


def test_dry_run_creates_training_request_json(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path)

    result = agent.run(inp)

    assert result.success is True
    assert result.training_plan_path is not None
    plan_path = Path(result.training_plan_path)
    assert plan_path.exists(), "training_request.json must be written to disk"

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["mode"] == "local_dry_run"
    assert plan["workflow_id"] == "wf_test_001"
    assert "would_run_command" in plan
    assert "yolo train" in plan["would_run_command"]
    assert "training_config" in plan


# ── 3. Dry-run creates training_report.md ─────────────────────────────────────


def test_dry_run_creates_training_report_md(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path)

    result = agent.run(inp)

    assert result.report_path is not None
    report_path = Path(result.report_path)
    assert report_path.exists(), "training_report.md must be written to disk"
    content = report_path.read_text(encoding="utf-8")
    assert "# Training Report" in content
    assert "fake" in content


# ── 4. TrainingConfig loads from YAML ─────────────────────────────────────────


def test_training_config_loads_from_yaml(tmp_path: Path) -> None:
    config_data = {
        "model": "yolo11s.pt",
        "epochs": 50,
        "imgsz": 416,
        "batch": 32,
        "optimizer": "SGD",
        "patience": 10,
        "project": "my_project",
        "name": "run_v2",
        "mode": "local_dry_run",
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")

    cfg = TrainingConfig.from_yaml(config_path)

    assert cfg.model == "yolo11s.pt"
    assert cfg.epochs == 50
    assert cfg.imgsz == 416
    assert cfg.batch == 32
    assert cfg.optimizer == "SGD"
    assert cfg.mode == TrainingMode.LOCAL_DRY_RUN
    assert cfg.project == "my_project"


# ── 5. Structured TrainingOutput in dry-run ───────────────────────────────────


def test_training_agent_returns_structured_output(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path)

    result = agent.run(inp)

    # Pydantic model fields
    assert result.success is True
    assert result.job_id is not None
    assert result.job_id.startswith("dry_run_")
    assert result.job_status == TrainingJobStatus.COMPLETED
    assert result.mode == TrainingMode.LOCAL_DRY_RUN
    assert len(result.training_artifacts) == 1
    assert result.training_artifacts[0].artifact_type == "plan"
    # Report path must be listed in artifacts list
    assert result.report_path in result.artifacts


# ── 6. FakeAzureMLTrainingClient not called in dry-run ────────────────────────


def test_fake_azure_not_called_in_dry_run(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path)

    agent.run(inp)

    assert fake_azure.submitted_jobs == [], (
        "Azure ML client must not be called during local_dry_run"
    )


# ── 7-9. Training proceeds when validation is passed / warning / None ─────────


@pytest.mark.parametrize("status", ["passed", "warning", None])
def test_training_proceeds_on_non_failed_status(
    tmp_path: Path,
    fake_azure: FakeAzureMLTrainingClient,
    fake_mlflow: FakeMLflowClient,
    status: str | None,
) -> None:
    agent = _make_agent(tmp_path, fake_azure, fake_mlflow)
    inp = _dry_run_input(tmp_path, validation_status=status)

    result = agent.run(inp)

    assert result.success is True
    assert result.job_status == TrainingJobStatus.COMPLETED
