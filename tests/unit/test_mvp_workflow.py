"""Unit tests for MVPWorkflow end-to-end orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.cli.main import app
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalOutput, ApprovalStatus
from agentic_mlops.contracts.datasets import DatasetValidationOutput
from agentic_mlops.contracts.evaluation import EvaluationMetrics, EvaluationOutput
from agentic_mlops.contracts.training import TrainingJobStatus, TrainingOutput
from agentic_mlops.contracts.workflows import MVPWorkflowInput, MVPWorkflowStatus
from agentic_mlops.workflows.mvp_workflow import MVPWorkflow

# ── Stub agents ────────────────────────────────────────────────────────────────


class _StubAgent(BaseAgent):
    def __init__(self, artifacts_dir: Path, result: object) -> None:
        super().__init__(artifacts_dir)
        self._result = result

    def run(self, input: object) -> object:  # noqa: A002
        return self._result


class _RaisingAgent(BaseAgent):
    def __init__(self, artifacts_dir: Path, exc: Exception) -> None:
        super().__init__(artifacts_dir)
        self._exc = exc

    def run(self, input: object) -> object:  # noqa: A002
        raise self._exc


# ── Result factories ───────────────────────────────────────────────────────────


def _val_ok() -> DatasetValidationOutput:
    return DatasetValidationOutput(
        success=True,
        message="ok",
        status="passed",
        num_images=6,
        num_labels=6,
    )


def _val_fail() -> DatasetValidationOutput:
    return DatasetValidationOutput(
        success=False,
        message="bad dataset",
        status="failed",
        num_images=0,
        num_labels=0,
        errors=["Missing label files"],
    )


def _train_ok() -> TrainingOutput:
    return TrainingOutput(
        success=True,
        message="trained",
        job_status=TrainingJobStatus.COMPLETED,
    )


def _train_fail() -> TrainingOutput:
    return TrainingOutput(
        success=False,
        message="training exploded",
        job_status=TrainingJobStatus.FAILED,
        errors=["CUDA OOM"],
    )


def _eval_ok() -> EvaluationOutput:
    return EvaluationOutput(
        success=True,
        message="eval ok",
        metrics=EvaluationMetrics(map50=0.86, map50_95=0.59, precision=0.84, recall=0.79),
        recommendation="promote_candidate",
        passed_checks=["map50 0.8600 >= 0.7500"],
    )


def _eval_fail() -> EvaluationOutput:
    return EvaluationOutput(
        success=False,
        message="eval blocked",
        errors=["Upstream training failed"],
    )


def _approval_ok(action: ApprovalAction = ApprovalAction.APPROVE_MODEL) -> ApprovalOutput:
    return ApprovalOutput(
        success=True,
        message="approved",
        status=ApprovalStatus.APPROVED,
        action=action,
        approver="TestBot",
        generated_artifacts=[],
    )


def _approval_reject() -> ApprovalOutput:
    return ApprovalOutput(
        success=True,
        message="rejected",
        status=ApprovalStatus.REJECTED,
        action=ApprovalAction.REJECT_MODEL,
        approver="TestBot",
        generated_artifacts=[],
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_training_config(path: Path) -> Path:
    cfg = {"model": "yolo11m.pt", "epochs": 1, "mode": "local_dry_run"}
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _make_workflow(
    tmp_path: Path,
    *,
    val_result=None,
    train_result=None,
    eval_result=None,
    approval_result=None,
) -> tuple[MVPWorkflow, MVPWorkflowInput, Path]:
    cfg_path = _make_training_config(tmp_path / "train.yaml")
    out_dir = tmp_path / "out"
    workflow = MVPWorkflow(
        _validation_factory=lambda d: _StubAgent(d, val_result or _val_ok()),
        _training_factory=lambda d: _StubAgent(d, train_result or _train_ok()),
        _evaluation_factory=lambda d: _StubAgent(d, eval_result or _eval_ok()),
        _approval_factory=lambda d: _StubAgent(d, approval_result or _approval_ok()),
    )
    inp = MVPWorkflowInput(
        dataset_path=str(tmp_path / "ds"),
        data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
        training_config_path=str(cfg_path),
        output_dir=str(out_dir),
        interactive_approval=False,
        approval_action=ApprovalAction.APPROVE_MODEL,
        approver="TestBot",
        dry_run=True,
    )
    return workflow, inp, out_dir


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_happy_path_all_steps_succeed(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    result = workflow.run(inp)

    assert result.success is True
    assert result.workflow_status == MVPWorkflowStatus.COMPLETED
    assert len(result.steps) == 4
    assert all(s.success for s in result.steps)


def test_workflow_summary_json_created(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    result = workflow.run(inp)

    assert result.workflow_summary_path is not None
    json_file = Path(result.workflow_summary_path)
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["workflow_status"] == "completed"
    assert data["success"] is True
    assert len(data["steps"]) == 4


def test_workflow_summary_md_created(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    result = workflow.run(inp)

    assert result.workflow_summary_md_path is not None
    md_file = Path(result.workflow_summary_md_path)
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "MVP Workflow Summary" in content
    assert "COMPLETED" in content


def test_validation_failure_stops_workflow(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path, val_result=_val_fail())
    result = workflow.run(inp)

    assert result.success is False
    assert result.workflow_status == MVPWorkflowStatus.FAILED
    step_names = [s.step for s in result.steps]
    assert "validation" in step_names
    assert "training" not in step_names
    assert "evaluation" not in step_names
    assert "approval" not in step_names


def test_training_failure_stops_workflow(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path, train_result=_train_fail())
    result = workflow.run(inp)

    assert result.success is False
    step_names = [s.step for s in result.steps]
    assert "training" in step_names
    assert "evaluation" not in step_names
    assert "approval" not in step_names


def test_evaluation_failure_stops_workflow(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path, eval_result=_eval_fail())
    result = workflow.run(inp)

    assert result.success is False
    step_names = [s.step for s in result.steps]
    assert "evaluation" in step_names
    assert "approval" not in step_names


def test_approval_rejection_is_workflow_success(tmp_path: Path) -> None:
    """Rejecting a model is a valid human decision — workflow itself succeeds."""
    workflow, inp, out_dir = _make_workflow(tmp_path, approval_result=_approval_reject())
    result = workflow.run(inp)

    assert result.success is True
    approval_step = next(s for s in result.steps if s.step == "approval")
    assert approval_step.status == "rejected"


def test_exception_in_step_is_caught(tmp_path: Path) -> None:
    cfg_path = _make_training_config(tmp_path / "train.yaml")
    out_dir = tmp_path / "out"
    exc = RuntimeError("boom")
    workflow = MVPWorkflow(
        _validation_factory=lambda d: _RaisingAgent(d, exc),
        _training_factory=lambda d: _StubAgent(d, _train_ok()),
        _evaluation_factory=lambda d: _StubAgent(d, _eval_ok()),
        _approval_factory=lambda d: _StubAgent(d, _approval_ok()),
    )
    inp = MVPWorkflowInput(
        dataset_path=str(tmp_path / "ds"),
        data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
        training_config_path=str(cfg_path),
        output_dir=str(out_dir),
        interactive_approval=False,
        approval_action=ApprovalAction.APPROVE_MODEL,
        dry_run=True,
    )
    result = workflow.run(inp)

    assert result.success is False
    assert any("boom" in e for e in result.errors)
    assert result.workflow_summary_path is not None


def test_output_dir_structure_created(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    workflow.run(inp)

    assert (out_dir / "validation").exists()
    assert (out_dir / "training").exists()
    assert (out_dir / "evaluation").exists()
    assert (out_dir / "approval").exists()
    assert (out_dir / "workflow_summary.json").exists()
    assert (out_dir / "workflow_summary.md").exists()


# ── Azure ML wiring ────────────────────────────────────────────────────────────


def _make_azure_config_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "azure_ml.yaml"
    p.write_text(
        "subscription_id: sub\nresource_group: rg\nworkspace_name: ws\n"
        "compute_name: c\nenvironment:\n  mode: registered\n"
        "  registered_environment: azureml:e:1\n",
        encoding="utf-8",
    )
    return p


def test_azure_training_runner_missing_config_fails_fast(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    inp = inp.model_copy(update={"training_runner": "azure-ml"})

    result = workflow.run(inp)

    assert result.success is False
    assert "azure_config_path" in result.message
    assert result.steps == []  # fails before any step (incl. validation) runs


def test_azure_evaluation_runner_missing_config_fails_fast(tmp_path: Path) -> None:
    workflow, inp, out_dir = _make_workflow(tmp_path)
    inp = inp.model_copy(update={"evaluation_runner": "azure-ml"})

    result = workflow.run(inp)

    assert result.success is False
    assert "azure_config_path" in result.message
    assert result.steps == []


def test_azure_registry_backend_missing_config_fails_fast(tmp_path: Path) -> None:
    from agentic_mlops.contracts.model_registry import RegistryBackend

    workflow, inp, out_dir = _make_workflow(tmp_path)
    inp = inp.model_copy(update={"registry_backend": RegistryBackend.AZURE_ML})

    result = workflow.run(inp)

    assert result.success is False
    assert "azure_config_path" in result.message
    assert result.steps == []


def test_azure_training_runner_wired_into_default_training_agent(
    tmp_path: Path, monkeypatch
) -> None:
    """training_runner='azure-ml' + azure_config_path resolves a real
    AzureMLTrainingRunner and injects it into the *default* TrainingAgent factory
    (i.e. without overriding _training_factory) — proves the run()-level wiring,
    not AzureMLTrainingRunner's own job-submission behavior (covered elsewhere)."""
    azure_cfg_path = _make_azure_config_yaml(tmp_path)
    calls: list[Path] = []

    def fake_run(self, inp, artifacts_dir):  # noqa: ANN001
        calls.append(artifacts_dir)
        return _train_ok()

    monkeypatch.setattr("agentic_mlops.tools.training_runner.AzureMLTrainingRunner.run", fake_run)

    cfg_path = _make_training_config(tmp_path / "train.yaml")
    out_dir = tmp_path / "out"
    workflow = MVPWorkflow(
        _validation_factory=lambda d: _StubAgent(d, _val_ok()),
        _evaluation_factory=lambda d: _StubAgent(d, _eval_ok()),
        _approval_factory=lambda d: _StubAgent(d, _approval_ok()),
    )
    inp = MVPWorkflowInput(
        dataset_path=str(tmp_path / "ds"),
        data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
        training_config_path=str(cfg_path),
        output_dir=str(out_dir),
        interactive_approval=False,
        approval_action=ApprovalAction.APPROVE_MODEL,
        dry_run=False,
        training_runner="azure-ml",
        azure_config_path=str(azure_cfg_path),
    )

    result = workflow.run(inp)

    assert len(calls) == 1
    assert result.success is True
    training_step = next(s for s in result.steps if s.step == "training")
    assert training_step.success is True


def test_cli_run_mvp_dry_run(tmp_path: Path) -> None:
    """CLI run-mvp happy path using real agents in dry-run mode."""
    from tests.conftest import make_valid_dataset

    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    make_valid_dataset(ds_dir)
    data_yaml = ds_dir / "data.yaml"
    cfg_path = _make_training_config(tmp_path / "train.yaml")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-mvp",
            "--dataset-path",
            str(ds_dir),
            "--data-yaml",
            str(data_yaml),
            "--training-config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--approver",
            "CI",
            "--approval-action",
            "request_retraining",
            "--dry-run",
            "--no-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "workflow_summary.json").exists()
    data = json.loads((out_dir / "workflow_summary.json").read_text(encoding="utf-8"))
    assert data["workflow_status"] == "completed"
