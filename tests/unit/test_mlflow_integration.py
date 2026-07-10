"""Tests for MLflow tracking integration.

Coverage:
  1.  NoOpMLflowTrackingClient — silent no-op, no mlflow import required
  2.  LocalMLflowTrackingClient — raises RuntimeError when mlflow missing
  3.  FakeMLflowTrackingClient — records params/metrics/artifacts/tags in memory
  4.  MLflowConfig — from_yaml, disabled() factory, defaults
  5.  DatasetValidationAgent — logs expected params/metrics/tags/artifacts when enabled
  6.  TrainingAgent — logs expected params/tags/artifacts when enabled
  7.  EvaluationAgent — logs expected metrics/params/tags/artifacts when enabled
  8.  HumanApprovalAgent — logs approval decision when enabled
  9.  MVPWorkflow — creates parent run, logs workflow tags, ends run on success/failure
  10. MLflow disabled by default — no mlflow package required for normal unit tests
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agentic_mlops.agents.dataset_validation import DatasetValidationAgent
from agentic_mlops.agents.evaluation import EvaluationAgent
from agentic_mlops.agents.human_approval import HumanApprovalAgent
from agentic_mlops.agents.training import TrainingAgent
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalInput
from agentic_mlops.contracts.datasets import DatasetValidationInput
from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMetrics, EvaluationMode
from agentic_mlops.contracts.mlflow_config import MLflowConfig
from agentic_mlops.contracts.training import TrainingConfig, TrainingInput, TrainingMode
from agentic_mlops.integrations.mlflow_client import (
    FakeMLflowTrackingClient,
    LocalMLflowTrackingClient,
    MLflowClientBase,
    MLflowTrackingClientBase,
    NoOpMLflowTrackingClient,
)
from tests.conftest import make_valid_dataset

# ── shared helpers ─────────────────────────────────────────────────────────────


def _fake_client() -> FakeMLflowTrackingClient:
    return FakeMLflowTrackingClient()


def _make_ds(tmp_path: Path) -> Path:
    """Create a valid YOLO dataset under tmp_path/ds and return the dir."""
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(ds_dir)
    return ds_dir


def _make_training_cfg(
    tmp_path: Path,
    mode: TrainingMode = TrainingMode.LOCAL_DRY_RUN,
) -> TrainingConfig:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        yaml.dump({"model": "yolo11n.pt", "epochs": 1, "mode": str(mode)}), encoding="utf-8"
    )
    return TrainingConfig.from_yaml(str(cfg_path))


def _make_eval_report(path: Path, recommendation: str = "promote_candidate") -> None:
    import json  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommendation": recommendation,
        "metrics": {"map50": 0.86, "map50_95": 0.59, "precision": 0.84, "recall": 0.79},
        "failed_checks": [],
        "passed_checks": ["map50 0.8600 >= 0.7500"],
        "message": "Evaluation passed.",
        "mode": "local_dry_run",
        "runner": "fake",
        "model_path": "dry_run",
        "started_at": None,
        "completed_at": None,
        "reasons": [],
        "artifacts": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ── 1. NoOpMLflowTrackingClient ────────────────────────────────────────────────


class TestNoOpMLflowTrackingClient:
    def test_is_subclass(self) -> None:
        assert issubclass(NoOpMLflowTrackingClient, MLflowTrackingClientBase)

    def test_start_run_returns_string(self) -> None:
        client = NoOpMLflowTrackingClient()
        run_id = client.start_run("exp", "run-1")
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_all_methods_return_none(self) -> None:
        client = NoOpMLflowTrackingClient()
        run_id = client.start_run("exp", "run-1")
        assert client.log_params(run_id, {"k": "v"}) is None
        assert client.log_metrics(run_id, {"m": 1.0}) is None
        assert client.log_artifact(run_id, "/some/file.json") is None
        assert client.log_tags(run_id, {"t": "v"}) is None
        assert client.end_run(run_id) is None

    def test_does_not_import_mlflow(self) -> None:
        """NoOp must work even when mlflow is not installed."""
        real_import = builtins.__import__

        def _block_mlflow(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("mlflow not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_mlflow):
            client = NoOpMLflowTrackingClient()
            run_id = client.start_run("exp", "run")
            client.log_params(run_id, {"a": "b"})


# ── 2. LocalMLflowTrackingClient ───────────────────────────────────────────────


class TestLocalMLflowTrackingClient:
    def test_raises_runtime_error_when_mlflow_missing(self) -> None:
        real_import = builtins.__import__

        def _block_mlflow(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("mlflow not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_mlflow):
            # Remove cached mlflow from sys.modules if present
            cached = sys.modules.pop("mlflow", None)
            try:
                with pytest.raises(RuntimeError, match="mlflow is not installed"):
                    LocalMLflowTrackingClient(tracking_uri="file:./test_mlruns")
            finally:
                if cached is not None:
                    sys.modules["mlflow"] = cached

    def test_is_subclass(self) -> None:
        assert issubclass(LocalMLflowTrackingClient, MLflowTrackingClientBase)


# ── 3. FakeMLflowTrackingClient ────────────────────────────────────────────────


class TestFakeMLflowTrackingClient:
    def test_is_subclass(self) -> None:
        assert issubclass(FakeMLflowTrackingClient, MLflowTrackingClientBase)

    def test_start_run_records_experiment_and_name(self) -> None:
        client = _fake_client()
        run_id = client.start_run("my-exp", "my-run")
        assert run_id in client.runs
        assert client.runs[run_id]["experiment"] == "my-exp"
        assert client.runs[run_id]["name"] == "my-run"
        assert client.runs[run_id]["status"] == "running"

    def test_log_params_recorded(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.log_params(run_id, {"lr": "0.01", "batch": "32"})
        assert client.runs[run_id]["params"]["lr"] == "0.01"
        assert client.runs[run_id]["params"]["batch"] == "32"

    def test_log_metrics_recorded(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.log_metrics(run_id, {"map50": 0.85, "recall": 0.9})
        assert client.runs[run_id]["metrics"]["map50"] == pytest.approx(0.85)
        assert client.runs[run_id]["metrics"]["recall"] == pytest.approx(0.9)

    def test_log_artifact_recorded(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.log_artifact(run_id, "/path/to/report.json")
        assert "/path/to/report.json" in client.runs[run_id]["artifacts"]

    def test_log_tags_recorded(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.log_tags(run_id, {"workflow_step": "training", "status": "ok"})
        assert client.runs[run_id]["tags"]["workflow_step"] == "training"

    def test_end_run_sets_status(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.end_run(run_id, status="FINISHED")
        assert client.runs[run_id]["status"] == "finished"

    def test_end_run_failed_status(self) -> None:
        client = _fake_client()
        run_id = client.start_run("e", "r")
        client.end_run(run_id, status="FAILED")
        assert client.runs[run_id]["status"] == "failed"

    def test_multiple_runs_independent(self) -> None:
        client = _fake_client()
        rid1 = client.start_run("e", "run-1")
        rid2 = client.start_run("e", "run-2")
        client.log_params(rid1, {"a": "1"})
        client.log_params(rid2, {"b": "2"})
        assert "a" not in client.runs[rid2]["params"]
        assert "b" not in client.runs[rid1]["params"]

    def test_backwards_compat_alias(self) -> None:
        from agentic_mlops.integrations.mlflow_client import FakeMLflowClient

        assert FakeMLflowClient is FakeMLflowTrackingClient

    def test_base_alias(self) -> None:
        assert MLflowClientBase is MLflowTrackingClientBase


# ── 4. MLflowConfig ────────────────────────────────────────────────────────────


class TestMLflowConfig:
    def test_disabled_factory(self) -> None:
        cfg = MLflowConfig.disabled()
        assert cfg.enabled is False

    def test_defaults(self) -> None:
        cfg = MLflowConfig()
        assert cfg.enabled is False
        assert cfg.tracking_uri == "sqlite:///outputs/mlruns.db"
        assert cfg.experiment_name == "agentic-mlops-local"
        assert cfg.run_name_prefix == "mvp-local-yolo"
        assert cfg.log_artifacts is True
        assert cfg.log_reports is True

    def test_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mlflow.yaml"
        cfg_file.write_text(
            "enabled: true\n"
            "tracking_uri: file:./test_mlruns\n"
            "experiment_name: test-exp\n"
            "run_name_prefix: test-run\n"
            "log_artifacts: false\n"
            "log_reports: true\n",
            encoding="utf-8",
        )
        cfg = MLflowConfig.from_yaml(str(cfg_file))
        assert cfg.enabled is True
        assert cfg.tracking_uri == "file:./test_mlruns"
        assert cfg.experiment_name == "test-exp"
        assert cfg.log_artifacts is False

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        """Unspecified fields fall back to defaults."""
        cfg_file = tmp_path / "mlflow.yaml"
        cfg_file.write_text("enabled: true\n", encoding="utf-8")
        cfg = MLflowConfig.from_yaml(str(cfg_file))
        assert cfg.enabled is True
        assert cfg.experiment_name == "agentic-mlops-local"


# ── 5. DatasetValidationAgent MLflow logging ──────────────────────────────────


class TestDatasetValidationAgentMLflow:
    def test_logs_params_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = DatasetValidationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            DatasetValidationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            )
        )

        params = client.runs[run_id]["params"]
        assert "validation.dataset_path" in params
        assert "validation.data_yaml" in params
        assert "validation.num_classes" in params

    def test_logs_metrics_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = DatasetValidationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            DatasetValidationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            )
        )

        metrics = client.runs[run_id]["metrics"]
        assert "validation.total_images" in metrics
        assert "validation.total_label_files" in metrics
        assert "validation.blocking_issue_count" in metrics
        assert "validation.warning_count" in metrics
        assert metrics["validation.total_images"] == pytest.approx(6.0)

    def test_logs_tags_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = DatasetValidationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            DatasetValidationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            )
        )

        tags = client.runs[run_id]["tags"]
        assert tags.get("workflow_step") == "dataset_validation"
        assert "validation_status" in tags

    def test_logs_artifacts_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = DatasetValidationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            DatasetValidationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            )
        )

        artifacts = client.runs[run_id]["artifacts"]
        assert any("dataset_quality_report.json" in a for a in artifacts)
        assert any("dataset_quality_report.md" in a for a in artifacts)

    def test_no_mlflow_call_when_client_is_none(self, tmp_path: Path) -> None:
        """Agent must work normally without an MLflow client."""
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        agent = DatasetValidationAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(
            DatasetValidationInput(dataset_path=str(tmp_path / "ds"))
        )
        assert result.success is True

    def test_no_mlflow_call_when_run_id_is_none(self, tmp_path: Path) -> None:
        """Client present but run_id absent — no logging should happen."""
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        agent = DatasetValidationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=None,
        )
        agent.run(DatasetValidationInput(dataset_path=str(tmp_path / "ds")))
        assert client.runs == {}


# ── 6. TrainingAgent MLflow logging ───────────────────────────────────────────


class TestTrainingAgentMLflow:
    def test_logs_params_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")
        cfg = _make_training_cfg(tmp_path)

        agent = TrainingAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            TrainingInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                training_config=cfg,
                dataset_validation_status="passed",
            )
        )

        params = client.runs[run_id]["params"]
        assert "training.model" in params
        assert "training.epochs" in params
        assert "training.runner" in params
        assert params["training.runner"] == "fake"
        assert "training.data_yaml" in params

    def test_logs_tags_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")
        cfg = _make_training_cfg(tmp_path)

        agent = TrainingAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            TrainingInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                training_config=cfg,
                dataset_validation_status="passed",
            )
        )

        tags = client.runs[run_id]["tags"]
        assert tags.get("workflow_step") == "training"
        assert tags.get("training_runner") == "fake"
        assert "training_status" in tags

    def test_logs_artifacts_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")
        cfg = _make_training_cfg(tmp_path)

        agent = TrainingAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            TrainingInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                training_config=cfg,
                dataset_validation_status="passed",
            )
        )

        assert len(client.runs[run_id]["artifacts"]) > 0

    def test_no_logging_when_training_blocked(self, tmp_path: Path) -> None:
        """When dataset validation failed, training is blocked — no MLflow logging."""
        client = _fake_client()
        run_id = client.start_run("exp", "run")
        cfg = _make_training_cfg(tmp_path)

        agent = TrainingAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            TrainingInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                training_config=cfg,
                dataset_validation_status="failed",
            )
        )
        # Early return before _log_to_mlflow — no params/tags should be written
        assert client.runs[run_id]["params"] == {}


# ── 7. EvaluationAgent MLflow logging ─────────────────────────────────────────


class TestEvaluationAgentMLflow:
    def test_logs_metrics_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")
        override = EvaluationMetrics(map50=0.86, map50_95=0.59, precision=0.84, recall=0.79)

        agent = EvaluationAgent(
            artifacts_dir=tmp_path / "artifacts",
            dry_run_override_metrics=override,
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            EvaluationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                weights_path="dry_run",
                mode=EvaluationMode.LOCAL_DRY_RUN,
                training_status="completed",
            )
        )

        metrics = client.runs[run_id]["metrics"]
        assert "evaluation.map50" in metrics
        assert "evaluation.map50_95" in metrics
        assert "evaluation.precision" in metrics
        assert "evaluation.recall" in metrics
        assert metrics["evaluation.map50"] == pytest.approx(0.86)

    def test_logs_params_and_tags_when_mlflow_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
        make_valid_dataset(tmp_path / "ds")
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = EvaluationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            EvaluationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                weights_path="dry_run",
                mode=EvaluationMode.LOCAL_DRY_RUN,
                training_status="completed",
            )
        )

        params = client.runs[run_id]["params"]
        tags = client.runs[run_id]["tags"]
        assert "evaluation.model_path" in params
        assert tags.get("workflow_step") == "evaluation"
        assert "recommendation" in tags

    def test_no_logging_when_training_failed(self, tmp_path: Path) -> None:
        """Blocked evaluation skips MLflow logging (no metrics to report)."""
        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = EvaluationAgent(
            artifacts_dir=tmp_path / "artifacts",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            EvaluationInput(
                dataset_path=str(tmp_path / "ds"),
                data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
                weights_path="dry_run",
                mode=EvaluationMode.LOCAL_DRY_RUN,
                training_status="failed",
            )
        )
        assert client.runs[run_id]["metrics"] == {}


# ── 8. HumanApprovalAgent MLflow logging ──────────────────────────────────────


class TestHumanApprovalAgentMLflow:
    def test_logs_approval_decision_when_mlflow_enabled(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "evaluation" / "evaluation_report.json"
        _make_eval_report(eval_json)

        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = HumanApprovalAgent(
            artifacts_dir=tmp_path / "approval",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            ApprovalInput(
                evaluation_output_path=str(eval_json),
                output_dir=str(tmp_path / "approval"),
                interactive=False,
                action=ApprovalAction.APPROVE_MODEL,
                approver="CI",
            )
        )

        params = client.runs[run_id]["params"]
        metrics = client.runs[run_id]["metrics"]
        tags = client.runs[run_id]["tags"]

        assert "approval.interactive" in params
        assert "approval.action" in params
        assert "approval.approved" in metrics
        assert metrics["approval.approved"] == pytest.approx(1.0)
        assert tags.get("workflow_step") == "human_approval"
        assert "approval_status" in tags

    def test_logs_reject_decision(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "evaluation" / "evaluation_report.json"
        _make_eval_report(eval_json)

        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = HumanApprovalAgent(
            artifacts_dir=tmp_path / "approval",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            ApprovalInput(
                evaluation_output_path=str(eval_json),
                output_dir=str(tmp_path / "approval"),
                interactive=False,
                action=ApprovalAction.REJECT_MODEL,
                approver="CI",
            )
        )

        metrics = client.runs[run_id]["metrics"]
        assert metrics["approval.approved"] == pytest.approx(0.0)

    def test_logs_artifacts_for_approval(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "evaluation" / "evaluation_report.json"
        _make_eval_report(eval_json)

        client = _fake_client()
        run_id = client.start_run("exp", "run")

        agent = HumanApprovalAgent(
            artifacts_dir=tmp_path / "approval",
            mlflow_client=client,
            mlflow_run_id=run_id,
        )
        agent.run(
            ApprovalInput(
                evaluation_output_path=str(eval_json),
                output_dir=str(tmp_path / "approval"),
                interactive=False,
                action=ApprovalAction.APPROVE_MODEL,
                approver="CI",
            )
        )

        artifacts = client.runs[run_id]["artifacts"]
        assert any("approval_decision" in a for a in artifacts)


# ── 9. MVPWorkflow MLflow parent run management ────────────────────────────────


from agentic_mlops.agents.base import BaseAgent  # noqa: E402
from agentic_mlops.contracts.approvals import ApprovalOutput, ApprovalStatus  # noqa: E402
from agentic_mlops.contracts.datasets import DatasetValidationOutput  # noqa: E402
from agentic_mlops.contracts.evaluation import EvaluationOutput  # noqa: E402
from agentic_mlops.contracts.training import TrainingJobStatus, TrainingOutput  # noqa: E402
from agentic_mlops.contracts.workflows import MVPWorkflowInput  # noqa: E402
from agentic_mlops.workflows.mvp_workflow import MVPWorkflow  # noqa: E402


class _StubAgent(BaseAgent):
    def __init__(self, artifacts_dir: Path, result: object) -> None:
        super().__init__(artifacts_dir)
        self._result = result

    def run(self, input: object) -> object:  # noqa: A002
        return self._result


def _stub_factories(
    val_result: DatasetValidationOutput,
    train_result: TrainingOutput,
    eval_result: EvaluationOutput,
    approval_result: ApprovalOutput,
) -> dict:
    return {
        "_validation_factory": lambda d: _StubAgent(d, val_result),
        "_training_factory": lambda d: _StubAgent(d, train_result),
        "_evaluation_factory": lambda d: _StubAgent(d, eval_result),
        "_approval_factory": lambda d: _StubAgent(d, approval_result),
    }


def _default_results() -> tuple:
    val = DatasetValidationOutput(
        success=True, message="ok", status="passed", num_images=6, num_labels=6
    )
    train = TrainingOutput(success=True, message="trained", job_status=TrainingJobStatus.COMPLETED)
    evl = EvaluationOutput(
        success=True,
        message="eval ok",
        metrics=EvaluationMetrics(map50=0.86, map50_95=0.59, precision=0.84, recall=0.79),
        recommendation="promote_candidate",
    )
    approval = ApprovalOutput(
        success=True,
        message="approved",
        status=ApprovalStatus.APPROVED,
        action=ApprovalAction.APPROVE_MODEL,
        generated_artifacts=[],
    )
    return val, train, evl, approval


def _make_mvp_input(tmp_path: Path) -> MVPWorkflowInput:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.dump({"model": "yolo11n.pt", "epochs": 1}), encoding="utf-8")
    return MVPWorkflowInput(
        dataset_path=str(tmp_path / "ds"),
        data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
        training_config_path=str(cfg_path),
        output_dir=str(tmp_path / "out"),
        interactive_approval=False,
        approval_action=ApprovalAction.APPROVE_MODEL,
        approver="CI",
        dry_run=True,
    )


class TestMVPWorkflowMLflow:
    def test_creates_parent_run_when_enabled(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True, experiment_name="test-exp", run_name_prefix="test")
        val, train, evl, approval = _default_results()

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        assert result.mlflow_run_id is not None
        assert result.mlflow_run_id in client.runs
        assert result.mlflow_experiment_name == "test-exp"

    def test_run_ends_with_finished_on_success(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True)
        val, train, evl, approval = _default_results()

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        run = client.runs[result.mlflow_run_id]
        assert run["status"] == "finished"

    def test_run_ends_with_failed_when_workflow_fails(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True)
        val_fail = DatasetValidationOutput(
            success=False, message="bad", status="failed", num_images=0, num_labels=0
        )
        train = TrainingOutput(success=True, message="ok", job_status=TrainingJobStatus.COMPLETED)
        evl = EvaluationOutput(success=True, message="ok", metrics=EvaluationMetrics())
        approval = ApprovalOutput(success=True, message="ok", generated_artifacts=[])

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val_fail, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        assert result.success is False
        run_id = result.mlflow_run_id
        assert run_id is not None
        assert client.runs[run_id]["status"] == "failed"

    def test_logs_workflow_status_tag(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True)
        val, train, evl, approval = _default_results()

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        tags = client.runs[result.mlflow_run_id]["tags"]
        assert "workflow_status" in tags
        assert "workflow_name" in tags

    def test_mlflow_disabled_by_default(self, tmp_path: Path) -> None:
        """No MLflow calls when config is not passed."""
        val, train, evl, approval = _default_results()
        workflow = MVPWorkflow(**_stub_factories(val, train, evl, approval))
        result = workflow.run(_make_mvp_input(tmp_path))

        assert result.mlflow_run_id is None
        assert result.success is True

    def test_mlflow_tracking_uri_propagated(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True, tracking_uri="file:./custom_mlruns")
        val, train, evl, approval = _default_results()

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        assert result.mlflow_tracking_uri == "file:./custom_mlruns"

    def test_workflow_summary_md_contains_mlflow_section(self, tmp_path: Path) -> None:
        client = _fake_client()
        config = MLflowConfig(enabled=True, experiment_name="demo-exp")
        val, train, evl, approval = _default_results()

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            **_stub_factories(val, train, evl, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))

        md_path = Path(result.workflow_summary_md_path)
        content = md_path.read_text(encoding="utf-8")
        assert "MLflow Tracking" in content
        assert result.mlflow_run_id in content

    def test_workflow_summary_md_shows_disabled_when_no_mlflow(self, tmp_path: Path) -> None:
        val, train, evl, approval = _default_results()
        workflow = MVPWorkflow(**_stub_factories(val, train, evl, approval))
        result = workflow.run(_make_mvp_input(tmp_path))

        md_path = Path(result.workflow_summary_md_path)
        content = md_path.read_text(encoding="utf-8")
        assert "MLflow: disabled" in content

    def test_user_injected_factories_bypass_mlflow_defaults(self, tmp_path: Path) -> None:
        """User-supplied factories must not have MLflow injected into them."""
        client = _fake_client()
        config = MLflowConfig(enabled=True)
        val, train, evl, approval = _default_results()
        factories_called_with_mlflow_args: list[bool] = []

        def _custom_val_factory(d: Path) -> _StubAgent:
            factories_called_with_mlflow_args.append(True)
            return _StubAgent(d, val)

        workflow = MVPWorkflow(
            mlflow_client=client,
            mlflow_config=config,
            _validation_factory=_custom_val_factory,
            _training_factory=lambda d: _StubAgent(d, train),
            _evaluation_factory=lambda d: _StubAgent(d, evl),
            _approval_factory=lambda d: _StubAgent(d, approval),
        )
        result = workflow.run(_make_mvp_input(tmp_path))
        assert result.success is True
        assert len(factories_called_with_mlflow_args) == 1


# ── 10. Workflow without mlflow package ────────────────────────────────────────


class TestNoMLflowPackageRequired:
    def test_noop_client_works_without_mlflow_installed(self) -> None:
        """Core pipeline functionality must not require the mlflow package."""
        client = NoOpMLflowTrackingClient()
        run_id = client.start_run("exp", "run")
        client.log_params(run_id, {"a": "b"})
        client.log_metrics(run_id, {"m": 1.0})
        client.log_artifact(run_id, "/path")
        client.log_tags(run_id, {"t": "v"})
        client.end_run(run_id)

    def test_mlflow_config_import_does_not_require_mlflow(self) -> None:
        """MLflowConfig is a pure Pydantic model — no mlflow dependency."""
        cfg = MLflowConfig(enabled=False)
        assert not cfg.enabled
