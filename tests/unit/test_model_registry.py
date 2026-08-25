"""Unit tests for the Model Registry Agent, contracts, integrations, and workflow step."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.agents.model_registry import ModelRegistryAgent
from agentic_mlops.cli.main import app
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalOutput, ApprovalStatus
from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.contracts.datasets import DatasetValidationOutput
from agentic_mlops.contracts.evaluation import EvaluationMetrics, EvaluationOutput
from agentic_mlops.contracts.model_registry import (
    ModelLineage,
    ModelRegistrationInput,
    RegistrationStatus,
    RegistryBackend,
)
from agentic_mlops.contracts.training import TrainingJobStatus, TrainingOutput
from agentic_mlops.contracts.workflows import MVPWorkflowInput
from agentic_mlops.integrations.azure_ml_client import (
    DefaultAzureMLClientFactory,
    FakeAzureMLClientFactory,
    FakeMLClient,
)
from agentic_mlops.integrations.model_registry import (
    AzureMLModelRegistryClient,
    FakeModelRegistryClient,
    LocalModelRegistryClient,
    MLflowModelRegistryClient,
    create_registry_client,
)
from agentic_mlops.workflows.mvp_workflow import MVPWorkflow

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _write_training_output(dir_: Path, best_weights: Path | None = None) -> Path:
    data = {
        "success": True,
        "message": "trained",
        "job_id": "local_abcd1234",
        "job_status": "completed",
        "runner": "local-yolo",
        "best_weights_path": str(best_weights) if best_weights else None,
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
        "mode": "local_train",
        "mlflow_run_id": None,
        "training_plan_path": None,
        "training_artifacts": [],
    }
    p = dir_ / "training_output.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_evaluation_output(dir_: Path, success: bool = True) -> Path:
    data = {
        "success": success,
        "message": "eval done",
        "recommendation": "promote_candidate" if success else None,
        "metrics": {
            "map50": 0.85,
            "map50_95": 0.60,
            "precision": 0.80,
            "recall": 0.82,
        },
        "passed_checks": ["map50 0.85 >= 0.75"] if success else [],
        "failed_checks": [],
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
    }
    p = dir_ / "evaluation_output.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_approval_decision(
    dir_: Path,
    status: str = "approved",
    action: str = "approve_model",
) -> Path:
    data = {
        "success": True,
        "message": "approved",
        "status": status,
        "action": action,
        "approver": "TestBot",
        "timestamp": "2026-06-19T12:00:00+00:00",
        "comment": "LGTM",
        "approval_request": {
            "candidate_model": "best.pt",
            "dataset_version_or_path": "/tmp/ds",
            "evaluation_status": "promote_candidate",
            "recommendation": "promote_candidate",
            "key_metrics": {},
            "threshold_summary": [],
            "reasons": [],
            "next_actions": [],
        },
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
        "generated_artifacts": [],
        "evaluation_output_path": "",
    }
    p = dir_ / "approval_decision.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_fake_weights(dir_: Path, name: str = "best.pt") -> Path:
    p = dir_ / name
    p.write_bytes(b"fake-model-weights")
    return p


def _make_registration_input(
    tmp_path: Path,
    *,
    registry_dir: Path | None = None,
    approval_status: str = "approved",
    approval_action: str = "approve_model",
    eval_success: bool = True,
    training_job_status: str = "completed",
    with_weights: bool = True,
) -> tuple[ModelRegistrationInput, Path]:
    train_dir = tmp_path / "training"
    train_dir.mkdir(parents=True)
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True)
    approval_dir = tmp_path / "approval"
    approval_dir.mkdir(parents=True)
    reg_dir = registry_dir or (tmp_path / "registry")

    weights: Path | None = None
    if with_weights:
        weights = _make_fake_weights(train_dir)

    # Patch training job_status
    train_data = {
        "success": True,
        "message": "trained",
        "job_id": "local_abcd1234",
        "job_status": training_job_status,
        "runner": "local-yolo",
        "best_weights_path": str(weights) if weights else None,
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
        "mode": "local_train",
        "mlflow_run_id": None,
        "training_plan_path": None,
        "training_artifacts": [],
    }
    train_p = train_dir / "training_output.json"
    train_p.write_text(json.dumps(train_data), encoding="utf-8")

    eval_p = _write_evaluation_output(eval_dir, success=eval_success)
    approval_p = _write_approval_decision(approval_dir, approval_status, approval_action)

    inp = ModelRegistrationInput(
        model_name="test-model",
        training_output_path=str(train_p),
        evaluation_output_path=str(eval_p),
        approval_decision_path=str(approval_p),
        registry_dir=str(reg_dir),
        backend=RegistryBackend.LOCAL,
    )
    return inp, reg_dir


# ── Contract tests ─────────────────────────────────────────────────────────────


class TestModelRegistrationInputValidation:
    def test_valid_name(self) -> None:
        inp = ModelRegistrationInput(
            model_name="my-model-v1",
            training_output_path="/a",
            evaluation_output_path="/b",
            approval_decision_path="/c",
            registry_dir="/reg",
        )
        assert inp.model_name == "my-model-v1"

    def test_name_starts_with_digit(self) -> None:
        inp = ModelRegistrationInput(
            model_name="1model",
            training_output_path="/a",
            evaluation_output_path="/b",
            approval_decision_path="/c",
            registry_dir="/reg",
        )
        assert inp.model_name == "1model"

    def test_name_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ModelRegistrationInput(
                model_name="a" * 65,
                training_output_path="/a",
                evaluation_output_path="/b",
                approval_decision_path="/c",
                registry_dir="/reg",
            )

    def test_name_with_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ModelRegistrationInput(
                model_name="my/model",
                training_output_path="/a",
                evaluation_output_path="/b",
                approval_decision_path="/c",
                registry_dir="/reg",
            )

    def test_name_with_dotdot_raises(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ModelRegistrationInput(
                model_name="../evil",
                training_output_path="/a",
                evaluation_output_path="/b",
                approval_decision_path="/c",
                registry_dir="/reg",
            )

    def test_default_backend_is_local(self) -> None:
        inp = ModelRegistrationInput(
            model_name="m",
            training_output_path="/a",
            evaluation_output_path="/b",
            approval_decision_path="/c",
            registry_dir="/reg",
        )
        assert inp.backend == RegistryBackend.LOCAL


# ── Gate tests ─────────────────────────────────────────────────────────────────


class TestModelRegistryAgentGates:
    def test_gate_approval_status_not_approved(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path, approval_status="rejected")
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.BLOCKED
        assert "rejected" in result.block_reason

    def test_gate_approval_action_not_approve_model(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path, approval_action="request_retraining")
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.BLOCKED
        assert "request_retraining" in result.block_reason

    def test_gate_evaluation_not_successful(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path, eval_success=False)
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.BLOCKED

    def test_gate_training_not_completed(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path, training_job_status="failed")
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.BLOCKED
        assert "failed" in result.block_reason

    def test_gate_weights_missing(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path, with_weights=False)
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.BLOCKED

    def test_gate_missing_training_json(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path)
        inp = inp.model_copy(update={"training_output_path": "/nonexistent/training_output.json"})
        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)
        assert result.success is False
        assert result.status == RegistrationStatus.FAILED


# ── LocalModelRegistryClient tests ────────────────────────────────────────────


class TestLocalModelRegistryClient:
    def test_registers_version_1(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        result = agent.run(inp)

        assert result.success is True
        assert result.status == RegistrationStatus.REGISTERED
        assert result.version == 1
        assert result.model_name == "test-model"

    def test_registry_structure_created(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        result = agent.run(inp)

        version_dir = Path(result.registry_path)
        assert (version_dir / "model" / "best.pt").exists()
        assert (version_dir / "lineage.json").exists()
        assert (version_dir / "model_card.md").exists()
        assert (version_dir / "registration_output.json").exists()

    def test_latest_json_updated(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        agent.run(inp)

        latest = json.loads((reg_dir / "test-model" / "latest.json").read_text(encoding="utf-8"))
        assert latest["version"] == 1
        assert latest["model_name"] == "test-model"
        assert "sha256" in latest

    def test_second_registration_creates_version_2(self, tmp_path: Path) -> None:
        reg_dir = tmp_path / "registry"
        inp, _ = _make_registration_input(tmp_path, registry_dir=reg_dir)

        client = LocalModelRegistryClient()
        agent1 = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts1",
            registry_client=client,
        )
        agent2 = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts2",
            registry_client=client,
        )

        r1 = agent1.run(inp)
        r2 = agent2.run(inp)

        assert r1.version == 1
        assert r2.version == 2
        latest = json.loads((reg_dir / "test-model" / "latest.json").read_text(encoding="utf-8"))
        assert latest["version"] == 2

    def test_sha256_written_to_latest(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        agent.run(inp)

        latest = json.loads((reg_dir / "test-model" / "latest.json").read_text(encoding="utf-8"))
        assert len(latest["sha256"]) == 64  # hex SHA-256

    def test_lineage_json_contains_metrics(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        result = agent.run(inp)

        version_dir = Path(result.registry_path)
        lineage = json.loads((version_dir / "lineage.json").read_text(encoding="utf-8"))
        assert lineage["map50"] == pytest.approx(0.85)
        assert lineage["approved_by"] == "TestBot"
        assert lineage["approval_action"] == "approve_model"

    def test_model_card_contains_model_name(self, tmp_path: Path) -> None:
        inp, reg_dir = _make_registration_input(tmp_path)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        result = agent.run(inp)

        version_dir = Path(result.registry_path)
        card = (version_dir / "model_card.md").read_text(encoding="utf-8")
        assert "test-model" in card
        assert "v1" in card
        assert "TestBot" in card

    def test_registration_output_json_in_agent_artifacts_dir(self, tmp_path: Path) -> None:
        inp, _ = _make_registration_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        agent = ModelRegistryAgent(
            artifacts_dir=artifacts_dir,
            registry_client=LocalModelRegistryClient(),
        )
        agent.run(inp)
        assert (artifacts_dir / "registration_output.json").exists()

    def test_partial_version_dir_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        """If registration fails mid-copy, no partial version dir should remain."""
        inp, reg_dir = _make_registration_input(tmp_path, with_weights=False)
        # Manually create training_output.json with a nonexistent path
        (tmp_path / "training" / "training_output.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "job_status": "completed",
                    "best_weights_path": str(tmp_path / "nonexistent.pt"),
                    "runner": "local-yolo",
                    "job_id": "x",
                }
            ),
            encoding="utf-8",
        )
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts",
            registry_client=LocalModelRegistryClient(),
        )
        result = agent.run(inp)

        assert result.success is False
        assert result.status in (RegistrationStatus.FAILED, RegistrationStatus.BLOCKED)
        versions_dir = reg_dir / "test-model" / "versions"
        assert not versions_dir.exists() or not any(versions_dir.iterdir())


# ── FakeModelRegistryClient tests ─────────────────────────────────────────────


class TestFakeModelRegistryClient:
    def test_records_call(self, tmp_path: Path) -> None:
        client = FakeModelRegistryClient()
        lineage = ModelLineage(training_job_id="j1")
        inp = ModelRegistrationInput(
            model_name="m",
            training_output_path="/a",
            evaluation_output_path="/b",
            approval_decision_path="/c",
            registry_dir="/reg",
        )
        result = client.register(inp, lineage, tmp_path)
        assert len(client.calls) == 1
        assert result.status == RegistrationStatus.REGISTERED
        assert result.version == 1

    def test_increments_version(self, tmp_path: Path) -> None:
        client = FakeModelRegistryClient()
        lineage = ModelLineage()
        inp = ModelRegistrationInput(
            model_name="m",
            training_output_path="/a",
            evaluation_output_path="/b",
            approval_decision_path="/c",
            registry_dir="/reg",
        )
        r1 = client.register(inp, lineage, tmp_path)
        r2 = client.register(inp, lineage, tmp_path)
        assert r1.version == 1
        assert r2.version == 2


# ── MLflowModelRegistryClient tests ───────────────────────────────────────────
# Uses the real `mlflow` package against a private SQLite store per test (tmp_path) —
# no server required, and it exercises the actual Model Registry API, not a mock.


def _make_mlflow_registration_input(
    tmp_path: Path,
    *,
    model_name: str = "mlflow-test-model",
    with_weights: bool = True,
) -> ModelRegistrationInput:
    train_dir = tmp_path / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    weights = _make_fake_weights(train_dir) if with_weights else None
    train_p = _write_training_output(train_dir, best_weights=weights)

    return ModelRegistrationInput(
        model_name=model_name,
        training_output_path=str(train_p),
        evaluation_output_path=str(tmp_path / "unused_eval.json"),
        approval_decision_path=str(tmp_path / "unused_approval.json"),
        registry_dir=str(tmp_path / "unused_registry"),
        backend=RegistryBackend.MLFLOW,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_experiment_name="mlflow-registry-tests",
    )


class TestMLflowModelRegistryClient:
    def test_registers_version_1(self, tmp_path: Path) -> None:
        inp = _make_mlflow_registration_input(tmp_path)
        lineage = ModelLineage(map50=0.85, precision=0.8, recall=0.82, approved_by="TestBot")

        client = MLflowModelRegistryClient()
        result = client.register(inp, lineage, tmp_path / "artifacts")

        assert result.success is True
        assert result.status == RegistrationStatus.REGISTERED
        assert result.version == 1
        assert result.registry_path == f"models:/{inp.model_name}/1"

    def test_second_registration_creates_version_2(self, tmp_path: Path) -> None:
        inp = _make_mlflow_registration_input(tmp_path)
        lineage = ModelLineage()
        client = MLflowModelRegistryClient()

        r1 = client.register(inp, lineage, tmp_path / "artifacts1")
        r2 = client.register(inp, lineage, tmp_path / "artifacts2")

        assert r1.version == 1
        assert r2.version == 2

    def test_local_audit_files_written(self, tmp_path: Path) -> None:
        inp = _make_mlflow_registration_input(tmp_path)
        lineage = ModelLineage(map50=0.9)
        artifacts_dir = tmp_path / "artifacts"

        client = MLflowModelRegistryClient()
        result = client.register(inp, lineage, artifacts_dir)

        assert result.success is True
        assert (artifacts_dir / "lineage.json").exists()
        assert (artifacts_dir / "model_card.md").exists()
        assert (artifacts_dir / "registration_output.json").exists()
        card = (artifacts_dir / "model_card.md").read_text(encoding="utf-8")
        assert inp.model_name in card

    def test_lineage_tags_set_on_model_version(self, tmp_path: Path) -> None:
        import mlflow  # noqa: PLC0415

        inp = _make_mlflow_registration_input(tmp_path)
        lineage = ModelLineage(map50=0.85, approved_by="TestBot")

        client = MLflowModelRegistryClient()
        result = client.register(inp, lineage, tmp_path / "artifacts")

        mlflow.set_tracking_uri(inp.mlflow_tracking_uri)
        mv = mlflow.tracking.MlflowClient().get_model_version(inp.model_name, str(result.version))
        assert mv.tags["map50"] == "0.8500"
        assert mv.tags["approved_by"] == "TestBot"

    def test_missing_weights_returns_failed(self, tmp_path: Path) -> None:
        inp = _make_mlflow_registration_input(tmp_path, with_weights=False)
        client = MLflowModelRegistryClient()
        result = client.register(inp, ModelLineage(), tmp_path / "artifacts")

        assert result.success is False
        assert result.status == RegistrationStatus.FAILED

    def test_reuses_existing_mlflow_run_when_provided(self, tmp_path: Path) -> None:
        import mlflow  # noqa: PLC0415

        inp = _make_mlflow_registration_input(tmp_path)
        mlflow.set_tracking_uri(inp.mlflow_tracking_uri)
        mlflow.set_experiment(inp.mlflow_experiment_name)
        run = mlflow.start_run(run_name="parent-run")
        inp = inp.model_copy(update={"mlflow_run_id": run.info.run_id})
        mlflow.end_run()

        client = MLflowModelRegistryClient()
        result = client.register(inp, ModelLineage(), tmp_path / "artifacts")

        assert result.success is True
        assert result.metadata["mlflow_run_id"] == run.info.run_id

    def test_returns_failed_when_mlflow_not_installed(self, tmp_path: Path) -> None:
        inp = _make_mlflow_registration_input(tmp_path)
        real_import = builtins.__import__

        def _block_mlflow(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("mlflow not installed")
            return real_import(name, *args, **kwargs)

        cached = sys.modules.pop("mlflow", None)
        try:
            with patch("builtins.__import__", side_effect=_block_mlflow):
                client = MLflowModelRegistryClient()
                result = client.register(inp, ModelLineage(), tmp_path / "artifacts")
            assert result.success is False
            assert result.status == RegistrationStatus.FAILED
            assert "not installed" in result.message
        finally:
            if cached is not None:
                sys.modules["mlflow"] = cached


# ── AzureMLModelRegistryClient tests ──────────────────────────────────────────
# azure-ai-ml is NOT required to be installed — the SDK's constants/entities modules
# are mocked; Azure connectivity is replaced with FakeAzureMLClientFactory/FakeMLClient.


def _minimal_azure_config(**overrides) -> AzureMLConfig:
    defaults: dict = {
        "subscription_id": "sub-123",
        "resource_group": "rg-test",
        "workspace_name": "ws-test",
        "compute_name": "gpu-cluster",
        "environment": {
            "mode": "registered",
            "registered_environment": "azureml:yolo-env:1",
        },
    }
    defaults.update(overrides)
    return AzureMLConfig.model_validate(defaults)


def _mock_azure_ml_entities_modules():
    """Return (mock_constants, mock_entities) with a fake Model() constructor."""
    mock_asset_types = MagicMock(CUSTOM_MODEL="custom_model")
    mock_constants = MagicMock(AssetTypes=mock_asset_types)

    def fake_model(**kwargs):
        m = MagicMock()
        m.path = kwargs.get("path")
        m.name = kwargs.get("name")
        m.type = kwargs.get("type")
        m.description = kwargs.get("description")
        m.tags = kwargs.get("tags")
        return m

    mock_entities = MagicMock(Model=fake_model)
    return mock_constants, mock_entities


def _make_azure_registration_input(
    tmp_path: Path,
    *,
    model_name: str = "azure-test-model",
    with_weights: bool = True,
) -> ModelRegistrationInput:
    train_dir = tmp_path / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    weights = _make_fake_weights(train_dir) if with_weights else None
    train_p = _write_training_output(train_dir, best_weights=weights)

    return ModelRegistrationInput(
        model_name=model_name,
        training_output_path=str(train_p),
        evaluation_output_path=str(tmp_path / "unused_eval.json"),
        approval_decision_path=str(tmp_path / "unused_approval.json"),
        registry_dir=str(tmp_path / "unused_registry"),
        backend=RegistryBackend.AZURE_ML,
    )


class TestAzureMLModelRegistryClient:
    def test_registers_version_1(self, tmp_path: Path) -> None:
        factory = FakeAzureMLClientFactory(job_status="Completed")
        client = AzureMLModelRegistryClient(_minimal_azure_config(), client_factory=factory)
        inp = _make_azure_registration_input(tmp_path)

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            result = client.register(inp, ModelLineage(map50=0.85), tmp_path / "artifacts")

        assert result.success is True
        assert result.status == RegistrationStatus.REGISTERED
        assert result.version == 1
        assert "azure-test-model" in result.registry_path

    def test_second_registration_creates_version_2(self, tmp_path: Path) -> None:
        # A real Azure workspace holds versioning state server-side, so two
        # registrations against the same workspace increment the version even
        # though each call gets its own MLClient object. FakeAzureMLClientFactory
        # deliberately returns an *independent* client per create() call (see
        # test_fake_client_factory_creates_independent_clients), so simulate the
        # "same workspace" case here with a factory that reuses one FakeMLClient.
        class _SingleClientFactory:
            def __init__(self, client: FakeMLClient) -> None:
                self._client = client

            def create(self, config: AzureMLConfig) -> FakeMLClient:
                return self._client

        shared_client = FakeMLClient(job_status="Completed")
        client = AzureMLModelRegistryClient(
            _minimal_azure_config(), client_factory=_SingleClientFactory(shared_client)
        )
        inp = _make_azure_registration_input(tmp_path)

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            r1 = client.register(inp, ModelLineage(), tmp_path / "artifacts1")
            r2 = client.register(inp, ModelLineage(), tmp_path / "artifacts2")

        assert r1.version == 1
        assert r2.version == 2

    def test_local_audit_files_written(self, tmp_path: Path) -> None:
        factory = FakeAzureMLClientFactory(job_status="Completed")
        client = AzureMLModelRegistryClient(_minimal_azure_config(), client_factory=factory)
        inp = _make_azure_registration_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            result = client.register(inp, ModelLineage(map50=0.9), artifacts_dir)

        assert result.success is True
        assert (artifacts_dir / "lineage.json").exists()
        assert (artifacts_dir / "model_card.md").exists()
        assert (artifacts_dir / "registration_output.json").exists()

    def test_missing_weights_returns_failed(self, tmp_path: Path) -> None:
        factory = FakeAzureMLClientFactory(job_status="Completed")
        client = AzureMLModelRegistryClient(_minimal_azure_config(), client_factory=factory)
        inp = _make_azure_registration_input(tmp_path, with_weights=False)

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            result = client.register(inp, ModelLineage(), tmp_path / "artifacts")

        assert result.success is False
        assert result.status == RegistrationStatus.FAILED

    def test_model_asset_built_with_lineage_tags(self, tmp_path: Path) -> None:
        factory = FakeAzureMLClientFactory(job_status="Completed")
        client = AzureMLModelRegistryClient(_minimal_azure_config(), client_factory=factory)
        inp = _make_azure_registration_input(tmp_path)

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            client.register(
                inp, ModelLineage(map50=0.85, approved_by="TestBot"), tmp_path / "artifacts"
            )

        created_model = factory.last_client.models.created[-1]
        assert created_model.tags["map50"] == "0.8500"
        assert created_model.tags["approved_by"] == "TestBot"
        assert created_model.name == inp.model_name


class TestCreateRegistryClient:
    def test_local_backend(self) -> None:
        assert isinstance(create_registry_client(RegistryBackend.LOCAL), LocalModelRegistryClient)

    def test_mlflow_backend(self) -> None:
        assert isinstance(create_registry_client(RegistryBackend.MLFLOW), MLflowModelRegistryClient)

    def test_azure_ml_backend_raises_without_config(self) -> None:
        with pytest.raises(ValueError, match="AzureMLConfig"):
            create_registry_client(RegistryBackend.AZURE_ML)

    def test_agent_uses_injected_azure_ml_client(self, tmp_path: Path) -> None:
        """ModelRegistryAgent uses an explicitly injected Azure ML client, bypassing routing."""
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        approval_dir = tmp_path / "approval"
        approval_dir.mkdir()

        weights = _make_fake_weights(train_dir)
        train_p = _write_training_output(train_dir, best_weights=weights)
        eval_p = _write_evaluation_output(eval_dir)
        approval_p = _write_approval_decision(approval_dir)

        inp = ModelRegistrationInput(
            model_name="agent-routed-azure-model",
            training_output_path=str(train_p),
            evaluation_output_path=str(eval_p),
            approval_decision_path=str(approval_p),
            registry_dir=str(tmp_path / "unused_registry"),
            backend=RegistryBackend.AZURE_ML,
        )

        factory = FakeAzureMLClientFactory(job_status="Completed")
        azure_client = AzureMLModelRegistryClient(_minimal_azure_config(), client_factory=factory)
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "artifacts", registry_client=azure_client
        )

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with patch.dict(
            sys.modules,
            {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
        ):
            result = agent.run(inp)

        assert result.success is True
        assert result.status == RegistrationStatus.REGISTERED

    def test_agent_routes_to_mlflow_backend_from_input(self, tmp_path: Path) -> None:
        """ModelRegistryAgent picks the client from inp.backend when none is injected."""
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        approval_dir = tmp_path / "approval"
        approval_dir.mkdir()

        weights = _make_fake_weights(train_dir)
        train_p = _write_training_output(train_dir, best_weights=weights)
        eval_p = _write_evaluation_output(eval_dir)
        approval_p = _write_approval_decision(approval_dir)

        inp = ModelRegistrationInput(
            model_name="agent-routed-model",
            training_output_path=str(train_p),
            evaluation_output_path=str(eval_p),
            approval_decision_path=str(approval_p),
            registry_dir=str(tmp_path / "unused_registry"),
            backend=RegistryBackend.MLFLOW,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        )

        agent = ModelRegistryAgent(artifacts_dir=tmp_path / "artifacts")
        result = agent.run(inp)

        assert result.success is True
        assert result.status == RegistrationStatus.REGISTERED
        assert result.registry_path == "models:/agent-routed-model/1"


# ── Workflow Step 5 tests ──────────────────────────────────────────────────────


class _StubAgent(BaseAgent):
    def __init__(self, artifacts_dir: Path, result: object) -> None:
        super().__init__(artifacts_dir)
        self._result = result

    def run(self, input: object) -> object:  # noqa: A002
        return self._result


def _make_training_config(path: Path) -> Path:
    cfg = {"model": "yolo11m.pt", "epochs": 1, "mode": "local_dry_run"}
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _approval_ok(action: ApprovalAction = ApprovalAction.APPROVE_MODEL) -> ApprovalOutput:
    return ApprovalOutput(
        success=True,
        message="approved",
        status=ApprovalStatus.APPROVED,
        action=action,
        approver="TestBot",
        generated_artifacts=[],
    )


def _eval_ok() -> EvaluationOutput:
    return EvaluationOutput(
        success=True,
        message="eval ok",
        metrics=EvaluationMetrics(map50=0.86, map50_95=0.59, precision=0.84, recall=0.79),
        recommendation="promote_candidate",
        passed_checks=["map50 0.86 >= 0.75"],
    )


def _train_ok() -> TrainingOutput:
    return TrainingOutput(
        success=True,
        message="trained",
        job_status=TrainingJobStatus.COMPLETED,
    )


def _val_ok() -> DatasetValidationOutput:
    return DatasetValidationOutput(
        success=True,
        message="ok",
        status="passed",
        num_images=6,
        num_labels=6,
    )


class TestMVPWorkflowStep5:
    def _make_registry_agent_stub(
        self, tmp_path: Path
    ) -> tuple[ModelRegistryAgent, FakeModelRegistryClient]:
        client = FakeModelRegistryClient()
        agent = ModelRegistryAgent(
            artifacts_dir=tmp_path / "registry_artifacts",
            registry_client=client,
        )
        return agent, client

    def _make_workflow_with_registry(
        self,
        tmp_path: Path,
        *,
        register: bool = True,
        approval_result: ApprovalOutput | None = None,
        registry_agent: ModelRegistryAgent | None = None,
        train_dir: Path | None = None,
        eval_dir: Path | None = None,
        approval_dir: Path | None = None,
    ) -> tuple[MVPWorkflow, MVPWorkflowInput]:
        cfg_path = _make_training_config(tmp_path / "train.yaml")
        out_dir = tmp_path / "out"

        # Pre-write required JSON files so the registry agent can read them
        train_out = out_dir / "training"
        train_out.mkdir(parents=True, exist_ok=True)
        weights = _make_fake_weights(train_out)
        _write_training_output(train_out, best_weights=weights)

        eval_out = out_dir / "evaluation"
        eval_out.mkdir(parents=True, exist_ok=True)
        _write_evaluation_output(eval_out)

        approval_out = out_dir / "approval"
        approval_out.mkdir(parents=True, exist_ok=True)
        _write_approval_decision(approval_out)

        reg_dir = tmp_path / "reg"

        fake_client = FakeModelRegistryClient()
        reg_agent_to_use = registry_agent or ModelRegistryAgent(
            artifacts_dir=tmp_path / "registry_artifacts",
            registry_client=fake_client,
        )

        workflow = MVPWorkflow(
            _validation_factory=lambda d: _StubAgent(d, _val_ok()),
            _training_factory=lambda d: _StubAgent(d, _train_ok()),
            _evaluation_factory=lambda d: _StubAgent(d, _eval_ok()),
            _approval_factory=lambda d: _StubAgent(d, approval_result or _approval_ok()),
            _registry_factory=lambda d: reg_agent_to_use,
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
            register_approved_model=register,
            model_name="test-model",
            registry_dir=str(reg_dir),
        )
        return workflow, inp

    def test_registry_step_runs_when_approved(self, tmp_path: Path) -> None:
        workflow, inp = self._make_workflow_with_registry(tmp_path, register=True)
        result = workflow.run(inp)

        assert result.success is True
        registry_step = next((s for s in result.steps if s.step == "model_registry"), None)
        assert registry_step is not None
        assert registry_step.status == "registered"
        assert result.registration_status == "registered"
        assert result.registered_model_version == 1

    def test_registry_step_skipped_when_flag_false(self, tmp_path: Path) -> None:
        workflow, inp = self._make_workflow_with_registry(tmp_path, register=False)
        result = workflow.run(inp)

        assert result.success is True
        registry_step = next((s for s in result.steps if s.step == "model_registry"), None)
        assert registry_step is None
        assert result.registration_status is None

    def test_registry_step_skipped_when_not_approve_model(self, tmp_path: Path) -> None:
        rejection = ApprovalOutput(
            success=True,
            message="rejected",
            status=ApprovalStatus.REJECTED,
            action=ApprovalAction.REJECT_MODEL,
            approver="TestBot",
            generated_artifacts=[],
        )
        workflow, inp = self._make_workflow_with_registry(
            tmp_path, register=True, approval_result=rejection
        )
        result = workflow.run(inp)

        assert result.success is True
        registry_step = next((s for s in result.steps if s.step == "model_registry"), None)
        assert registry_step is not None
        assert registry_step.status == "skipped"
        assert result.registration_status is None

    def test_workflow_has_5_steps_with_registry(self, tmp_path: Path) -> None:
        workflow, inp = self._make_workflow_with_registry(tmp_path, register=True)
        result = workflow.run(inp)
        assert len(result.steps) == 5

    def test_workflow_has_4_steps_without_registry(self, tmp_path: Path) -> None:
        workflow, inp = self._make_workflow_with_registry(tmp_path, register=False)
        result = workflow.run(inp)
        assert len(result.steps) == 4


# ── CLI tests ──────────────────────────────────────────────────────────────────

runner = CliRunner()


class TestRegisterModelCLI:
    def test_register_model_command_exists(self) -> None:
        result = runner.invoke(app, ["register-model", "--help"])
        assert result.exit_code == 0
        assert "register" in result.output.lower()

    def test_register_model_success(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        approval_dir = tmp_path / "approval"
        approval_dir.mkdir()
        reg_dir = tmp_path / "registry"

        weights = _make_fake_weights(train_dir)
        _write_training_output(train_dir, best_weights=weights)
        _write_evaluation_output(eval_dir)
        _write_approval_decision(approval_dir)

        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name",
                "cli-test-model",
                "--training-output",
                str(train_dir / "training_output.json"),
                "--evaluation-output",
                str(eval_dir / "evaluation_output.json"),
                "--approval-decision",
                str(approval_dir / "approval_decision.json"),
                "--registry-dir",
                str(reg_dir),
                "--output-dir",
                str(tmp_path / "artifacts"),
            ],
        )
        assert result.exit_code == 0
        assert "REGISTERED" in result.output

    def test_register_model_blocked_shows_block_reason(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        approval_dir = tmp_path / "approval"
        approval_dir.mkdir()
        reg_dir = tmp_path / "registry"

        weights = _make_fake_weights(train_dir)
        _write_training_output(train_dir, best_weights=weights)
        _write_evaluation_output(eval_dir)
        _write_approval_decision(approval_dir, status="rejected", action="reject_model")

        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name",
                "cli-test-model",
                "--training-output",
                str(train_dir / "training_output.json"),
                "--evaluation-output",
                str(eval_dir / "evaluation_output.json"),
                "--approval-decision",
                str(approval_dir / "approval_decision.json"),
                "--registry-dir",
                str(reg_dir),
                "--output-dir",
                str(tmp_path / "artifacts"),
            ],
        )
        assert result.exit_code == 1
        assert "BLOCKED" in result.output

    def test_register_model_invalid_name_raises(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name",
                "bad/name",
                "--training-output",
                "/a",
                "--evaluation-output",
                "/b",
                "--approval-decision",
                "/c",
                "--registry-dir",
                str(tmp_path / "registry"),
            ],
        )
        assert result.exit_code != 0

    def test_run_mvp_has_register_flag(self) -> None:
        result = runner.invoke(app, ["run-mvp", "--help"])
        # Rich may truncate the long flag name; check a stable prefix
        assert "--register-approve" in result.output
        assert "--model-name" in result.output
        assert "--registry-backend" in result.output
        assert "--registry-dir" in result.output


class TestRegisterModelAzureMLCLI:
    def test_azure_ml_without_azure_config_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name",
                "m",
                "--training-output",
                "/a",
                "--evaluation-output",
                "/b",
                "--approval-decision",
                "/c",
                "--registry-dir",
                str(tmp_path / "registry"),
                "--backend",
                "azure_ml",
            ],
        )
        assert result.exit_code == 1
        assert "azure-config" in result.output

    def test_azure_ml_with_config_succeeds(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        approval_dir = tmp_path / "approval"
        approval_dir.mkdir()

        weights = _make_fake_weights(train_dir)
        _write_training_output(train_dir, best_weights=weights)
        _write_evaluation_output(eval_dir)
        _write_approval_decision(approval_dir)

        azure_cfg_path = tmp_path / "azure_ml.yaml"
        azure_cfg_path.write_text(
            "subscription_id: sub\nresource_group: rg\nworkspace_name: ws\n"
            "compute_name: c\nenvironment:\n  mode: registered\n"
            "  registered_environment: azureml:e:1\n",
            encoding="utf-8",
        )

        mock_constants, mock_entities = _mock_azure_ml_entities_modules()
        with (
            patch.object(
                DefaultAzureMLClientFactory,
                "create",
                return_value=FakeMLClient(job_status="Completed"),
            ),
            patch.dict(
                sys.modules,
                {"azure.ai.ml.constants": mock_constants, "azure.ai.ml.entities": mock_entities},
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "register-model",
                    "--model-name",
                    "cli-azure-model",
                    "--training-output",
                    str(train_dir / "training_output.json"),
                    "--evaluation-output",
                    str(eval_dir / "evaluation_output.json"),
                    "--approval-decision",
                    str(approval_dir / "approval_decision.json"),
                    "--registry-dir",
                    str(tmp_path / "registry"),
                    "--backend",
                    "azure_ml",
                    "--azure-config",
                    str(azure_cfg_path),
                    "--output-dir",
                    str(tmp_path / "artifacts"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "REGISTERED" in result.output
