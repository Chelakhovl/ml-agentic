"""Unit tests for integrations/artifact_store.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_mlops.integrations.artifact_store import (
    AzureBlobArtifactStore,
    FakeArtifactStore,
    NoOpArtifactStore,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_azure_config(
    enabled: bool = True,
    account_url: str | None = None,
    container_name: str | None = None,
    artifact_prefix: str = "agentic-mlops",
    datastore_name: str = "workspaceblobstore",
) -> Any:
    """Build a minimal AzureMLConfig-like object without requiring Azure SDK."""
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

    return AzureMLConfig.model_validate(
        {
            "subscription_id": "fake-sub",
            "resource_group": "fake-rg",
            "workspace_name": "fake-ws",
            "compute_name": "fake-compute",
            "environment": {"mode": "registered", "registered_environment": "env:1"},
            "storage": {
                "enabled": enabled,
                "account_url": account_url,
                "container_name": container_name,
                "artifact_prefix": artifact_prefix,
                "datastore_name": datastore_name,
            },
        }
    )


def _mock_blob_sdk() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (BlobServiceClient mock, blob_client mock, DefaultAzureCredential mock)."""
    blob_client_mock = MagicMock()
    blob_client_mock.exists.return_value = True
    blob_client_mock.download_blob.return_value.readall.return_value = b"data"

    service_mock = MagicMock()
    service_mock.get_blob_client.return_value = blob_client_mock

    cred_mock = MagicMock()

    return service_mock, blob_client_mock, cred_mock


# ── NoOpArtifactStore ─────────────────────────────────────────────────────────


class TestNoOpArtifactStore:
    def test_upload_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert NoOpArtifactStore().upload_file(f, "some/key") == ""

    def test_upload_directory_returns_empty_list(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        assert NoOpArtifactStore().upload_directory(tmp_path, "prefix") == []

    def test_exists_returns_false(self):
        assert NoOpArtifactStore().exists("any/key") is False

    def test_download_file_is_noop(self, tmp_path):
        dest = tmp_path / "out.txt"
        NoOpArtifactStore().download_file("key", dest)
        assert not dest.exists()


# ── FakeArtifactStore ─────────────────────────────────────────────────────────


class TestFakeArtifactStore:
    def test_upload_file_records_and_returns_uri(self, tmp_path):
        store = FakeArtifactStore()
        f = tmp_path / "model.pt"
        f.write_bytes(b"weights")
        uri = store.upload_file(f, "wf/training/model.pt")
        assert "wf/training/model.pt" in uri
        assert (f, "wf/training/model.pt") in store.uploaded

    def test_upload_directory_records(self, tmp_path):
        store = FakeArtifactStore()
        uris = store.upload_directory(tmp_path, "wf/training")
        assert (tmp_path, "wf/training") in store.uploaded
        assert uris == []

    def test_upload_file_multiple_calls_accumulate(self, tmp_path):
        store = FakeArtifactStore()
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a")
        f2.write_text("b")
        store.upload_file(f1, "k1")
        store.upload_file(f2, "k2")
        assert len(store.uploaded) == 2

    def test_download_file_records(self, tmp_path):
        store = FakeArtifactStore()
        dest = tmp_path / "out.txt"
        store.download_file("some/key", dest)
        assert ("some/key", dest) in store.downloaded

    def test_exists_returns_false(self):
        assert FakeArtifactStore().exists("any/key") is False


# ── AzureBlobArtifactStore unit tests (mock azure-storage-blob) ───────────────


class TestAzureBlobArtifactStoreUnit:
    """Tests use the FakeMLClient / fake Azure SDK to avoid real network calls."""

    def _make_store_via_fake_factory(
        self,
        service_mock: MagicMock,
        cred_mock: MagicMock,
        account_url: str | None = None,
        container_name: str | None = None,
    ) -> AzureBlobArtifactStore:
        """Construct AzureBlobArtifactStore with mocked SDK calls."""
        from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory

        cfg = _make_azure_config(
            account_url=account_url,
            container_name=container_name,
        )

        fake_blob_module = MagicMock()
        fake_blob_module.BlobServiceClient.return_value = service_mock
        fake_id_module = MagicMock()
        fake_id_module.DefaultAzureCredential.return_value = cred_mock

        with patch.dict(sys.modules, {
            "azure.storage.blob": fake_blob_module,
            "azure.identity": fake_id_module,
        }):
            store = AzureBlobArtifactStore(cfg, client_factory=FakeAzureMLClientFactory())
        return store

    def test_upload_file_calls_upload_blob(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        f = tmp_path / "best.pt"
        f.write_bytes(b"weights")
        uri = store.upload_file(f, "training/best.pt")
        blob_client.upload_blob.assert_called_once()
        _, kwargs = blob_client.upload_blob.call_args
        assert kwargs.get("overwrite") is True
        assert "training/best.pt" in uri

    def test_upload_file_includes_prefix(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        f = tmp_path / "metrics.json"
        f.write_bytes(b"{}")
        store.upload_file(f, "evaluation/metrics.json")
        call_kwargs = service.get_blob_client.call_args[1]
        assert call_kwargs["container"] == "fake-container"
        assert "agentic-mlops" in call_kwargs["blob"]
        assert "evaluation/metrics.json" in call_kwargs["blob"]

    def test_upload_file_exception_returns_empty_string(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        blob_client.upload_blob.side_effect = OSError("network error")
        store = self._make_store_via_fake_factory(service, cred)
        f = tmp_path / "a.txt"
        f.write_text("x")
        result = store.upload_file(f, "key")
        assert result == ""

    def test_upload_directory_uploads_each_file(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        uris = store.upload_directory(tmp_path, "wf/step")
        assert blob_client.upload_blob.call_count == 2
        assert len(uris) == 2

    def test_upload_directory_skips_pipeline_outputs(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        (tmp_path / "metrics.json").write_text("{}")
        pipeline_dir = tmp_path / "_pipeline_outputs"
        pipeline_dir.mkdir()
        (pipeline_dir / "temp.pt").write_bytes(b"x")
        store.upload_directory(tmp_path, "wf/step")
        assert blob_client.upload_blob.call_count == 1

    def test_upload_directory_skips_pyc(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        (tmp_path / "model.pyc").write_bytes(b"bytecode")
        (tmp_path / "metrics.json").write_text("{}")
        store.upload_directory(tmp_path, "wf/step")
        assert blob_client.upload_blob.call_count == 1

    def test_upload_directory_nonexistent_returns_empty(self, tmp_path):
        service, blob_client, cred = _mock_blob_sdk()
        store = self._make_store_via_fake_factory(service, cred)
        uris = store.upload_directory(tmp_path / "nonexistent", "wf/step")
        assert uris == []
        blob_client.upload_blob.assert_not_called()

    def test_auto_resolve_uses_datastores_get(self, tmp_path):
        """auto-resolve mode calls ml_client.datastores.get(datastore_name)."""
        from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory

        service, _, cred = _mock_blob_sdk()
        factory = FakeAzureMLClientFactory()
        cfg = _make_azure_config(account_url=None, container_name=None, datastore_name="myds")

        fake_blob_module = MagicMock()
        fake_blob_module.BlobServiceClient.return_value = service
        fake_id_module = MagicMock()
        fake_id_module.DefaultAzureCredential.return_value = cred

        with patch.dict(sys.modules, {
            "azure.storage.blob": fake_blob_module,
            "azure.identity": fake_id_module,
        }):
            AzureBlobArtifactStore(cfg, client_factory=factory)

        assert "myds" in factory.last_client.datastores.fetched

    def test_explicit_mode_skips_datastores_get(self, tmp_path):
        """explicit account_url mode must NOT call ml_client.datastores.get()."""
        from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory

        service, _, cred = _mock_blob_sdk()
        factory = FakeAzureMLClientFactory()
        cfg = _make_azure_config(
            account_url="https://myaccount.blob.core.windows.net",
            container_name="mycontainer",
        )

        fake_blob_module = MagicMock()
        fake_blob_module.BlobServiceClient.return_value = service
        fake_id_module = MagicMock()
        fake_id_module.DefaultAzureCredential.return_value = cred

        with patch.dict(sys.modules, {
            "azure.storage.blob": fake_blob_module,
            "azure.identity": fake_id_module,
        }):
            AzureBlobArtifactStore(cfg, client_factory=factory)

        # No MLClient was created — no datastore lookups
        assert factory.last_client is None

    def test_explicit_mode_requires_container_name(self):
        """account_url without container_name must raise ValueError."""
        from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory

        service, _, cred = _mock_blob_sdk()
        factory = FakeAzureMLClientFactory()
        cfg = _make_azure_config(
            account_url="https://myaccount.blob.core.windows.net",
            container_name=None,  # missing!
        )

        fake_blob_module = MagicMock()
        fake_blob_module.BlobServiceClient.return_value = service
        fake_id_module = MagicMock()
        fake_id_module.DefaultAzureCredential.return_value = cred

        with patch.dict(sys.modules, {
            "azure.storage.blob": fake_blob_module,
            "azure.identity": fake_id_module,
        }):
            with pytest.raises(ValueError, match="container_name"):
                AzureBlobArtifactStore(cfg, client_factory=factory)

    def test_missing_sdk_raises_runtime_error(self):
        """ImportError from azure-storage-blob must surface as RuntimeError."""
        from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory

        cfg = _make_azure_config(account_url="https://x.blob.core.windows.net", container_name="c")

        with patch.dict(sys.modules, {"azure.storage.blob": None}):  # type: ignore[dict-item]
            with pytest.raises((RuntimeError, ImportError)):
                AzureBlobArtifactStore(cfg, client_factory=FakeAzureMLClientFactory())


# ── Orchestrator integration ──────────────────────────────────────────────────


def _make_simple_orchestrator_input(tmp_path: Path, dataset_path: Path) -> Any:
    from agentic_mlops.contracts.orchestrator import OrchestratorInput

    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [cat]\n")
    return OrchestratorInput(
        workflow_id="test_artifact_wf",
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        steps=["dataset_validation"],
        runs_dir=str(tmp_path / "runs"),
    )


def _make_minimal_dataset(tmp_path: Path) -> Path:
    """Create a tiny but valid YOLO dataset structure with distinct images per split."""
    ds = tmp_path / "ds"
    for i, split in enumerate(("train", "val")):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        img = ds / "images" / split / f"img_{split}.jpg"
        # unique content per split to avoid duplicate-image validation failure
        img.write_bytes(b"\xff\xd8\xff" + split.encode() + b"\xff\xd9")
        lbl = ds / "labels" / split / f"img_{split}.txt"
        lbl.write_text("0 0.5 0.5 0.1 0.1\n")
    return ds


class TestOrchestratorArtifactUpload:
    def test_upload_directory_called_after_step(self, tmp_path):
        """FakeArtifactStore.upload_directory should be called after dataset_validation."""
        store = FakeArtifactStore()
        ds = _make_minimal_dataset(tmp_path)
        inp = _make_simple_orchestrator_input(tmp_path, ds)

        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        OrchestratorWorkflow(artifact_store=store).run(inp)

        prefixes = [prefix for _, prefix in store.uploaded]
        assert any("dataset_validation" in str(p) for p in prefixes)

    def test_state_and_audit_uploaded_after_step(self, tmp_path):
        """state.json and audit_log.jsonl must be uploaded after each step."""
        store = FakeArtifactStore()
        ds = _make_minimal_dataset(tmp_path)
        inp = _make_simple_orchestrator_input(tmp_path, ds)

        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        OrchestratorWorkflow(artifact_store=store).run(inp)

        paths_or_keys = [str(p) for p, _ in store.uploaded]
        assert any("state.json" in k for k in paths_or_keys)
        assert any("audit_log.jsonl" in k for k in paths_or_keys)

    def test_noop_store_default_no_error(self, tmp_path):
        """OrchestratorWorkflow without artifact_store must not error."""
        ds = _make_minimal_dataset(tmp_path)
        inp = _make_simple_orchestrator_input(tmp_path, ds)

        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        result = OrchestratorWorkflow().run(inp)
        assert result is not None


# ── MVPWorkflow integration ───────────────────────────────────────────────────


def _make_stub_agent(success: bool = True, msg: str = "ok") -> type:
    """Return a stub agent class whose run() returns a minimal ToolResult-like dict."""

    class _Stub:
        def __init__(self, artifacts_dir: Path) -> None:
            self._dir = artifacts_dir

        def run(self, inp: Any) -> Any:
            from agentic_mlops.contracts.validation import DatasetValidationOutput

            from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalOutput
            from agentic_mlops.contracts.evaluation import EvaluationOutput
            from agentic_mlops.contracts.model_registry import ModelRegistrationOutput
            from agentic_mlops.contracts.training import TrainingOutput

            # Return a plausible stub for every type of agent that MVPWorkflow calls.
            name = type(inp).__name__
            if "Validation" in name:
                return DatasetValidationOutput(
                    success=True, message="ok", status="passed",
                    num_images={"train": 1, "val": 1},
                    num_labels={"train": 1, "val": 1},
                    issues=[], warnings=[],
                )
            if "Training" in name:
                pt = self._dir / "best.pt"
                pt.write_bytes(b"fake")
                return TrainingOutput(
                    success=True, message="ok",
                    job_status="completed",
                    best_weights_path=str(pt),
                )
            if "Evaluation" in name:
                return EvaluationOutput(
                    success=True, message="ok",
                    map50=0.9, map50_95=0.7, precision=0.85, recall=0.80,
                    recommendation="PROMOTE_CANDIDATE",
                    passed_checks=[], failed_checks=[],
                )
            if "Approval" in name:
                return ApprovalOutput(
                    success=True, message="ok",
                    status="approved",
                    action=ApprovalAction.APPROVE_MODEL,
                    approver="test",
                )
            if "Registration" in name:
                return ModelRegistrationOutput(
                    success=True, message="ok",
                    model_name="m", version="1",
                    registry_path=str(self._dir),
                )
            raise ValueError(f"Unknown input type: {name}")

    return _Stub


class TestMVPWorkflowArtifactUpload:
    def _make_input(self, tmp_path: Path) -> Any:
        from agentic_mlops.contracts.workflows import MVPWorkflowInput

        ds = _make_minimal_dataset(tmp_path)
        data_yaml = ds / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [cat]\n")
        training_cfg = tmp_path / "training.yaml"
        training_cfg.write_text("runner: fake\nepochs: 1\n")
        return MVPWorkflowInput(
            dataset_path=str(ds),
            data_yaml_path=str(data_yaml),
            training_config_path=str(training_cfg),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            interactive_approval=False,
            approval_action="approve_model",
        )

    def test_upload_called_for_training_and_evaluation(self, tmp_path):
        store = FakeArtifactStore()
        from agentic_mlops.workflows.mvp_workflow import MVPWorkflow

        inp = self._make_input(tmp_path)
        MVPWorkflow(artifact_store=store).run(inp)

        prefixes = [str(prefix) for _, prefix in store.uploaded]
        assert any("training" in p for p in prefixes)
        assert any("evaluation" in p for p in prefixes)

    def test_noop_store_default_no_error(self, tmp_path):
        from agentic_mlops.workflows.mvp_workflow import MVPWorkflow

        inp = self._make_input(tmp_path)
        result = MVPWorkflow().run(inp)
        assert result is not None
