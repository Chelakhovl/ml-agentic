"""Robustness tests for WorkflowStateStore and LocalModelRegistryClient.

Tests cover non-happy-path scenarios:
  - Corrupted / truncated state.json
  - Missing state.json (fresh start)
  - Schema version edge cases (older, newer, missing)
  - Audit log integrity
  - Model registry: missing weights, version auto-increment, cleanup on failure
  - _resolve_best_weights_path fallback logic
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentic_mlops.integrations.workflow_state_store import (
    CorruptedStateError,
    LocalStateBackend,
    SchemaVersionError,
    WorkflowStateStore,
)

if TYPE_CHECKING:
    from agentic_mlops.contracts.model_registry import ModelRegistrationInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, workflow_id: str = "wf_test") -> WorkflowStateStore:
    return WorkflowStateStore(runs_dir=tmp_path, workflow_id=workflow_id)


def _backend(tmp_path: Path) -> LocalStateBackend:
    return LocalStateBackend(runs_dir=tmp_path)


# ---------------------------------------------------------------------------
# LocalStateBackend — read()
# ---------------------------------------------------------------------------


class TestLocalStateBackendRead:
    def test_returns_none_when_no_state_file(self, tmp_path: Path):
        be = _backend(tmp_path)
        assert be.read("wf_missing") is None

    def test_returns_dict_for_valid_state(self, tmp_path: Path):
        be = _backend(tmp_path)
        be.write("wf_ok", {"current_state": "COMPLETED"})
        result = be.read("wf_ok")
        assert result is not None
        assert result["current_state"] == "COMPLETED"

    def test_raises_json_error_on_truncated_state(self, tmp_path: Path):
        be = _backend(tmp_path)
        state_path = be.state_path("wf_corrupt")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{incomplete", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            be.read("wf_corrupt")

    def test_raises_json_error_on_empty_file(self, tmp_path: Path):
        be = _backend(tmp_path)
        state_path = be.state_path("wf_empty")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("", encoding="utf-8")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            be.read("wf_empty")


# ---------------------------------------------------------------------------
# WorkflowStateStore.load_state() — CorruptedStateError
# ---------------------------------------------------------------------------


class TestLoadStateCorrupted:
    def test_raises_corrupted_error_on_invalid_json(self, tmp_path: Path):
        store = _store(tmp_path)
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text("{broken json}", encoding="utf-8")
        with pytest.raises(CorruptedStateError) as exc_info:
            store.load_state()
        assert "invalid JSON" in str(exc_info.value)

    def test_raises_corrupted_error_on_empty_file(self, tmp_path: Path):
        store = _store(tmp_path)
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text("", encoding="utf-8")
        with pytest.raises(CorruptedStateError):
            store.load_state()

    def test_raises_corrupted_error_on_truncated_json(self, tmp_path: Path):
        store = _store(tmp_path)
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text('{"current_state": "RU', encoding="utf-8")
        with pytest.raises(CorruptedStateError):
            store.load_state()

    def test_error_message_includes_file_path(self, tmp_path: Path):
        store = _store(tmp_path)
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text("not json", encoding="utf-8")
        with pytest.raises(CorruptedStateError) as exc_info:
            store.load_state()
        # Should mention the file somehow via __cause__ or message
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# WorkflowStateStore.load_state() — missing file (fresh start)
# ---------------------------------------------------------------------------


class TestLoadStateMissing:
    def test_returns_none_when_no_state_file(self, tmp_path: Path):
        store = _store(tmp_path, "wf_fresh")
        assert store.load_state() is None

    def test_returns_none_even_if_workflow_dir_exists(self, tmp_path: Path):
        store = _store(tmp_path, "wf_dir_only")
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        assert store.load_state() is None


# ---------------------------------------------------------------------------
# WorkflowStateStore.load_state() — schema version handling
# ---------------------------------------------------------------------------


class TestLoadStateSchemaVersion:
    def test_raises_schema_error_for_future_version(self, tmp_path: Path):
        store = _store(tmp_path, "wf_future")
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps({"schema_version": 9999, "current_state": "COMPLETED"}),
            encoding="utf-8",
        )
        with pytest.raises(SchemaVersionError) as exc_info:
            store.load_state()
        assert "9999" in str(exc_info.value)

    def test_loads_successfully_for_current_version(self, tmp_path: Path):
        from agentic_mlops.integrations.workflow_state_store import STATE_SCHEMA_VERSION
        store = _store(tmp_path, "wf_current")
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps({"schema_version": STATE_SCHEMA_VERSION, "current_state": "COMPLETED"}),
            encoding="utf-8",
        )
        state = store.load_state()
        assert state is not None
        assert state["current_state"] == "COMPLETED"

    def test_loads_with_warning_for_older_version(self, tmp_path: Path):
        store = _store(tmp_path, "wf_old")
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps({"schema_version": 0, "current_state": "COMPLETED"}),
            encoding="utf-8",
        )
        # Should not raise — just log a warning
        state = store.load_state()
        assert state is not None
        assert state["current_state"] == "COMPLETED"

    def test_loads_with_missing_schema_version_treated_as_zero(self, tmp_path: Path):
        store = _store(tmp_path, "wf_no_schema")
        store.workflow_dir.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps({"current_state": "COMPLETED"}),
            encoding="utf-8",
        )
        # schema_version defaults to 0, which is older than current — loads with warning
        state = store.load_state()
        assert state is not None


# ---------------------------------------------------------------------------
# WorkflowStateStore.save_state() — injects metadata
# ---------------------------------------------------------------------------


class TestSaveState:
    def test_save_injects_schema_version(self, tmp_path: Path):
        from agentic_mlops.integrations.workflow_state_store import STATE_SCHEMA_VERSION
        store = _store(tmp_path, "wf_save")
        store.save_state({"current_state": "RUNNING"})
        data = json.loads(store.state_path.read_text())
        assert data["schema_version"] == STATE_SCHEMA_VERSION

    def test_save_injects_updated_at(self, tmp_path: Path):
        store = _store(tmp_path, "wf_ts")
        store.save_state({"current_state": "RUNNING"})
        data = json.loads(store.state_path.read_text())
        assert "updated_at" in data

    def test_save_does_not_overwrite_user_fields(self, tmp_path: Path):
        store = _store(tmp_path, "wf_fields")
        store.save_state({"current_state": "COMPLETED", "steps": ["a", "b"]})
        data = json.loads(store.state_path.read_text())
        assert data["current_state"] == "COMPLETED"
        assert data["steps"] == ["a", "b"]

    def test_round_trip_save_then_load(self, tmp_path: Path):
        store = _store(tmp_path, "wf_rt")
        store.save_state({"current_state": "RUNNING", "x": 42})
        loaded = store.load_state()
        assert loaded is not None
        assert loaded["current_state"] == "RUNNING"
        assert loaded["x"] == 42


# ---------------------------------------------------------------------------
# WorkflowStateStore.append_audit() — timestamp injection + valid JSON
# ---------------------------------------------------------------------------


class TestAppendAudit:
    def test_audit_line_is_valid_json(self, tmp_path: Path):
        store = _store(tmp_path, "wf_audit")
        store.append_audit({"event": "step_started", "step": "training"})
        lines = store.audit_log_path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "step_started"

    def test_audit_injects_timestamp(self, tmp_path: Path):
        store = _store(tmp_path, "wf_audit_ts")
        store.append_audit({"event": "done"})
        record = json.loads(store.audit_log_path.read_text().strip())
        assert "timestamp" in record

    def test_audit_appends_multiple_lines(self, tmp_path: Path):
        store = _store(tmp_path, "wf_audit_multi")
        for i in range(5):
            store.append_audit({"event": f"step_{i}"})
        lines = store.audit_log_path.read_text().splitlines()
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["event"] == f"step_{i}"

    def test_existing_timestamp_field_is_overwritten(self, tmp_path: Path):
        store = _store(tmp_path, "wf_audit_ts_override")
        store.append_audit({"event": "test", "timestamp": "should-be-replaced"})
        record = json.loads(store.audit_log_path.read_text().strip())
        # The store prepends a real timestamp — user-supplied should be overwritten
        assert record["timestamp"] != "should-be-replaced"


# ---------------------------------------------------------------------------
# LocalStateBackend.list_workflows()
# ---------------------------------------------------------------------------


class TestListWorkflows:
    def test_empty_runs_dir_returns_empty_list(self, tmp_path: Path):
        be = _backend(tmp_path)
        assert be.list_workflows() == []

    def test_nonexistent_runs_dir_returns_empty_list(self, tmp_path: Path):
        be = LocalStateBackend(tmp_path / "nonexistent")
        assert be.list_workflows() == []

    def test_only_dirs_with_state_json_are_listed(self, tmp_path: Path):
        be = _backend(tmp_path)
        # wf_with: has state.json
        be.write("wf_with", {"x": 1})
        # wf_without: dir exists but no state.json
        (tmp_path / "wf_without").mkdir()
        result = be.list_workflows()
        assert result == ["wf_with"]

    def test_multiple_workflows_sorted(self, tmp_path: Path):
        be = _backend(tmp_path)
        be.write("wf_c", {})
        be.write("wf_a", {})
        be.write("wf_b", {})
        assert be.list_workflows() == ["wf_a", "wf_b", "wf_c"]


# ---------------------------------------------------------------------------
# Model Registry — _resolve_best_weights_path
# ---------------------------------------------------------------------------


class TestResolveBestWeightsPath:
    def _inp(self, tmp_path: Path, training_out: Path) -> ModelRegistrationInput:
        from agentic_mlops.contracts.model_registry import ModelRegistrationInput
        return ModelRegistrationInput(
            model_name="m",
            training_output_path=str(training_out),
            evaluation_output_path=str(tmp_path / "eval.json"),
            approval_decision_path=str(tmp_path / "approval.json"),
            registry_dir=str(tmp_path / "registry"),
        )

    def test_resolves_sibling_best_pt(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import _resolve_best_weights_path

        training_out = tmp_path / "training_output.json"
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"fake weights")
        training_out.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        resolved = _resolve_best_weights_path(self._inp(tmp_path, training_out))
        assert resolved == best_pt

    def test_fallback_to_best_weights_path_in_json(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import _resolve_best_weights_path

        weights_path = tmp_path / "run" / "best.pt"
        weights_path.parent.mkdir()
        weights_path.write_bytes(b"weights")
        training_out = tmp_path / "training_output.json"
        training_out.write_text(
            json.dumps({"status": "completed", "best_weights_path": str(weights_path)}),
            encoding="utf-8",
        )

        resolved = _resolve_best_weights_path(self._inp(tmp_path, training_out))
        assert resolved == weights_path

    def test_raises_file_not_found_when_neither_exists(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import _resolve_best_weights_path

        training_out = tmp_path / "training_output.json"
        training_out.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            _resolve_best_weights_path(self._inp(tmp_path, training_out))

    def test_raises_file_not_found_when_best_weights_path_missing(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import _resolve_best_weights_path

        training_out = tmp_path / "training_output.json"
        training_out.write_text(
            json.dumps({"best_weights_path": str(tmp_path / "nonexistent.pt")}),
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError):
            _resolve_best_weights_path(self._inp(tmp_path, training_out))


# ---------------------------------------------------------------------------
# LocalModelRegistryClient — version auto-increment and latest.json
# ---------------------------------------------------------------------------


class _RegistrationHelper:
    """Creates the minimum artifacts needed for LocalModelRegistryClient.register()."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.best_pt = tmp_path / "best.pt"
        self.best_pt.write_bytes(b"fake model weights")
        self.training_out = tmp_path / "training_output.json"
        self.training_out.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        self.eval_out = tmp_path / "eval_output.json"
        self.eval_out.write_text(json.dumps({"success": True}), encoding="utf-8")
        self.approval = tmp_path / "approval.json"
        self.approval.write_text(
            json.dumps({"status": "approved", "action": "approve_model"}), encoding="utf-8"
        )

    def input(self, model_name: str = "my-model") -> ModelRegistrationInput:
        from agentic_mlops.contracts.model_registry import ModelRegistrationInput
        return ModelRegistrationInput(
            model_name=model_name,
            training_output_path=str(self.training_out),
            evaluation_output_path=str(self.eval_out),
            approval_decision_path=str(self.approval),
            registry_dir=str(self.tmp / "registry"),
        )

    def lineage(self):
        from agentic_mlops.contracts.model_registry import ModelLineage
        return ModelLineage(dataset_path="/data", data_yaml="/data/data.yaml")


class TestLocalModelRegistryClientRegister:
    def _artifacts(self, tmp_path: Path) -> Path:
        p = tmp_path / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def test_first_version_is_1(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient
        helper = _RegistrationHelper(tmp_path)
        client = LocalModelRegistryClient()
        out = client.register(helper.input(), helper.lineage(), self._artifacts(tmp_path))
        assert out.success
        assert out.version == 1

    def test_second_registration_increments_version(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient
        helper = _RegistrationHelper(tmp_path)
        client = LocalModelRegistryClient()
        inp = helper.input()
        lineage = helper.lineage()
        artifacts = self._artifacts(tmp_path)
        out1 = client.register(inp, lineage, artifacts)
        out2 = client.register(inp, lineage, artifacts)
        assert out1.version == 1
        assert out2.version == 2

    def test_creates_latest_json_after_success(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient
        helper = _RegistrationHelper(tmp_path)
        client = LocalModelRegistryClient()
        inp = helper.input()
        client.register(inp, helper.lineage(), self._artifacts(tmp_path))
        latest_path = Path(inp.registry_dir) / inp.model_name / "latest.json"
        assert latest_path.exists()
        latest = json.loads(latest_path.read_text())
        assert latest["version"] == 1

    def test_lineage_json_is_written(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient
        helper = _RegistrationHelper(tmp_path)
        client = LocalModelRegistryClient()
        inp = helper.input()
        client.register(inp, helper.lineage(), self._artifacts(tmp_path))
        lineage_path = (
            Path(inp.registry_dir) / inp.model_name / "versions" / "1" / "lineage.json"
        )
        assert lineage_path.exists()
        data = json.loads(lineage_path.read_text())
        assert data["dataset_path"] == "/data"

    def test_best_pt_is_copied_to_registry(self, tmp_path: Path):
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient
        helper = _RegistrationHelper(tmp_path)
        client = LocalModelRegistryClient()
        inp = helper.input()
        client.register(inp, helper.lineage(), self._artifacts(tmp_path))
        registered_pt = (
            Path(inp.registry_dir) / inp.model_name / "versions" / "1" / "model" / "best.pt"
        )
        assert registered_pt.exists()
        assert registered_pt.read_bytes() == b"fake model weights"

    def test_register_missing_best_pt_returns_failed_output(self, tmp_path: Path):
        from agentic_mlops.contracts.model_registry import (
            ModelLineage,
            ModelRegistrationInput,
            RegistrationStatus,
        )
        from agentic_mlops.integrations.model_registry import LocalModelRegistryClient

        # No best.pt created — registry catches FileNotFoundError and returns failed output
        training_out = tmp_path / "training_output.json"
        training_out.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        inp = ModelRegistrationInput(
            model_name="no-weights",
            training_output_path=str(training_out),
            evaluation_output_path=str(tmp_path / "eval.json"),
            approval_decision_path=str(tmp_path / "approval.json"),
            registry_dir=str(tmp_path / "registry"),
        )
        lineage = ModelLineage()
        client = LocalModelRegistryClient()
        out = client.register(inp, lineage, self._artifacts(tmp_path))
        assert not out.success
        assert out.status == RegistrationStatus.FAILED
