"""Tests for WorkflowStateStore file locking and pluggable StateBackend."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from agentic_mlops.integrations.workflow_state_store import (
    LocalStateBackend,
    SchemaVersionError,
    WorkflowLockError,
    WorkflowStateStore,
    _pid_alive,
)


# ---------------------------------------------------------------------------
# _pid_alive helpers
# ---------------------------------------------------------------------------


class TestPidAlive:
    def test_current_process_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        # PID 0 is never a regular process; PID 999999 is very unlikely.
        assert _pid_alive(999_999_999) is False


# ---------------------------------------------------------------------------
# LocalStateBackend — read / write / append_audit / list_workflows
# ---------------------------------------------------------------------------


class TestLocalStateBackendIO:
    def test_read_returns_none_for_missing(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        assert backend.read("wf_x") is None

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        state = {"workflow_id": "wf1", "status": "running"}
        backend.write("wf1", state)
        result = backend.read("wf1")
        assert result is not None
        assert result["workflow_id"] == "wf1"

    def test_write_is_atomic_via_tmp(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        backend.write("wf_atom", {"x": 1})
        # The .tmp file must not linger after a successful write.
        tmp_file = tmp_path / "wf_atom" / "state.json.tmp"
        assert not tmp_file.exists()

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        backend.write("wf2", {"v": 1})
        backend.write("wf2", {"v": 2})
        assert backend.read("wf2")["v"] == 2

    def test_append_audit_creates_jsonl(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        backend.append_audit("wf3", {"event": "started"})
        backend.append_audit("wf3", {"event": "finished"})
        lines = (tmp_path / "wf3" / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "started"
        assert json.loads(lines[1])["event"] == "finished"

    def test_list_workflows_empty(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        assert backend.list_workflows() == []

    def test_list_workflows_returns_ids_with_state_json(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        backend.write("wf_a", {"x": 1})
        backend.write("wf_b", {"x": 2})
        # A directory without state.json should NOT appear.
        (tmp_path / "orphan").mkdir()
        ids = backend.list_workflows()
        assert sorted(ids) == ["wf_a", "wf_b"]


# ---------------------------------------------------------------------------
# LocalStateBackend — acquire_lock
# ---------------------------------------------------------------------------


class TestLocalStateBackendLock:
    def test_lock_acquired_and_released(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        lock_path = tmp_path / "wf_l" / "workflow.lock"

        with backend.acquire_lock("wf_l", timeout=5.0):
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_lock_file_contains_pid(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        with backend.acquire_lock("wf_pid", timeout=5.0):
            pid = int((tmp_path / "wf_pid" / "workflow.lock").read_text())
            assert pid == os.getpid()

    def test_second_acquire_within_timeout_raises(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        with backend.acquire_lock("wf_contend", timeout=5.0):
            with pytest.raises(WorkflowLockError):
                with backend.acquire_lock("wf_contend", timeout=0.2):
                    pass

    def test_stale_lock_removed_when_pid_dead(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        wf_dir = tmp_path / "wf_stale"
        wf_dir.mkdir(parents=True, exist_ok=True)
        # Write a lock file with a PID that definitely doesn't exist.
        (wf_dir / "workflow.lock").write_text("999999999")

        # Should succeed because the PID is not alive.
        with backend.acquire_lock("wf_stale", timeout=2.0):
            assert (wf_dir / "workflow.lock").exists()

    def test_lock_released_on_exception(self, tmp_path: Path) -> None:
        backend = LocalStateBackend(tmp_path)
        lock_path = tmp_path / "wf_exc" / "workflow.lock"

        with pytest.raises(RuntimeError):
            with backend.acquire_lock("wf_exc", timeout=5.0):
                raise RuntimeError("boom")

        assert not lock_path.exists()

    def test_concurrent_threads_serialize(self, tmp_path: Path) -> None:
        """Two threads must not hold the same lock at the same time."""
        backend = LocalStateBackend(tmp_path)
        results: list[str] = []
        errors: list[Exception] = []

        def worker(name: str) -> None:
            try:
                with backend.acquire_lock("wf_threads", timeout=5.0):
                    results.append(f"enter:{name}")
                    time.sleep(0.05)
                    results.append(f"exit:{name}")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, errors
        # Both threads should enter and exit, but never overlap.
        assert len(results) == 4
        # Find whoever went first and verify non-overlapping interleave.
        first_enter = results[0]  # e.g. "enter:A"
        first_exit_idx = results.index(first_enter.replace("enter:", "exit:"))
        assert first_exit_idx == 1, f"Unexpected interleave: {results}"


# ---------------------------------------------------------------------------
# WorkflowStateStore — schema-version checks
# ---------------------------------------------------------------------------


class TestWorkflowStateStoreSchema:
    def test_load_state_none_when_missing(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_miss")
        assert store.load_state() is None

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_rt")
        store.save_state({"status": "running"})
        state = store.load_state()
        assert state is not None
        assert state["status"] == "running"
        assert "schema_version" in state

    def test_save_injects_updated_at(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_ts")
        store.save_state({"x": 1})
        state = store.load_state()
        assert "updated_at" in state

    def test_load_raises_for_newer_schema(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_new_schema")
        store.save_state({"x": 1})
        # Manually bump schema_version in the file.
        raw = json.loads(store.state_path.read_text(encoding="utf-8"))
        raw["schema_version"] = 9999
        store.state_path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SchemaVersionError):
            store.load_state()

    def test_load_warns_for_older_schema(self, tmp_path: Path, caplog) -> None:
        store = WorkflowStateStore(tmp_path, "wf_old_schema")
        store.save_state({"x": 1})
        raw = json.loads(store.state_path.read_text(encoding="utf-8"))
        raw["schema_version"] = 0
        store.state_path.write_text(json.dumps(raw), encoding="utf-8")
        import logging

        with caplog.at_level(logging.WARNING):
            result = store.load_state()
        assert result is not None
        assert any("best-effort" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# WorkflowStateStore — acquire_lock convenience wrapper
# ---------------------------------------------------------------------------


class TestWorkflowStateStoreLock:
    def test_acquire_lock_delegates_to_backend(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_del", lock_timeout_seconds=5.0)
        lock_path = tmp_path / "wf_del" / "workflow.lock"
        with store.acquire_lock():
            assert lock_path.exists()
        assert not lock_path.exists()

    def test_acquire_lock_uses_custom_timeout(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_to", lock_timeout_seconds=0.1)
        with store.acquire_lock():
            with pytest.raises(WorkflowLockError):
                with store.acquire_lock(timeout=0.1):
                    pass

    def test_acquire_lock_explicit_timeout_overrides_default(self, tmp_path: Path) -> None:
        store = WorkflowStateStore(tmp_path, "wf_ov", lock_timeout_seconds=5.0)
        with store.acquire_lock():
            with pytest.raises(WorkflowLockError):
                # Explicit 0.1s should override the 5s default.
                with store.acquire_lock(timeout=0.1):
                    pass


# ---------------------------------------------------------------------------
# WorkflowStateStore — custom backend injection
# ---------------------------------------------------------------------------


class FakeStateBackend:
    """In-memory backend for unit-testing WorkflowStateStore."""

    def __init__(self) -> None:
        self._states: dict[str, dict] = {}
        self._audits: dict[str, list] = {}
        self.locked: bool = False

    def read(self, workflow_id: str):
        return self._states.get(workflow_id)

    def write(self, workflow_id: str, state: dict) -> None:
        self._states[workflow_id] = state

    def append_audit(self, workflow_id: str, event: dict) -> None:
        self._audits.setdefault(workflow_id, []).append(event)

    def list_workflows(self) -> list[str]:
        return sorted(self._states.keys())

    def acquire_lock(self, workflow_id: str, timeout: float):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self.locked = True
            try:
                yield
            finally:
                self.locked = False

        return _ctx()


class TestCustomBackendInjection:
    def test_save_state_uses_injected_backend(self, tmp_path: Path) -> None:
        backend = FakeStateBackend()
        store = WorkflowStateStore(tmp_path, "wf_fake", backend=backend)
        store.save_state({"status": "running"})
        assert "wf_fake" in backend._states
        assert backend._states["wf_fake"]["status"] == "running"

    def test_load_state_reads_from_backend(self, tmp_path: Path) -> None:
        backend = FakeStateBackend()
        backend._states["wf_fake2"] = {"status": "completed", "schema_version": 1}
        store = WorkflowStateStore(tmp_path, "wf_fake2", backend=backend)
        state = store.load_state()
        assert state is not None
        assert state["status"] == "completed"

    def test_append_audit_uses_backend(self, tmp_path: Path) -> None:
        backend = FakeStateBackend()
        store = WorkflowStateStore(tmp_path, "wf_fake3", backend=backend)
        store.append_audit({"event": "started"})
        assert len(backend._audits["wf_fake3"]) == 1
        assert backend._audits["wf_fake3"][0]["event"] == "started"

    def test_acquire_lock_uses_backend(self, tmp_path: Path) -> None:
        backend = FakeStateBackend()
        store = WorkflowStateStore(tmp_path, "wf_fake4", backend=backend)
        assert not backend.locked
        with store.acquire_lock():
            assert backend.locked
        assert not backend.locked

    def test_list_all_delegates_to_backend(self, tmp_path: Path) -> None:
        backend = FakeStateBackend()
        backend._states["wf_a"] = {}
        backend._states["wf_b"] = {}
        store = WorkflowStateStore(tmp_path, "wf_a", backend=backend)
        assert store.list_all() == ["wf_a", "wf_b"]


# ---------------------------------------------------------------------------
# OrchestratorInput — new fields
# ---------------------------------------------------------------------------


class TestOrchestratorInputNewFields:
    def test_defaults(self) -> None:
        from agentic_mlops.contracts.orchestrator import OrchestratorInput

        inp = OrchestratorInput(
            workflow_id="wf_defaults",
            dataset_path="/tmp/ds",
            data_yaml_path="/tmp/ds/data.yaml",
        )
        assert inp.lock_timeout_seconds == 30.0
        assert inp.state_backend == "local"
        assert inp.state_blob_container is None

    def test_custom_lock_timeout(self) -> None:
        from agentic_mlops.contracts.orchestrator import OrchestratorInput

        inp = OrchestratorInput(
            workflow_id="wf_lt",
            lock_timeout_seconds=60.0,
            dataset_path="/tmp/ds",
            data_yaml_path="/tmp/ds/data.yaml",
        )
        assert inp.lock_timeout_seconds == 60.0

    def test_azure_blob_backend_fields(self) -> None:
        from agentic_mlops.contracts.orchestrator import OrchestratorInput

        inp = OrchestratorInput(
            workflow_id="wf_az",
            state_backend="azure_blob",
            state_blob_container="my-container",
            dataset_path="/tmp/ds",
            data_yaml_path="/tmp/ds/data.yaml",
        )
        assert inp.state_backend == "azure_blob"
        assert inp.state_blob_container == "my-container"
