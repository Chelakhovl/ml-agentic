"""Workflow state store for the Orchestrator — a local JSON state file + JSONL audit log.

Per agentic_mlops_workflow_docs/agents/00_orchestrator_agent.md's "MVP
implementation" note: no networked State Store / Audit Log service exists by
default — just two files per workflow run:

    runs/<workflow_id>/state.json        current state snapshot (overwritten each step)
    runs/<workflow_id>/audit_log.jsonl   append-only event history, one JSON object per line

The I/O is delegated to a ``StateBackend`` (protocol).  Two implementations
ship out of the box:

* ``LocalStateBackend``      — filesystem (the default, no extra deps)
* ``AzureBlobStateBackend``  — Azure Blob Storage (requires ``[azure]`` extra)

``WorkflowStateStore`` is a thin facade that adds schema-version checking,
timestamp injection, structured logging, and a cross-platform lock — it
delegates every raw I/O call to its ``_backend``.

Concurrency safety
------------------
``WorkflowStateStore.acquire_lock()`` returns a context manager that must be
held for the duration of a workflow run.  This prevents two concurrent
``run-workflow`` invocations on the same ``workflow_id`` from interleaving
their state writes.

* ``LocalStateBackend`` — lockfile in the workflow directory (``workflow.lock``).
  Uses ``O_CREAT | O_EXCL`` for atomic creation and checks whether the PID in
  a stale lockfile is still alive.
* ``AzureBlobStateBackend`` — creates a lock blob (``workflow.lock``) using
  ``upload_blob(overwrite=False)``; treats blobs older than ``2 × timeout``
  seconds as stale.

``save_state`` in ``LocalStateBackend`` writes atomically via a ``.tmp``
file + ``os.replace()``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

# Increment this when the structure of state.json changes in a
# backwards-incompatible way.  load_state() will refuse to resume runs whose
# stored version is *newer* than this constant.
STATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SchemaVersionError(RuntimeError):
    """Raised when a state.json was written by an incompatible code version."""


class WorkflowLockError(RuntimeError):
    """Raised when the workflow lock cannot be acquired within the timeout."""


class CorruptedStateError(RuntimeError):
    """Raised when state.json exists but contains invalid JSON (truncated/corrupted write)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID appears to be running."""
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415

            PROCESS_QUERY_INFORMATION = 0x0400
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        # POSIX: signal 0 checks existence without delivering a signal.
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# StateBackend protocol
# ---------------------------------------------------------------------------


class StateBackend:
    """Structural protocol for workflow state backends.

    Any object that implements all five methods below is a valid backend —
    no explicit inheritance required.
    """

    def read(self, workflow_id: str) -> dict[str, Any] | None:
        """Return the stored state dict, or None if it doesn't exist."""
        raise NotImplementedError

    def write(self, workflow_id: str, state: dict[str, Any]) -> None:
        """Persist *state* for *workflow_id* (create or overwrite)."""
        raise NotImplementedError

    def append_audit(self, workflow_id: str, event: dict[str, Any]) -> None:
        """Append *event* as a single JSON line to the audit log."""
        raise NotImplementedError

    def list_workflows(self) -> list[str]:
        """Return a list of all workflow IDs known to this backend."""
        raise NotImplementedError

    @contextlib.contextmanager
    def acquire_lock(self, workflow_id: str, timeout: float) -> Iterator[None]:
        """Acquire an exclusive lock for *workflow_id*, hold it for the block."""
        raise NotImplementedError
        yield  # make it a generator — subclasses must override


# ---------------------------------------------------------------------------
# LocalStateBackend
# ---------------------------------------------------------------------------


class LocalStateBackend(StateBackend):
    """Filesystem backend — state.json + audit_log.jsonl per workflow run directory."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = Path(runs_dir)

    # -- internal path helpers ------------------------------------------

    def workflow_dir(self, workflow_id: str) -> Path:
        return self._runs_dir / workflow_id

    def state_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "state.json"

    def audit_log_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "audit_log.jsonl"

    def _lock_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "workflow.lock"

    # -- StateBackend implementation ------------------------------------

    def read(self, workflow_id: str) -> dict[str, Any] | None:
        path = self.state_path(workflow_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, workflow_id: str, state: dict[str, Any]) -> None:
        wdir = self.workflow_dir(workflow_id)
        wdir.mkdir(parents=True, exist_ok=True)
        path = self.state_path(workflow_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)  # atomic on POSIX; best-effort on Windows

    def append_audit(self, workflow_id: str, event: dict[str, Any]) -> None:
        wdir = self.workflow_dir(workflow_id)
        wdir.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path(workflow_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def list_workflows(self) -> list[str]:
        if not self._runs_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._runs_dir.iterdir()
            if d.is_dir() and (d / "state.json").exists()
        )

    @contextlib.contextmanager
    def acquire_lock(self, workflow_id: str, timeout: float) -> Iterator[None]:
        lock_path = self._lock_path(workflow_id)
        self.workflow_dir(workflow_id).mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout

        while True:
            try:
                # O_CREAT | O_EXCL is atomic on all major OS/FS combinations.
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode())
                finally:
                    os.close(fd)
                break  # lock acquired
            except FileExistsError:
                # Check whether the existing lock is stale (owner process dead).
                try:
                    pid_str = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if pid_str and not _pid_alive(int(pid_str)):
                        lock_path.unlink(missing_ok=True)
                        continue  # retry immediately
                except (ValueError, OSError):
                    pass

                if time.monotonic() >= deadline:
                    raise WorkflowLockError(
                        f"Could not acquire lock for workflow '{workflow_id}' within "
                        f"{timeout}s. Another process may be running it "
                        f"(lock file: {lock_path})."
                    )
                time.sleep(0.1)

        try:
            yield
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# AzureBlobStateBackend
# ---------------------------------------------------------------------------


class AzureBlobStateBackend(StateBackend):
    """Azure Blob Storage backend — state.json + audit_log.jsonl in a blob container.

    ``connection_string`` is the Azure Storage account connection string (or
    pass ``account_url`` + a ``DefaultAzureCredential`` by subclassing).
    ``container_name`` is created on first use if it doesn't exist.

    Locking uses a dedicated lock blob (``<workflow_id>/workflow.lock``)
    created with ``overwrite=False`` so creation is effectively atomic; blobs
    older than ``2 × timeout`` seconds are treated as stale.

    Audit log appends use Azure Append Blobs so each call is an atomic server-
    side append — no download-modify-upload race.

    Requires: ``pip install 'agentic-mlops[azure]'``
    """

    def __init__(self, connection_string: str, container_name: str) -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AzureBlobStateBackend requires 'azure-storage-blob'. "
                "Install it with: pip install 'agentic-mlops[azure]'"
            ) from exc

        self._container_name = container_name
        self._svc: Any = BlobServiceClient.from_connection_string(connection_string)
        # Best-effort container creation — ignore "already exists" errors.
        try:
            self._svc.create_container(container_name)
        except Exception:  # ResourceExistsError or any auth/network issue
            pass

    def _blob_client(self, blob_name: str) -> Any:
        return self._svc.get_blob_client(container=self._container_name, blob=blob_name)

    def _state_blob(self, workflow_id: str) -> str:
        return f"{workflow_id}/state.json"

    def _audit_blob(self, workflow_id: str) -> str:
        return f"{workflow_id}/audit_log.jsonl"

    def _lock_blob(self, workflow_id: str) -> str:
        return f"{workflow_id}/workflow.lock"

    # -- StateBackend implementation ------------------------------------

    def read(self, workflow_id: str) -> dict[str, Any] | None:
        try:
            data = self._blob_client(self._state_blob(workflow_id)).download_blob().readall()
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def write(self, workflow_id: str, state: dict[str, Any]) -> None:
        data = json.dumps(state, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        self._blob_client(self._state_blob(workflow_id)).upload_blob(data, overwrite=True)

    def append_audit(self, workflow_id: str, event: dict[str, Any]) -> None:
        from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415

        line = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        blob = self._blob_client(self._audit_blob(workflow_id))
        try:
            blob.append_block(line)
        except (ResourceNotFoundError, Exception):
            # Blob doesn't exist yet as an append blob — create it first.
            try:
                blob.create_append_blob()
                blob.append_block(line)
            except Exception:
                # Last resort: download + overwrite (no concurrent-append safety).
                try:
                    existing = blob.download_blob().readall()
                except Exception:
                    existing = b""
                blob.upload_blob(existing + line, overwrite=True)

    def list_workflows(self) -> list[str]:
        try:
            cc = self._svc.get_container_client(self._container_name)
            ids: set[str] = set()
            for b in cc.list_blobs():
                parts = b.name.split("/", 1)
                if len(parts) == 2 and parts[1] == "state.json":
                    ids.add(parts[0])
            return sorted(ids)
        except Exception:
            return []

    @contextlib.contextmanager
    def acquire_lock(self, workflow_id: str, timeout: float) -> Iterator[None]:
        lock_name = self._lock_blob(workflow_id)
        deadline = time.monotonic() + timeout
        content = json.dumps({"pid": os.getpid(), "ts": time.time()}).encode()

        while True:
            blob = self._blob_client(lock_name)
            try:
                blob.upload_blob(content, overwrite=False)
                break  # lock acquired
            except Exception:
                # Check whether the existing lock is stale.
                try:
                    lock_data = json.loads(blob.download_blob().readall())
                    if time.time() - lock_data.get("ts", 0) > timeout * 2:
                        blob.delete_blob()
                        continue  # retry immediately
                except Exception:
                    pass

                if time.monotonic() >= deadline:
                    raise WorkflowLockError(
                        f"Could not acquire Azure Blob lock for workflow '{workflow_id}' "
                        f"within {timeout}s."
                    )
                time.sleep(0.5)

        try:
            yield
        finally:
            try:
                blob.delete_blob()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# WorkflowStateStore — thin facade
# ---------------------------------------------------------------------------


def _make_local_backend(runs_dir: Path) -> LocalStateBackend:
    return LocalStateBackend(runs_dir)


class WorkflowStateStore:
    """Reads/writes state.json and appends to audit_log.jsonl for one workflow_id.

    Delegates all raw I/O to a ``StateBackend`` (default: ``LocalStateBackend``).
    Exposes ``workflow_dir``, ``state_path``, ``audit_log_path`` as *local*
    filesystem paths — these are used by the orchestrator for artifact
    directories and error messages even when a remote backend is in use (step
    artifacts are always written locally first).
    """

    def __init__(
        self,
        runs_dir: Path,
        workflow_id: str,
        lock_timeout_seconds: float = 30.0,
        backend: StateBackend | None = None,
    ) -> None:
        self.workflow_dir = Path(runs_dir) / workflow_id
        self.state_path = self.workflow_dir / "state.json"
        self.audit_log_path = self.workflow_dir / "audit_log.jsonl"
        self._workflow_id = workflow_id
        self._lock_timeout = lock_timeout_seconds
        self._backend: StateBackend = backend if backend is not None else LocalStateBackend(
            Path(runs_dir)
        )

    # -- public API ---------------------------------------------------------

    def load_state(self) -> dict[str, Any] | None:
        try:
            state = self._backend.read(self._workflow_id)
        except (json.JSONDecodeError, ValueError) as exc:
            raise CorruptedStateError(
                f"{self.state_path} contains invalid JSON and cannot be loaded. "
                f"The file may have been truncated or corrupted during a previous write. "
                f"Error: {exc}"
            ) from exc
        if state is None:
            return None
        stored_version = state.get("schema_version", 0)
        if stored_version > STATE_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{self.state_path} was written by a newer version of agentic-mlops "
                f"(schema_version={stored_version}); current code supports up to "
                f"schema_version={STATE_SCHEMA_VERSION}. Upgrade agentic-mlops to resume "
                f"this run."
            )
        if stored_version < STATE_SCHEMA_VERSION:
            logger.warning(
                "state.json schema_version=%d is older than current=%d; "
                "loading with best-effort compatibility",
                stored_version,
                STATE_SCHEMA_VERSION,
                extra={"workflow_id": self._workflow_id},
            )
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        state["schema_version"] = STATE_SCHEMA_VERSION
        state["updated_at"] = datetime.now(tz=UTC).isoformat()
        self._backend.write(self._workflow_id, state)

    def append_audit(self, event: dict[str, Any]) -> None:
        event = {**event, "timestamp": datetime.now(tz=UTC).isoformat()}
        self._backend.append_audit(self._workflow_id, event)
        # "message"/"asctime" are reserved LogRecord attribute names — logging
        # a dict that contains one as `extra` raises a KeyError.
        log_extra = {("detail" if k == "message" else k): v for k, v in event.items()}
        logger.info("Workflow audit event", extra=log_extra)

    @contextlib.contextmanager
    def acquire_lock(self, timeout: float | None = None) -> Iterator[None]:
        """Acquire an exclusive lock for this workflow's run directory.

        Raises ``WorkflowLockError`` if the lock cannot be acquired within
        *timeout* seconds (defaults to ``lock_timeout_seconds`` passed to
        ``__init__``).
        """
        t = timeout if timeout is not None else self._lock_timeout
        with self._backend.acquire_lock(self._workflow_id, t):
            yield

    def list_all(self) -> list[str]:
        """Return all workflow IDs known to the backend."""
        return self._backend.list_workflows()
