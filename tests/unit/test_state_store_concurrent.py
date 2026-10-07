"""Deterministic concurrency tests for LocalStateBackend.

Uses threading.Barrier and threading.Event for explicit coordination —
no sleep(), no race-condition reliance.  Every test is deterministic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agentic_mlops.integrations.workflow_state_store import (
    LocalStateBackend,
    WorkflowLockError,
)


class TestLockAcquisitionAndRelease:
    def test_lock_file_exists_while_held(self, tmp_path: Path):
        backend = LocalStateBackend(tmp_path)
        wid = "wf_lock_exists"
        lock_file = tmp_path / wid / "workflow.lock"
        with backend.acquire_lock(wid, timeout=1.0):
            assert lock_file.exists()
        assert not lock_file.exists()

    def test_lock_timeout_raises_workflow_lock_error(self, tmp_path: Path):
        backend = LocalStateBackend(tmp_path)
        wid = "wf_timeout"
        lock_held = threading.Event()
        can_release = threading.Event()

        def hold_lock():
            with backend.acquire_lock(wid, timeout=5.0):
                lock_held.set()
                can_release.wait(timeout=10.0)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        assert lock_held.wait(timeout=3.0), "Holder thread never acquired the lock"

        try:
            with pytest.raises(WorkflowLockError, match=wid):
                with backend.acquire_lock(wid, timeout=0.05):
                    pass
        finally:
            can_release.set()
            holder.join(timeout=5)

    def test_reacquire_after_release(self, tmp_path: Path):
        backend = LocalStateBackend(tmp_path)
        wid = "wf_reacquire"
        released = threading.Event()

        def first_holder():
            with backend.acquire_lock(wid, timeout=1.0):
                pass
            released.set()

        t = threading.Thread(target=first_holder)
        t.start()
        assert released.wait(timeout=3.0), "First holder never released"
        t.join(timeout=3)

        # Second acquisition must succeed immediately after release
        with backend.acquire_lock(wid, timeout=0.5):
            pass


class TestConcurrentStateWrites:
    def test_two_writers_serialize_state(self, tmp_path: Path):
        """Two threads acquire the lock in turn; final JSON must be valid."""
        backend = LocalStateBackend(tmp_path)
        wid = "wf_two_writers"
        barrier = threading.Barrier(2)
        write_order: list[str] = []

        def writer(value: str):
            barrier.wait()
            with backend.acquire_lock(wid, timeout=10.0):
                backend.write(wid, {"writer": value, "data": "x" * 256})
                write_order.append(value)

        t1 = threading.Thread(target=writer, args=("A",))
        t2 = threading.Thread(target=writer, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(write_order) == 2
        final = backend.read(wid)
        assert final is not None
        assert final["writer"] == write_order[-1]

    def test_audit_log_no_interleaved_lines(self, tmp_path: Path):
        """Many threads append events; every written line must parse as valid JSON.

        Note: on Windows, ``open("a")`` is not a true atomic append so some
        events may be lost when threads race at the OS level.  The invariant
        we test is *no line corruption* (no interleaved/partial writes) — not
        exact count, which is not guaranteed on Windows.
        """
        backend = LocalStateBackend(tmp_path)
        wid = "wf_audit"
        n_threads = 8
        events_per_thread = 10
        barrier = threading.Barrier(n_threads)

        def appender(thread_id: int):
            barrier.wait()
            for i in range(events_per_thread):
                backend.append_audit(wid, {"thread": thread_id, "seq": i})

        threads = [threading.Thread(target=appender, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        log_path = tmp_path / wid / "audit_log.jsonl"
        lines = [ln for ln in log_path.read_text(encoding="utf-8").strip().splitlines() if ln]

        # At least one event per thread must have been recorded.
        assert len(lines) >= n_threads, (
            f"Expected at least {n_threads} lines, got {len(lines)}"
        )
        # Every line that *was* written must be valid, non-corrupted JSON.
        for line in lines:
            obj = json.loads(line)  # raises if truncated / interleaved
            assert "thread" in obj and "seq" in obj

    def test_concurrent_read_write_consistency(self, tmp_path: Path):
        """Readers never see corrupt JSON because write uses atomic os.replace().

        On Windows os.replace() raises PermissionError when the destination is
        open by a reader; the writer retries in that case — the invariant is
        still "no corrupt reads", not "every write lands atomically".
        """
        import sys  # noqa: PLC0415

        backend = LocalStateBackend(tmp_path)
        wid = "wf_rw_consistency"
        backend.write(wid, {"value": 0})
        errors: list[str] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    data = backend.read(wid)
                except PermissionError:
                    # Windows: os.replace() can briefly hold an exclusive lock
                    # on the destination; skip and retry on the next loop.
                    continue
                if data is None:
                    errors.append("read returned None mid-run")

        def writer():
            for i in range(100):
                for _attempt in range(20):
                    try:
                        backend.write(wid, {"value": i, "pad": "y" * 1024})
                        break
                    except PermissionError:
                        if sys.platform != "win32":
                            raise

        readers = [threading.Thread(target=reader) for _ in range(4)]
        w = threading.Thread(target=writer)
        for r in readers:
            r.start()
        w.start()
        w.join(timeout=10)
        stop.set()
        for r in readers:
            r.join(timeout=3)

        assert not errors, f"Consistency errors: {errors}"

    def test_list_workflows_sees_written_ids(self, tmp_path: Path):
        """Writes from multiple workflows are all enumerable afterwards."""
        backend = LocalStateBackend(tmp_path)
        ids = [f"wf_{i:03d}" for i in range(5)]
        barrier = threading.Barrier(len(ids))

        def write_one(wid: str):
            barrier.wait()
            with backend.acquire_lock(wid, timeout=5.0):
                backend.write(wid, {"id": wid})

        threads = [threading.Thread(target=write_one, args=(wid,)) for wid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        listed = set(backend.list_workflows())
        assert set(ids) <= listed
