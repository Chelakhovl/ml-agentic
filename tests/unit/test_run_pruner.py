"""Unit tests for WorkflowPruner and the prune-runs CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentic_mlops.tools.run_pruner import WorkflowPruner

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_state(
    runs_dir: Path,
    workflow_id: str,
    *,
    status: str = "completed",
    started_at: str | None = None,
) -> None:
    d = runs_dir / workflow_id
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": workflow_id,
        "status": status,
        "started_at": started_at or "2026-08-25T10:00:00+00:00",
    }
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _days_ago(n: float) -> str:
    dt = datetime.now(tz=UTC) - timedelta(days=n)
    return dt.isoformat()


# ── WorkflowPruner unit tests ─────────────────────────────────────────────────


class TestPrunerValidation:
    def test_keep_last_and_older_than_raises(self):
        with pytest.raises(ValueError, match="OR"):
            WorkflowPruner(keep_last=3, older_than_days=7)

    def test_negative_keep_last_raises(self):
        with pytest.raises(ValueError, match="keep_last"):
            WorkflowPruner(keep_last=-1)

    def test_neither_raises_only_at_prune_time_via_cli(self, tmp_path):
        # WorkflowPruner itself doesn't raise if neither is given; CLI validates this
        p = WorkflowPruner()
        result = p.prune(tmp_path)
        assert result.deleted_count == 0

    def test_nonexistent_runs_dir_returns_empty_result(self, tmp_path):
        p = WorkflowPruner(keep_last=5)
        result = p.prune(tmp_path / "no_such_dir")
        assert result.deleted_count == 0
        assert not result.kept


class TestKeepLast:
    def test_keep_last_zero_deletes_all(self, tmp_path):
        for wf in ("wf_1", "wf_2", "wf_3"):
            _write_state(tmp_path, wf)
        result = WorkflowPruner(keep_last=0).prune(tmp_path)
        assert result.deleted_count == 3
        assert not result.kept

    def test_keep_last_keeps_newest(self, tmp_path):
        _write_state(tmp_path, "wf_old", started_at=_days_ago(10))
        _write_state(tmp_path, "wf_mid", started_at=_days_ago(5))
        _write_state(tmp_path, "wf_new", started_at=_days_ago(1))
        result = WorkflowPruner(keep_last=2).prune(tmp_path)
        assert "wf_old" in result.deleted
        assert "wf_mid" in result.kept
        assert "wf_new" in result.kept
        assert result.deleted_count == 1

    def test_keep_last_larger_than_count_deletes_nothing(self, tmp_path):
        _write_state(tmp_path, "wf_1")
        _write_state(tmp_path, "wf_2")
        result = WorkflowPruner(keep_last=10).prune(tmp_path)
        assert result.deleted_count == 0
        assert len(result.kept) == 2

    def test_keep_last_exactly_N_items(self, tmp_path):
        for i in range(5):
            _write_state(tmp_path, f"wf_{i:02d}", started_at=_days_ago(5 - i))
        result = WorkflowPruner(keep_last=3).prune(tmp_path)
        assert result.deleted_count == 2
        assert len(result.kept) == 3


class TestOlderThanDays:
    def test_old_run_deleted(self, tmp_path):
        _write_state(tmp_path, "wf_old", started_at=_days_ago(30))
        _write_state(tmp_path, "wf_new", started_at=_days_ago(1))
        result = WorkflowPruner(older_than_days=7).prune(tmp_path)
        assert "wf_old" in result.deleted
        assert "wf_new" in result.kept

    def test_no_old_runs_nothing_deleted(self, tmp_path):
        _write_state(tmp_path, "wf_recent", started_at=_days_ago(2))
        result = WorkflowPruner(older_than_days=30).prune(tmp_path)
        assert result.deleted_count == 0

    def test_none_started_at_skipped_not_deleted(self, tmp_path):
        d = tmp_path / "wf_no_ts"
        d.mkdir()
        (d / "state.json").write_text(
            json.dumps({"workflow_id": "wf_no_ts", "status": "completed"}),
            encoding="utf-8",
        )
        result = WorkflowPruner(older_than_days=7).prune(tmp_path)
        assert "wf_no_ts" in result.kept
        assert "wf_no_ts" not in result.deleted

    def test_malformed_started_at_kept(self, tmp_path):
        d = tmp_path / "wf_bad_ts"
        d.mkdir()
        (d / "state.json").write_text(
            json.dumps(
                {"workflow_id": "wf_bad_ts", "status": "completed", "started_at": "not-a-date"}
            ),
            encoding="utf-8",
        )
        result = WorkflowPruner(older_than_days=7).prune(tmp_path)
        assert "wf_bad_ts" in result.kept


class TestStatusFilter:
    def test_status_filter_limits_deletion(self, tmp_path):
        _write_state(tmp_path, "wf_done", started_at=_days_ago(30), status="completed")
        _write_state(tmp_path, "wf_fail", started_at=_days_ago(30), status="failed")
        _write_state(tmp_path, "wf_run", started_at=_days_ago(30), status="running")
        result = WorkflowPruner(older_than_days=7, statuses=["completed", "failed"]).prune(tmp_path)
        assert "wf_done" in result.deleted
        assert "wf_fail" in result.deleted
        assert "wf_run" in result.kept

    def test_status_filter_case_insensitive(self, tmp_path):
        _write_state(tmp_path, "wf_a", started_at=_days_ago(30), status="COMPLETED")
        result = WorkflowPruner(older_than_days=7, statuses=["completed"]).prune(tmp_path)
        assert "wf_a" in result.deleted

    def test_no_status_filter_deletes_all_matching_age(self, tmp_path):
        _write_state(tmp_path, "wf_x", started_at=_days_ago(30), status="failed")
        result = WorkflowPruner(older_than_days=7).prune(tmp_path)
        assert "wf_x" in result.deleted


class TestDryRun:
    def test_dry_run_does_not_delete_files(self, tmp_path):
        _write_state(tmp_path, "wf_old", started_at=_days_ago(30))
        result = WorkflowPruner(keep_last=0, dry_run=True).prune(tmp_path)
        assert result.deleted_count == 1
        assert (tmp_path / "wf_old").exists()  # not actually deleted

    def test_real_run_deletes_files(self, tmp_path):
        _write_state(tmp_path, "wf_old", started_at=_days_ago(30))
        result = WorkflowPruner(keep_last=0, dry_run=False).prune(tmp_path)
        assert result.deleted_count == 1
        assert not (tmp_path / "wf_old").exists()

    def test_dry_run_flag_set_on_result(self, tmp_path):
        result = WorkflowPruner(keep_last=5, dry_run=True).prune(tmp_path)
        assert result.dry_run is True


class TestSkipped:
    def test_unreadable_state_json_skipped(self, tmp_path):
        d = tmp_path / "wf_corrupt"
        d.mkdir()
        (d / "state.json").write_text("{{broken json", encoding="utf-8")
        result = WorkflowPruner(keep_last=0).prune(tmp_path)
        assert "wf_corrupt" in result.skipped
        assert "wf_corrupt" not in result.deleted

    def test_missing_state_json_skipped(self, tmp_path):
        d = tmp_path / "wf_no_state"
        d.mkdir()
        result = WorkflowPruner(keep_last=0).prune(tmp_path)
        assert "wf_no_state" in result.skipped


# ── CLI tests ─────────────────────────────────────────────────────────────────


class TestPruneRunsCLI:
    def test_no_criteria_exits_1(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(app, ["prune-runs", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "keep-last" in result.output or "older-than" in result.output

    def test_dry_run_shows_candidates(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(tmp_path, "wf_1", started_at=_days_ago(10))
        _write_state(tmp_path, "wf_2", started_at=_days_ago(1))
        result = CliRunner().invoke(
            app,
            ["prune-runs", "--keep-last", "1", "--dry-run", "--runs-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "wf_1" in result.output
        # directory must still exist after dry run
        assert (tmp_path / "wf_1").exists()

    def test_dry_run_nothing_to_prune(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(tmp_path, "wf_1")
        result = CliRunner().invoke(
            app,
            ["prune-runs", "--keep-last", "5", "--dry-run", "--runs-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "nothing" in result.output.lower()

    def test_yes_flag_skips_confirmation(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(tmp_path, "wf_old", started_at=_days_ago(30))
        result = CliRunner().invoke(
            app,
            ["prune-runs", "--keep-last", "0", "--yes", "--runs-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert not (tmp_path / "wf_old").exists()
        assert "Deleted" in result.output

    def test_older_than_days_cli(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(tmp_path, "wf_ancient", started_at=_days_ago(60))
        _write_state(tmp_path, "wf_recent", started_at=_days_ago(1))
        result = CliRunner().invoke(
            app,
            ["prune-runs", "--older-than-days", "30", "--yes", "--runs-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert not (tmp_path / "wf_ancient").exists()
        assert (tmp_path / "wf_recent").exists()
