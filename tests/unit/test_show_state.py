"""Unit tests for WorkflowStateInspector and the show-state CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_mlops.tools.state_inspector import WorkflowStateInspector

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_state(
    runs_dir: Path,
    workflow_id: str,
    *,
    steps: list[str],
    completed: list[str],
    step_status: dict[str, str] | None = None,
    step_outputs: dict | None = None,
    status: str = "running",
    current_state: str = "TRAINING_COMPLETED",
    pending_approval_id: str | None = None,
    mlflow_run_id: str | None = None,
    started_at: str = "2026-08-24T10:00:00+00:00",
    updated_at: str = "2026-08-24T11:00:00+00:00",
) -> Path:
    wf_dir = runs_dir / workflow_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": workflow_id,
        "status": status,
        "current_state": current_state,
        "steps": steps,
        "completed_steps": list(completed),
        "step_status": step_status or {s: "COMPLETED" for s in completed},
        "step_outputs": step_outputs or {s: {"report_path": f"fake/{s}.json"} for s in completed},
        "pending_approval_id": pending_approval_id,
        "mlflow_run_id": mlflow_run_id,
        "started_at": started_at,
        "updated_at": updated_at,
        "artifacts": [],
    }
    state_path = wf_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (wf_dir / "audit_log.jsonl").write_text("", encoding="utf-8")
    return state_path


def _append_audit(runs_dir: Path, workflow_id: str, events: list[dict]) -> None:
    log_path = runs_dir / workflow_id / "audit_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


# ── WorkflowStateInspector unit tests ────────────────────────────────────────


class TestInspectorBasic:
    def test_returns_none_for_missing_workflow(self, tmp_path):
        snap = WorkflowStateInspector().inspect(tmp_path, "nonexistent")
        assert snap is None

    def test_reads_workflow_id_and_status(self, tmp_path):
        _write_state(
            tmp_path, "wf1", steps=["training"], completed=["training"], status="completed"
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf1")
        assert snap is not None
        assert snap.workflow_id == "wf1"
        assert snap.status == "completed"

    def test_reads_current_state(self, tmp_path):
        _write_state(
            tmp_path,
            "wf2",
            steps=["training"],
            completed=["training"],
            current_state="TRAINING_COMPLETED",
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf2")
        assert snap.current_state == "TRAINING_COMPLETED"

    def test_reads_pending_approval_id(self, tmp_path):
        _write_state(
            tmp_path,
            "wf3",
            steps=["training", "approval"],
            completed=["training"],
            status="pending_approval",
            pending_approval_id="appr_wf3",
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf3")
        assert snap.pending_approval_id == "appr_wf3"

    def test_reads_mlflow_run_id(self, tmp_path):
        _write_state(
            tmp_path,
            "wf4",
            steps=["training"],
            completed=["training"],
            mlflow_run_id="mlflow-abc123",
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf4")
        assert snap.mlflow_run_id == "mlflow-abc123"

    def test_reads_timestamps(self, tmp_path):
        _write_state(
            tmp_path,
            "wf5",
            steps=["training"],
            completed=["training"],
            started_at="2026-08-24T10:00:00+00:00",
            updated_at="2026-08-24T11:30:00+00:00",
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf5")
        assert "2026-08-24" in snap.started_at
        assert "2026-08-24" in snap.updated_at


class TestStepSummaries:
    def test_completed_step_has_completed_status(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s1",
            steps=["dataset_validation", "training"],
            completed=["dataset_validation", "training"],
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s1")
        statuses = {ss.name: ss.status for ss in snap.step_summaries}
        assert statuses["dataset_validation"] == "completed"
        assert statuses["training"] == "completed"

    def test_pending_step_is_pending(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s2",
            steps=["dataset_validation", "training"],
            completed=["dataset_validation"],
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s2")
        statuses = {ss.name: ss.status for ss in snap.step_summaries}
        assert statuses["training"] == "pending"

    def test_failed_step_status(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s3",
            steps=["dataset_validation", "training"],
            completed=["dataset_validation"],
            step_status={"dataset_validation": "COMPLETED", "training": "FAILED"},
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s3")
        statuses = {ss.name: ss.status for ss in snap.step_summaries}
        assert statuses["training"] == "failed"

    def test_skipped_step_status(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s4",
            steps=["training", "approval", "model_registry"],
            completed=["training", "approval", "model_registry"],
            step_status={
                "training": "COMPLETED",
                "approval": "COMPLETED",
                "model_registry": "SKIPPED",
            },
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s4")
        statuses = {ss.name: ss.status for ss in snap.step_summaries}
        assert statuses["model_registry"] == "skipped"

    def test_artifacts_extracted_from_step_outputs(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s5",
            steps=["training"],
            completed=["training"],
            step_outputs={
                "training": {
                    "best_weights_path": "runs/wf_s5/artifacts/best.pt",
                    "report_path": "runs/wf_s5/artifacts/training_report.json",
                }
            },
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s5")
        ss = next(s for s in snap.step_summaries if s.name == "training")
        assert any("best.pt" in a for a in ss.artifacts)

    def test_num_completed_counter(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s6",
            steps=["dataset_validation", "training", "evaluation"],
            completed=["dataset_validation", "training"],
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s6")
        assert snap.num_completed == 2
        assert snap.num_failed == 0

    def test_num_failed_counter(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_s7",
            steps=["training", "evaluation"],
            completed=["training"],
            step_status={"training": "COMPLETED", "evaluation": "FAILED"},
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_s7")
        assert snap.num_failed == 1


class TestAuditEvents:
    def test_reads_audit_events(self, tmp_path):
        _write_state(tmp_path, "wf_a1", steps=["training"], completed=["training"])
        _append_audit(
            tmp_path,
            "wf_a1",
            [
                {"event": "workflow_started", "workflow_id": "wf_a1"},
                {"event": "step_completed", "step": "training"},
            ],
        )
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_a1", max_audit_events=10)
        event_names = [e.get("event") for e in snap.audit_events]
        assert "workflow_started" in event_names
        assert "step_completed" in event_names

    def test_max_audit_events_limit(self, tmp_path):
        _write_state(tmp_path, "wf_a2", steps=["training"], completed=["training"])
        _append_audit(tmp_path, "wf_a2", [{"event": f"e{i}"} for i in range(50)])
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_a2", max_audit_events=5)
        assert len(snap.audit_events) <= 5

    def test_empty_audit_log_is_fine(self, tmp_path):
        _write_state(tmp_path, "wf_a3", steps=["training"], completed=[])
        snap = WorkflowStateInspector().inspect(tmp_path, "wf_a3")
        assert snap.audit_events == []


class TestListWorkflows:
    def test_empty_runs_dir_returns_empty(self, tmp_path):
        assert WorkflowStateInspector().list_workflows(tmp_path) == []

    def test_nonexistent_runs_dir_returns_empty(self, tmp_path):
        assert WorkflowStateInspector().list_workflows(tmp_path / "nope") == []

    def test_lists_workflow_ids(self, tmp_path):
        _write_state(tmp_path, "wf_alpha", steps=["training"], completed=[])
        _write_state(tmp_path, "wf_beta", steps=["training"], completed=[])
        ids = WorkflowStateInspector().list_workflows(tmp_path)
        assert "wf_alpha" in ids
        assert "wf_beta" in ids

    def test_ignores_dirs_without_state_json(self, tmp_path):
        (tmp_path / "random_dir").mkdir()
        _write_state(tmp_path, "wf_real", steps=["training"], completed=[])
        ids = WorkflowStateInspector().list_workflows(tmp_path)
        assert "random_dir" not in ids
        assert "wf_real" in ids


# ── CLI tests ─────────────────────────────────────────────────────────────────


class TestShowStateCLI:
    def test_list_mode_when_no_workflow_id(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(
            tmp_path, "wf_list", steps=["training"], completed=["training"], status="completed"
        )
        runner = CliRunner()
        result = runner.invoke(app, ["show-state", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "wf_list" in result.output

    def test_detail_mode_for_existing_workflow(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(
            tmp_path,
            "wf_det",
            steps=["dataset_validation", "training"],
            completed=["dataset_validation"],
            status="running",
            current_state="DATASET_VALIDATION_COMPLETED",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["show-state", "wf_det", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "wf_det" in result.output
        assert "dataset_validation" in result.output
        assert "training" in result.output

    def test_exits_1_for_missing_workflow(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["show-state", "no_such_wf", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 1

    def test_shows_pending_approval_hint(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(
            tmp_path,
            "wf_pend",
            steps=["training", "approval"],
            completed=["training"],
            status="pending_approval",
            pending_approval_id="appr_wf_pend",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["show-state", "wf_pend", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "appr_wf_pend" in result.output

    def test_audit_flag_shows_events(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        _write_state(tmp_path, "wf_aud", steps=["training"], completed=["training"])
        _append_audit(
            tmp_path,
            "wf_aud",
            [
                {"event": "workflow_started"},
                {"event": "step_completed", "step": "training"},
            ],
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["show-state", "wf_aud", "--runs-dir", str(tmp_path), "--audit"],
        )
        assert result.exit_code == 0, result.output
        assert "workflow_started" in result.output or "step_completed" in result.output

    def test_empty_runs_dir_lists_nothing(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["show-state", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No workflows" in result.output


class TestShowStateJsonFormat:
    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def _app(self):
        from agentic_mlops.cli.main import app
        return app

    def test_list_mode_json(self, tmp_path):
        import json

        _write_state(tmp_path, "wf_j1", steps=["training"], completed=["training"])
        result = self._runner().invoke(
            self._app(), ["show-state", "--runs-dir", str(tmp_path), "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "workflow_ids" in data
        assert "wf_j1" in data["workflow_ids"]

    def test_detail_mode_json(self, tmp_path):
        import json

        _write_state(
            tmp_path, "wf_j2",
            steps=["dataset_validation", "training"],
            completed=["dataset_validation"],
            status="running",
        )
        result = self._runner().invoke(
            self._app(),
            ["show-state", "wf_j2", "--runs-dir", str(tmp_path), "--format", "json"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["workflow_id"] == "wf_j2"
        assert data["status"] == "running"
        assert any(s["name"] == "dataset_validation" for s in data["step_summaries"])

    def test_missing_workflow_exits_1_in_json_mode(self, tmp_path):
        result = self._runner().invoke(
            self._app(),
            ["show-state", "no_exist", "--runs-dir", str(tmp_path), "--format", "json"],
        )
        assert result.exit_code == 1
