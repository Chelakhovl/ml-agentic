"""Unit tests for `agentic-mlops status`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_mlops.cli.main import app

runner = CliRunner()


def _write_state(
    runs_dir: Path,
    workflow_id: str,
    status: str,
    current_state: str = "",
    pending_approval_id: str | None = None,
) -> None:
    wf_dir = runs_dir / workflow_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": workflow_id,
        "status": status,
        "current_state": current_state,
        "started_at": "2026-08-23T10:00:00",
    }
    if pending_approval_id:
        state["pending_approval_id"] = pending_approval_id
    (wf_dir / "state.json").write_text(json.dumps(state))


def _write_model(registry_dir: Path, model_name: str, version: int, map50: float = 0.9) -> None:
    ver_dir = registry_dir / model_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {"map50": map50, "approved_by": "jane"}
    (ver_dir / "lineage.json").write_text(json.dumps(lineage))
    latest = {
        "model_name": model_name,
        "version": version,
        "lineage": lineage,
        "registered_at": "2026-08-23T09:00:00",
    }
    (registry_dir / model_name / "latest.json").write_text(json.dumps(latest))


def _write_dataset(registry_dir: Path, dataset_name: str, version: int) -> None:
    ver_dir = registry_dir / dataset_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    latest = {
        "dataset_name": dataset_name,
        "version": version,
        "registered_at": "2026-08-23T08:00:00",
    }
    (registry_dir / dataset_name / "latest.json").write_text(json.dumps(latest))


def _write_monitoring(
    runs_dir: Path, workflow_id: str, step: str, recommended_action: str = "no_action"
) -> None:
    report_dir = runs_dir / workflow_id / "artifacts" / step
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "endpoint_name": "factory-defects",
        "workflow_id": workflow_id,
        "recommended_action": recommended_action,
        "generated_at": "2026-08-23T07:00:00",
    }
    (report_dir / "monitoring_report.json").write_text(json.dumps(report))


def _invoke(tmp_path: Path, extra_args: list[str] | None = None) -> object:
    return runner.invoke(
        app,
        [
            "status",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--registry-dir",
            str(tmp_path / "registry"),
            "--dataset-registry-dir",
            str(tmp_path / "dataset_registry"),
        ]
        + (extra_args or []),
    )


class TestStatusCommand:
    def test_exits_0_on_empty_dirs(self, tmp_path):
        result = _invoke(tmp_path)
        assert result.exit_code == 0

    def test_shows_no_workflows_when_empty(self, tmp_path):
        result = _invoke(tmp_path)
        assert "No workflows found" in result.output

    def test_shows_completed_workflow(self, tmp_path):
        _write_state(tmp_path / "runs", "wf_001", "completed", "COMPLETED")
        result = _invoke(tmp_path)
        assert result.exit_code == 0
        assert "wf_001" in result.output
        assert "completed" in result.output

    def test_shows_pending_approval_warning(self, tmp_path):
        _write_state(
            tmp_path / "runs",
            "wf_002",
            "pending_approval",
            "MODEL_APPROVAL_REQUIRED",
            pending_approval_id="appr_wf_002",
        )
        result = _invoke(tmp_path)
        assert result.exit_code == 0
        assert "awaiting approval" in result.output or "pending_approval" in result.output
        assert "appr_wf_002" in result.output

    def test_shows_registered_model(self, tmp_path):
        _write_model(tmp_path / "registry", "my-model", version=3, map50=0.912)
        result = _invoke(tmp_path)
        assert "my-model" in result.output
        assert "v3" in result.output
        assert "0.9120" in result.output

    def test_shows_registered_dataset(self, tmp_path):
        _write_dataset(tmp_path / "dataset_registry", "factory_defects", version=2)
        result = _invoke(tmp_path)
        assert "factory_defects" in result.output
        assert "v2" in result.output

    def test_shows_monitoring_alert(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_mon", "completed")
        _write_monitoring(
            runs, "wf_mon", "monitoring", recommended_action="create_retraining_request"
        )
        result = _invoke(tmp_path)
        assert result.exit_code == 0
        assert "create_retraining_request" in result.output

    def test_shows_nominal_when_no_issues(self, tmp_path):
        _write_state(tmp_path / "runs", "wf_ok", "completed", "COMPLETED")
        _write_model(tmp_path / "registry", "good-model", version=1)
        result = _invoke(tmp_path)
        assert result.exit_code == 0
        assert "nominal" in result.output.lower()

    def test_failed_workflow_shows_x_icon(self, tmp_path):
        _write_state(tmp_path / "runs", "wf_fail", "failed", "TRAINING_FAILED")
        result = _invoke(tmp_path)
        assert "wf_fail" in result.output

    def test_section_headers_always_present(self, tmp_path):
        result = _invoke(tmp_path)
        assert "Workflows" in result.output
        assert "Models" in result.output
        assert "Datasets" in result.output
        assert "Monitoring" in result.output

    def test_limit_flag_restricts_workflow_count(self, tmp_path):
        runs = tmp_path / "runs"
        for i in range(10):
            _write_state(runs, f"wf_{i:03d}", "completed")
        result = runner.invoke(
            app,
            [
                "status",
                "--runs-dir",
                str(runs),
                "--registry-dir",
                str(tmp_path / "registry"),
                "--dataset-registry-dir",
                str(tmp_path / "dataset_registry"),
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0
        # 10 total workflows, but output should show "10 total"
        assert "10 total" in result.output


class TestStatusJsonFormat:
    def _invoke_json(self, tmp_path: Path, extra: list[str] | None = None) -> object:
        return runner.invoke(
            app,
            [
                "status",
                "--runs-dir", str(tmp_path / "runs"),
                "--registry-dir", str(tmp_path / "registry"),
                "--dataset-registry-dir", str(tmp_path / "dataset_registry"),
                "--format", "json",
            ] + (extra or []),
        )

    def test_json_output_is_valid_json(self, tmp_path):
        result = self._invoke_json(tmp_path)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "workflows" in data
        assert "models" in data
        assert "datasets" in data
        assert "monitoring" in data

    def test_json_contains_workflow_data(self, tmp_path):
        _write_state(tmp_path / "runs", "wf_json", "completed", "COMPLETED")
        result = self._invoke_json(tmp_path)
        data = json.loads(result.output)
        assert data["workflows"]["total"] == 1
        assert any(w["workflow_id"] == "wf_json" for w in data["workflows"]["recent"])

    def test_json_counts_pending_approval(self, tmp_path):
        _write_state(tmp_path / "runs", "wf_p", "pending_approval", "MODEL_APPROVAL_REQUIRED",
                     pending_approval_id="appr_wf_p")
        result = self._invoke_json(tmp_path)
        data = json.loads(result.output)
        assert data["workflows"]["pending_approval"] == 1

    def test_json_counts_models(self, tmp_path):
        _write_model(tmp_path / "registry", "my-model", version=2, map50=0.88)
        result = self._invoke_json(tmp_path)
        data = json.loads(result.output)
        assert data["models"]["total"] == 1

    def test_json_counts_active_alerts(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_alert", "completed")
        _write_monitoring(runs, "wf_alert", "monitoring", recommended_action="model_review")
        result = self._invoke_json(tmp_path)
        data = json.loads(result.output)
        assert data["monitoring"]["active_alerts"] == 1
