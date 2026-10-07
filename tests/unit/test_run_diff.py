"""Unit tests for WorkflowRunDiffer and the diff-runs CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.tools.run_diff import WorkflowRunDiffer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_state(
    runs_dir: Path,
    workflow_id: str,
    *,
    steps: list[str],
    completed: list[str],
    step_status: dict[str, str] | None = None,
    step_outputs: dict | None = None,
    status: str = "completed",
    started_at: str = "2026-08-25T10:00:00+00:00",
) -> None:
    wf_dir = runs_dir / workflow_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": workflow_id,
        "status": status,
        "steps": steps,
        "completed_steps": list(completed),
        "step_status": step_status or {s: "COMPLETED" for s in completed},
        "step_outputs": step_outputs or {},
        "started_at": started_at,
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _differ() -> WorkflowRunDiffer:
    return WorkflowRunDiffer()


# ── WorkflowRunDiffer ─────────────────────────────────────────────────────────


class TestDiffBasic:
    def test_returns_none_when_first_missing(self, tmp_path):
        _write_state(tmp_path, "wf_b", steps=["training"], completed=["training"])
        assert _differ().diff(tmp_path, "no_such_wf", "wf_b") is None

    def test_returns_none_when_second_missing(self, tmp_path):
        _write_state(tmp_path, "wf_a", steps=["training"], completed=["training"])
        assert _differ().diff(tmp_path, "wf_a", "no_such_wf") is None

    def test_returns_none_when_both_missing(self, tmp_path):
        assert _differ().diff(tmp_path, "x", "y") is None

    def test_workflow_ids_in_result(self, tmp_path):
        for wf in ("wf_a", "wf_b"):
            _write_state(tmp_path, wf, steps=["training"], completed=["training"])
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert d.wf_a == "wf_a"
        assert d.wf_b == "wf_b"

    def test_statuses_read(self, tmp_path):
        _write_state(
            tmp_path, "wf_a", steps=["training"], completed=["training"], status="completed"
        )
        _write_state(tmp_path, "wf_b", steps=["training"], completed=[], status="failed")
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert d.status_a == "completed"
        assert d.status_b == "failed"

    def test_started_at_read(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["training"],
            completed=["training"],
            started_at="2026-08-25T08:00:00+00:00",
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["training"],
            completed=["training"],
            started_at="2026-08-26T09:00:00+00:00",
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert "2026-08-25" in d.started_at_a
        assert "2026-08-26" in d.started_at_b


class TestStepDiffs:
    def test_identical_steps_no_change(self, tmp_path):
        for wf in ("wf_a", "wf_b"):
            _write_state(
                tmp_path, wf, steps=["training", "evaluation"], completed=["training", "evaluation"]
            )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        for sd in d.step_diffs:
            assert not sd.changed

    def test_changed_step_status_detected(self, tmp_path):
        _write_state(
            tmp_path, "wf_a", steps=["training", "evaluation"], completed=["training", "evaluation"]
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["training", "evaluation"],
            completed=["training"],
            step_status={"training": "COMPLETED", "evaluation": "FAILED"},
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        eval_diff = next(sd for sd in d.step_diffs if sd.name == "evaluation")
        assert eval_diff.status_a == "completed"
        assert eval_diff.status_b == "failed"
        assert eval_diff.changed

    def test_steps_only_in_a(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["training", "evaluation", "model_registry"],
            completed=["training", "evaluation", "model_registry"],
        )
        _write_state(
            tmp_path, "wf_b", steps=["training", "evaluation"], completed=["training", "evaluation"]
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert "model_registry" in d.steps_only_in_a
        assert not d.steps_only_in_b

    def test_steps_only_in_b(self, tmp_path):
        _write_state(tmp_path, "wf_a", steps=["training"], completed=["training"])
        _write_state(
            tmp_path, "wf_b", steps=["training", "evaluation"], completed=["training", "evaluation"]
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert "evaluation" in d.steps_only_in_b
        assert not d.steps_only_in_a

    def test_skipped_step_resolved(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["training", "model_registry"],
            completed=["training", "model_registry"],
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["training", "model_registry"],
            completed=["training", "model_registry"],
            step_status={"training": "COMPLETED", "model_registry": "SKIPPED"},
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        reg_diff = next(sd for sd in d.step_diffs if sd.name == "model_registry")
        assert reg_diff.status_b == "skipped"
        assert reg_diff.changed


class TestMetricDiffs:
    def _eval_outputs(
        self, map50: float, precision: float = 0.8, recall: float = 0.75, map50_95: float = 0.5
    ) -> dict:
        return {
            "evaluation": {
                "map50": map50,
                "precision": precision,
                "recall": recall,
                "map50_95": map50_95,
            }
        }

    def test_metric_delta_computed(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs=self._eval_outputs(0.712),
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs=self._eval_outputs(0.834),
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        map50_diff = next(m for m in d.metric_diffs if m.metric == "evaluation.map50")
        assert abs(map50_diff.value_a - 0.712) < 1e-6
        assert abs(map50_diff.value_b - 0.834) < 1e-6
        assert abs(map50_diff.delta - 0.122) < 1e-5
        assert map50_diff.improved is True

    def test_metric_regression_detected(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs=self._eval_outputs(0.900),
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs=self._eval_outputs(0.750),
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        map50_diff = next(m for m in d.metric_diffs if m.metric == "evaluation.map50")
        assert map50_diff.improved is False
        assert map50_diff.delta < 0

    def test_missing_metric_delta_is_none(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs={"evaluation": {"map50": 0.7}},
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["evaluation"],
            completed=["evaluation"],
            step_outputs={"evaluation": {}},
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        map50_diff = next((m for m in d.metric_diffs if m.metric == "evaluation.map50"), None)
        assert map50_diff is not None
        assert map50_diff.delta is None
        assert map50_diff.improved is None

    def test_no_evaluation_no_metric_diffs(self, tmp_path):
        for wf in ("wf_a", "wf_b"):
            _write_state(
                tmp_path,
                wf,
                steps=["training"],
                completed=["training"],
                step_outputs={"training": {"best_weights_path": "runs/wf/best.pt"}},
            )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert all(
            m.metric.startswith("training.") or m.metric.startswith("evaluation.")
            for m in d.metric_diffs
        )
        # training step has no listed metrics, so no diffs
        eval_metrics = [m for m in d.metric_diffs if m.metric.startswith("evaluation.")]
        assert not eval_metrics

    def test_model_registry_version_diff(self, tmp_path):
        _write_state(
            tmp_path,
            "wf_a",
            steps=["model_registry"],
            completed=["model_registry"],
            step_outputs={"model_registry": {"version": 1}},
        )
        _write_state(
            tmp_path,
            "wf_b",
            steps=["model_registry"],
            completed=["model_registry"],
            step_outputs={"model_registry": {"version": 2}},
        )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        ver_diff = next((m for m in d.metric_diffs if m.metric == "model_registry.version"), None)
        assert ver_diff is not None
        assert ver_diff.value_a == 1.0
        assert ver_diff.value_b == 2.0

    def test_has_changes_false_when_identical(self, tmp_path):
        for wf in ("wf_a", "wf_b"):
            _write_state(
                tmp_path, wf, steps=["training"], completed=["training"], status="completed"
            )
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert not d.has_changes

    def test_has_changes_true_when_status_differs(self, tmp_path):
        _write_state(
            tmp_path, "wf_a", steps=["training"], completed=["training"], status="completed"
        )
        _write_state(tmp_path, "wf_b", steps=["training"], completed=[], status="failed")
        d = _differ().diff(tmp_path, "wf_a", "wf_b")
        assert d.has_changes


# ── CLI tests ─────────────────────────────────────────────────────────────────


class TestDiffRunsCLI:
    def _make_runs(self, tmp_path: Path) -> tuple[Path, str, str]:
        runs = tmp_path / "runs"
        runs.mkdir()
        _write_state(
            runs,
            "wf_1",
            steps=["dataset_validation", "training", "evaluation"],
            completed=["dataset_validation", "training", "evaluation"],
            step_outputs={
                "evaluation": {
                    "map50": 0.712,
                    "precision": 0.801,
                    "recall": 0.695,
                    "map50_95": 0.512,
                }
            },
        )
        _write_state(
            runs,
            "wf_2",
            steps=["dataset_validation", "training", "evaluation"],
            completed=["dataset_validation", "training", "evaluation"],
            step_outputs={
                "evaluation": {
                    "map50": 0.834,
                    "precision": 0.823,
                    "recall": 0.711,
                    "map50_95": 0.601,
                }
            },
        )
        return runs, "wf_1", "wf_2"

    def test_exit_0_for_valid_runs(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs, a, b = self._make_runs(tmp_path)
        result = CliRunner().invoke(app, ["diff-runs", a, b, "--runs-dir", str(runs)])
        assert result.exit_code == 0, result.output

    def test_output_contains_workflow_ids(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs, a, b = self._make_runs(tmp_path)
        result = CliRunner().invoke(app, ["diff-runs", a, b, "--runs-dir", str(runs)])
        assert "wf_1" in result.output
        assert "wf_2" in result.output

    def test_output_contains_metrics(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs, a, b = self._make_runs(tmp_path)
        result = CliRunner().invoke(app, ["diff-runs", a, b, "--runs-dir", str(runs)])
        assert "map50" in result.output
        assert "0.712" in result.output or "0.834" in result.output

    def test_exit_1_for_missing_run(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs, a, _ = self._make_runs(tmp_path)
        result = CliRunner().invoke(app, ["diff-runs", a, "no_such_wf", "--runs-dir", str(runs)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "no_such_wf" in result.output

    def test_shows_changed_step_marker(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs = tmp_path / "runs2"
        runs.mkdir()
        _write_state(
            runs, "wf_a", steps=["training", "evaluation"], completed=["training", "evaluation"]
        )
        _write_state(
            runs,
            "wf_b",
            steps=["training", "evaluation"],
            completed=["training"],
            step_status={"training": "COMPLETED", "evaluation": "FAILED"},
        )
        result = CliRunner().invoke(app, ["diff-runs", "wf_a", "wf_b", "--runs-dir", str(runs)])
        assert result.exit_code == 0
        assert "evaluation" in result.output
        assert "failed" in result.output


class TestDiffRunsJsonFormat:
    def _make_runs(self, tmp_path: Path) -> Path:
        runs = tmp_path / "runs"
        runs.mkdir()
        _write_state(
            runs, "wf_x",
            steps=["training", "evaluation"],
            completed=["training", "evaluation"],
            step_outputs={
                "evaluation": {"map50": 0.80, "precision": 0.85, "recall": 0.75, "map50_95": 0.55}
            },
        )
        _write_state(
            runs, "wf_y",
            steps=["training", "evaluation"],
            completed=["training", "evaluation"],
            step_outputs={
                "evaluation": {"map50": 0.90, "precision": 0.88, "recall": 0.80, "map50_95": 0.60}
            },
        )
        return runs

    def test_json_output_valid(self, tmp_path):
        import json

        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs = self._make_runs(tmp_path)
        result = CliRunner().invoke(
            app, ["diff-runs", "wf_x", "wf_y", "--runs-dir", str(runs), "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["wf_a"] == "wf_x"
        assert data["wf_b"] == "wf_y"
        assert "step_diffs" in data
        assert "metric_diffs" in data
        assert "has_changes" in data

    def test_json_metric_diffs_present(self, tmp_path):
        import json

        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runs = self._make_runs(tmp_path)
        result = CliRunner().invoke(
            app, ["diff-runs", "wf_x", "wf_y", "--runs-dir", str(runs), "--format", "json"]
        )
        data = json.loads(result.output)
        metrics = {m["metric"]: m for m in data["metric_diffs"]}
        assert "evaluation.map50" in metrics
        assert metrics["evaluation.map50"]["value_a"] == pytest.approx(0.80, abs=1e-4)
        assert metrics["evaluation.map50"]["value_b"] == pytest.approx(0.90, abs=1e-4)
