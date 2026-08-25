"""Tests for BaselineComparator, FakeBaselineComparator, and orchestrator integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.contracts.baseline import BaselineComparisonResult
from agentic_mlops.tools.baseline_comparator import BaselineComparator, FakeBaselineComparator

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_eval_report(path: Path, *, map50=0.80, map50_95=0.55, precision=0.78, recall=0.75):
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "success": True,
        "metrics": {
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


# ── Unit tests for BaselineComparator ─────────────────────────────────────────


class TestBaselineComparatorCompute:
    def test_positive_delta_when_new_is_better(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.90)
        _write_eval_report(base_path, map50=0.80)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.delta_map50 > 0
        assert round(result.delta_map50, 4) == round(0.10, 4)

    def test_negative_delta_when_new_is_worse(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.70)
        _write_eval_report(base_path, map50=0.80)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.delta_map50 < 0

    def test_zero_delta_for_identical_metrics(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.85, precision=0.80, recall=0.75)
        _write_eval_report(base_path, map50=0.85, precision=0.80, recall=0.75)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.delta_map50 == 0.0
        assert result.delta_precision == 0.0
        assert result.delta_recall == 0.0

    def test_improved_true_when_delta_exceeds_threshold(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.90)
        _write_eval_report(base_path, map50=0.80)
        result = BaselineComparator(min_improvement_map50=0.05).compare(eval_path, base_path)
        assert result.improved is True

    def test_improved_false_when_delta_below_threshold(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.82)
        _write_eval_report(base_path, map50=0.80)
        result = BaselineComparator(min_improvement_map50=0.05).compare(eval_path, base_path)
        assert result.improved is False

    def test_improved_true_at_exactly_threshold(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.85)
        _write_eval_report(base_path, map50=0.80)
        result = BaselineComparator(min_improvement_map50=0.05).compare(eval_path, base_path)
        assert result.improved is True

    def test_all_metric_fields_populated(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.90, map50_95=0.60, precision=0.88, recall=0.82)
        _write_eval_report(base_path, map50=0.80, map50_95=0.55, precision=0.78, recall=0.75)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.new_map50 == pytest.approx(0.90)
        assert result.baseline_map50 == pytest.approx(0.80)
        assert result.delta_precision == pytest.approx(0.10)
        assert result.delta_recall == pytest.approx(0.07)

    def test_default_threshold_zero_any_improvement_counts(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.801)
        _write_eval_report(base_path, map50=0.800)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.improved is True

    def test_paths_stored_in_result(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path)
        _write_eval_report(base_path)
        result = BaselineComparator().compare(eval_path, base_path)
        assert str(eval_path) in result.eval_report_path
        assert str(base_path) in result.baseline_path

    def test_generated_at_is_set(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path)
        _write_eval_report(base_path)
        result = BaselineComparator().compare(eval_path, base_path)
        assert result.generated_at  # non-empty ISO timestamp

    def test_raises_on_missing_eval_file(self, tmp_path):
        base_path = tmp_path / "baseline.json"
        _write_eval_report(base_path)
        with pytest.raises((OSError, FileNotFoundError)):
            BaselineComparator().compare(tmp_path / "no_such.json", base_path)

    def test_raises_on_missing_baseline_file(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        _write_eval_report(eval_path)
        with pytest.raises((OSError, FileNotFoundError)):
            BaselineComparator().compare(eval_path, tmp_path / "no_such.json")

    def test_raises_on_corrupt_json(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        eval_path.write_text("NOT JSON", encoding="utf-8")
        _write_eval_report(base_path)
        with pytest.raises(json.JSONDecodeError):
            BaselineComparator().compare(eval_path, base_path)


class TestBaselineComparatorOutput:
    def test_writes_json_when_output_dir_given(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path, map50=0.90)
        _write_eval_report(base_path, map50=0.80)
        out_dir = tmp_path / "out"
        BaselineComparator().compare(eval_path, base_path, output_dir=out_dir)
        out_file = out_dir / "baseline_comparison.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["delta_map50"] == pytest.approx(0.10, abs=1e-4)

    def test_no_file_written_without_output_dir(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path)
        _write_eval_report(base_path)
        BaselineComparator().compare(eval_path, base_path)
        assert not (tmp_path / "baseline_comparison.json").exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        base_path = tmp_path / "baseline.json"
        _write_eval_report(eval_path)
        _write_eval_report(base_path)
        out_dir = tmp_path / "deep" / "subdir"
        BaselineComparator().compare(eval_path, base_path, output_dir=out_dir)
        assert (out_dir / "baseline_comparison.json").exists()


# ── FakeBaselineComparator ────────────────────────────────────────────────────


class TestFakeBaselineComparator:
    def test_records_call(self, tmp_path):
        fake = FakeBaselineComparator()
        fake.compare("eval.json", "base.json")
        assert len(fake.calls) == 1
        assert fake.calls[0] == ("eval.json", "base.json")

    def test_returns_configured_result(self, tmp_path):
        custom = BaselineComparisonResult(new_map50=0.95, baseline_map50=0.80, delta_map50=0.15)
        fake = FakeBaselineComparator(result=custom)
        result = fake.compare("eval.json", "base.json")
        assert result.delta_map50 == 0.15

    def test_writes_file_when_output_dir_given(self, tmp_path):
        fake = FakeBaselineComparator()
        out_dir = tmp_path / "out"
        fake.compare("eval.json", "base.json", output_dir=out_dir)
        assert (out_dir / "baseline_comparison.json").exists()


# ── web reader: get_baseline_comparison ──────────────────────────────────────


class TestGetBaselineComparisonReader:
    def _write_comparison(self, runs_dir: Path, wf_id: str, data: dict) -> None:
        out_dir = runs_dir / wf_id / "artifacts" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "baseline_comparison.json").write_text(json.dumps(data), encoding="utf-8")

    def test_returns_none_for_missing_workflow(self, tmp_path):
        from agentic_mlops.web.reader import get_baseline_comparison

        assert get_baseline_comparison(tmp_path / "runs", "no_such") is None

    def test_returns_none_for_unsafe_id(self, tmp_path):
        from agentic_mlops.web.reader import get_baseline_comparison

        assert get_baseline_comparison(tmp_path, "../../evil") is None

    def test_returns_none_when_file_missing(self, tmp_path):
        from agentic_mlops.web.reader import get_baseline_comparison

        (tmp_path / "wf_01").mkdir()
        assert get_baseline_comparison(tmp_path, "wf_01") is None

    def test_returns_data_when_file_exists(self, tmp_path):
        from agentic_mlops.web.reader import get_baseline_comparison

        self._write_comparison(tmp_path, "wf_01", {"delta_map50": 0.05, "improved": True})
        data = get_baseline_comparison(tmp_path, "wf_01")
        assert data is not None
        assert data["delta_map50"] == 0.05
        assert data["improved"] is True

    def test_returns_none_for_corrupt_json(self, tmp_path):
        from agentic_mlops.web.reader import get_baseline_comparison

        out_dir = tmp_path / "wf_bad" / "artifacts" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "baseline_comparison.json").write_text("CORRUPT", encoding="utf-8")
        assert get_baseline_comparison(tmp_path, "wf_bad") is None


# ── Orchestrator integration ──────────────────────────────────────────────────


class TestOrchestratorBaselineIntegration:
    """Verify that OrchestratorWorkflow wires FakeBaselineComparator correctly."""

    def _make_workflow(self, tmp_path: Path, comparator=None):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        return OrchestratorWorkflow(baseline_comparator=comparator)

    def _run_with_fake_eval(
        self,
        tmp_path: Path,
        baseline_report_path: str | None,
        comparator=None,
    ):
        """Run the orchestrator with fake agents, stopping after evaluation."""
        from agentic_mlops.contracts.evaluation import (
            EvaluationOutput,
            EvaluationRecommendation,
        )
        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        # Build a fake evaluation_report.json that the comparator will read
        eval_dir = tmp_path / "runs" / "wf_t" / "artifacts" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_report = {
            "success": True,
            "metrics": {"map50": 0.90, "map50_95": 0.60, "precision": 0.88, "recall": 0.82},
        }
        (eval_dir / "evaluation_report.json").write_text(json.dumps(eval_report), encoding="utf-8")

        # Also write a fake evaluation_output.json (EvaluationOutput shape)
        eval_output_data = EvaluationOutput(
            success=True,
            message="ok",
            recommendation=EvaluationRecommendation.PROMOTE_CANDIDATE,
        ).model_dump(mode="json")
        (eval_dir / "evaluation_output.json").write_text(
            json.dumps(eval_output_data), encoding="utf-8"
        )

        wf = OrchestratorWorkflow(baseline_comparator=comparator)

        # Inject stub factories so no real ML runs happen.
        # We patch the _dispatch method to short-circuit the evaluation step.
        original_dispatch = wf._dispatch

        def patched_dispatch(step, inp, output_root, step_outputs, azure_config):
            from agentic_mlops.workflows.orchestrator import _StepOutcome

            if step == "dataset_validation":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={"dataset_path": str(tmp_path), "data_yaml_path": "data.yaml"},
                )
            if step == "training":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={"best_weights_path": "weights.pt", "job_status": "completed"},
                )
            if step == "evaluation":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={
                        "report_path": str(eval_dir / "evaluation_report.json"),
                        "output_json_path": str(eval_dir / "evaluation_output.json"),
                    },
                )
            if step == "approval":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={"action": "approve_model", "status": "approved"},
                )
            if step == "model_registry":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={"version": 1, "model_name": "m"},
                )
            return original_dispatch(step, inp, output_root, step_outputs, azure_config)

        wf._dispatch = patched_dispatch

        inp = OrchestratorInput(
            workflow_id="wf_t",
            steps=["dataset_validation", "training", "evaluation"],
            runs_dir=str(tmp_path / "runs"),
            dry_run=True,
            baseline_report_path=baseline_report_path,
            dataset_path=str(tmp_path),
            data_yaml_path="data.yaml",
        )
        return wf.run(inp)

    def test_comparator_called_when_baseline_configured(self, tmp_path):
        # Write a baseline report
        base_path = tmp_path / "baseline.json"
        _write_eval_report(base_path, map50=0.80)

        fake_cmp = FakeBaselineComparator()
        result = self._run_with_fake_eval(tmp_path, str(base_path), fake_cmp)
        assert result.success
        assert len(fake_cmp.calls) == 1

    def test_comparator_not_called_without_baseline_path(self, tmp_path):
        fake_cmp = FakeBaselineComparator()
        result = self._run_with_fake_eval(tmp_path, baseline_report_path=None, comparator=fake_cmp)
        assert result.success
        assert len(fake_cmp.calls) == 0

    def test_comparison_artifact_added_to_output(self, tmp_path):
        base_path = tmp_path / "baseline.json"
        _write_eval_report(base_path, map50=0.80)

        fake_cmp = FakeBaselineComparator()
        result = self._run_with_fake_eval(tmp_path, str(base_path), fake_cmp)
        assert any("baseline_comparison.json" in a for a in result.artifacts)

    def test_real_comparator_writes_file(self, tmp_path):
        base_path = tmp_path / "baseline.json"
        _write_eval_report(base_path, map50=0.80)

        result = self._run_with_fake_eval(tmp_path, str(base_path), comparator=None)
        assert result.success
        comp_path = (
            tmp_path / "runs" / "wf_t" / "artifacts" / "evaluation" / "baseline_comparison.json"
        )
        assert comp_path.exists()
        data = json.loads(comp_path.read_text())
        assert "delta_map50" in data

    def test_failure_in_comparator_is_non_fatal(self, tmp_path):
        """If the baseline path doesn't exist the workflow still succeeds."""
        result = self._run_with_fake_eval(
            tmp_path, baseline_report_path="/no/such/baseline.json", comparator=None
        )
        assert result.success
