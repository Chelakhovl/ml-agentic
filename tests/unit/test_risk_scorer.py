"""Tests for ModelRiskScorer, FakeModelRiskScorer, and orchestrator integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.contracts.risk import ModelRiskReport, RiskFactor, RiskLevel
from agentic_mlops.tools.risk_scorer import FakeModelRiskScorer, ModelRiskScorer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_eval_report(
    path: Path,
    *,
    map50=0.85,
    map50_95=0.60,
    precision=0.82,
    recall=0.80,
    recommendation="promote_candidate",
    failed_checks=None,
    passed_checks=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "success": True,
        "recommendation": recommendation,
        "metrics": {
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
        },
        "failed_checks": failed_checks or [],
        "passed_checks": passed_checks or ["map50 0.8500 >= 0.7500"],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_baseline_comparison(path: Path, *, delta_map50=0.05, improved=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"delta_map50": delta_map50, "improved": improved}
    path.write_text(json.dumps(data), encoding="utf-8")


# ── Risk level thresholds ─────────────────────────────────────────────────────


class TestRiskLevel:
    def test_low_below_035(self, tmp_path):
        from agentic_mlops.tools.risk_scorer import _level

        assert _level(0.0) == RiskLevel.LOW
        assert _level(0.34) == RiskLevel.LOW

    def test_medium_between_035_and_065(self, tmp_path):
        from agentic_mlops.tools.risk_scorer import _level

        assert _level(0.35) == RiskLevel.MEDIUM
        assert _level(0.50) == RiskLevel.MEDIUM
        assert _level(0.65) == RiskLevel.MEDIUM

    def test_high_above_065(self, tmp_path):
        from agentic_mlops.tools.risk_scorer import _level

        assert _level(0.66) == RiskLevel.HIGH
        assert _level(1.0) == RiskLevel.HIGH


# ── ModelRiskScorer compute ───────────────────────────────────────────────────


class TestModelRiskScorerCompute:
    def test_low_risk_for_excellent_model(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(
            p,
            map50=0.95,
            map50_95=0.75,
            precision=0.93,
            recall=0.90,
            recommendation="promote_candidate",
            failed_checks=[],
            passed_checks=["map50", "precision", "recall"],
        )
        report = ModelRiskScorer().score(p)
        assert report.risk_level == RiskLevel.LOW
        assert report.risk_score < 0.35

    def test_high_risk_for_failing_model(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(
            p,
            map50=0.60,
            map50_95=0.35,
            precision=0.55,
            recall=0.50,
            recommendation="retrain",
            failed_checks=["map50 0.60 < 0.75", "precision 0.55 < 0.70"],
            passed_checks=[],
        )
        report = ModelRiskScorer().score(p)
        assert report.risk_level == RiskLevel.HIGH
        assert report.risk_score > 0.65

    def test_four_factors_always_present(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        report = ModelRiskScorer().score(p)
        names = {f.name for f in report.factors}
        assert names == {
            "failed_checks",
            "metric_proximity",
            "recommendation",
            "baseline_regression",
        }

    def test_failed_checks_factor_zero_when_all_pass(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p, failed_checks=[], passed_checks=["a", "b", "c"])
        report = ModelRiskScorer().score(p)
        fc_factor = next(f for f in report.factors if f.name == "failed_checks")
        assert fc_factor.contribution == 0.0

    def test_failed_checks_full_weight_when_all_fail(self, tmp_path):
        from agentic_mlops.tools.risk_scorer import _WEIGHT_FAILED_CHECKS

        p = tmp_path / "eval.json"
        # Write JSON directly so passed_checks is definitely empty (helper default is non-empty)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "success": True,
                    "recommendation": "promote_candidate",
                    "metrics": {"map50": 0.95, "map50_95": 0.75, "precision": 0.90, "recall": 0.88},
                    "failed_checks": ["a", "b", "c"],
                    "passed_checks": [],
                }
            ),
            encoding="utf-8",
        )
        report = ModelRiskScorer().score(p)
        fc_factor = next(f for f in report.factors if f.name == "failed_checks")
        assert fc_factor.contribution == pytest.approx(_WEIGHT_FAILED_CHECKS, abs=1e-3)

    def test_recommendation_factor_zero_for_promote(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p, recommendation="promote_candidate")
        report = ModelRiskScorer().score(p)
        rec_factor = next(f for f in report.factors if f.name == "recommendation")
        assert rec_factor.contribution == 0.0

    def test_recommendation_factor_nonzero_for_retrain(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p, recommendation="retrain")
        report = ModelRiskScorer().score(p)
        rec_factor = next(f for f in report.factors if f.name == "recommendation")
        assert rec_factor.contribution > 0

    def test_baseline_regression_adds_risk(self, tmp_path):
        eval_p = tmp_path / "eval.json"
        bc_p = tmp_path / "bc.json"
        _write_eval_report(eval_p, recommendation="promote_candidate", failed_checks=[])
        _write_baseline_comparison(bc_p, delta_map50=-0.05, improved=False)
        report_without = ModelRiskScorer().score(eval_p)
        report_with = ModelRiskScorer().score(eval_p, baseline_comparison_path=bc_p)
        assert report_with.risk_score > report_without.risk_score

    def test_baseline_improvement_does_not_add_risk(self, tmp_path):
        eval_p = tmp_path / "eval.json"
        bc_p = tmp_path / "bc.json"
        _write_eval_report(eval_p, recommendation="promote_candidate", failed_checks=[])
        _write_baseline_comparison(bc_p, delta_map50=0.05, improved=True)
        bl_factor_result = ModelRiskScorer().score(eval_p, baseline_comparison_path=bc_p)
        bl = next(f for f in bl_factor_result.factors if f.name == "baseline_regression")
        assert bl.contribution == 0.0

    def test_missing_baseline_comparison_file_is_non_fatal(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        # Should not raise; missing baseline → 0 contribution
        report = ModelRiskScorer().score(p, baseline_comparison_path=tmp_path / "no_such.json")
        assert report is not None

    def test_score_capped_at_1(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(
            p,
            map50=0.50,
            map50_95=0.30,
            precision=0.40,
            recall=0.35,
            recommendation="retrain",
            failed_checks=["a", "b", "c", "d"],
            passed_checks=[],
        )
        bc_p = tmp_path / "bc.json"
        _write_baseline_comparison(bc_p, delta_map50=-0.20, improved=False)
        report = ModelRiskScorer().score(p, baseline_comparison_path=bc_p)
        assert report.risk_score <= 1.0

    def test_proximity_contributes_for_metrics_near_threshold(self, tmp_path):
        p = tmp_path / "eval.json"
        # map50 = 0.76, threshold = 0.75 → gap 0.01 < 0.05 → high proximity risk
        _write_eval_report(
            p,
            map50=0.76,
            map50_95=0.76,
            precision=0.76,
            recall=0.76,
            recommendation="promote_candidate",
            failed_checks=[],
            passed_checks=["ok"],
        )
        report = ModelRiskScorer().score(p)
        prox = next(f for f in report.factors if f.name == "metric_proximity")
        assert prox.contribution > 0

    def test_raises_on_missing_eval_file(self, tmp_path):
        with pytest.raises((OSError, FileNotFoundError)):
            ModelRiskScorer().score(tmp_path / "no_such.json")

    def test_raises_on_corrupt_eval_json(self, tmp_path):
        p = tmp_path / "eval.json"
        p.write_text("NOT JSON", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ModelRiskScorer().score(p)

    def test_generated_at_set(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        report = ModelRiskScorer().score(p)
        assert report.generated_at

    def test_eval_report_path_stored(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        report = ModelRiskScorer().score(p)
        assert str(p) in report.eval_report_path


class TestModelRiskScorerOutput:
    def test_writes_json_when_output_dir_given(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        out_dir = tmp_path / "out"
        ModelRiskScorer().score(p, output_dir=out_dir)
        assert (out_dir / "risk_report.json").exists()
        data = json.loads((out_dir / "risk_report.json").read_text())
        assert "risk_score" in data
        assert "risk_level" in data
        assert "factors" in data

    def test_no_file_without_output_dir(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        ModelRiskScorer().score(p)
        assert not (tmp_path / "risk_report.json").exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_eval_report(p)
        out_dir = tmp_path / "deep" / "subdir"
        ModelRiskScorer().score(p, output_dir=out_dir)
        assert (out_dir / "risk_report.json").exists()


# ── FakeModelRiskScorer ───────────────────────────────────────────────────────


class TestFakeModelRiskScorer:
    def test_records_call(self, tmp_path):
        fake = FakeModelRiskScorer()
        fake.score("eval.json")
        assert fake.calls == ["eval.json"]

    def test_returns_configured_result(self):
        custom = ModelRiskReport(
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
            factors=[RiskFactor(name="test", description="test", contribution=0.8)],
        )
        fake = FakeModelRiskScorer(result=custom)
        report = fake.score("eval.json")
        assert report.risk_score == 0.8

    def test_writes_file_when_output_dir_given(self, tmp_path):
        fake = FakeModelRiskScorer()
        out_dir = tmp_path / "out"
        fake.score("eval.json", output_dir=out_dir)
        assert (out_dir / "risk_report.json").exists()


# ── web reader: get_risk_report ───────────────────────────────────────────────


class TestGetRiskReportReader:
    def _write_report(self, runs_dir: Path, wf_id: str, data: dict) -> None:
        out_dir = runs_dir / wf_id / "artifacts" / "model_registry"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "risk_report.json").write_text(json.dumps(data), encoding="utf-8")

    def test_returns_none_for_missing_workflow(self, tmp_path):
        from agentic_mlops.web.reader import get_risk_report

        assert get_risk_report(tmp_path / "runs", "no_such") is None

    def test_returns_none_for_unsafe_id(self, tmp_path):
        from agentic_mlops.web.reader import get_risk_report

        assert get_risk_report(tmp_path, "../../evil") is None

    def test_returns_none_when_file_missing(self, tmp_path):
        from agentic_mlops.web.reader import get_risk_report

        (tmp_path / "wf_01").mkdir()
        assert get_risk_report(tmp_path, "wf_01") is None

    def test_returns_data_when_file_exists(self, tmp_path):
        from agentic_mlops.web.reader import get_risk_report

        self._write_report(tmp_path, "wf_01", {"risk_score": 0.2, "risk_level": "low"})
        data = get_risk_report(tmp_path, "wf_01")
        assert data["risk_score"] == 0.2
        assert data["risk_level"] == "low"

    def test_returns_none_for_corrupt_json(self, tmp_path):
        from agentic_mlops.web.reader import get_risk_report

        out_dir = tmp_path / "wf_bad" / "artifacts" / "model_registry"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "risk_report.json").write_text("CORRUPT", encoding="utf-8")
        assert get_risk_report(tmp_path, "wf_bad") is None


# ── Orchestrator integration ──────────────────────────────────────────────────


class TestOrchestratorRiskIntegration:
    def _run_with_registry(self, tmp_path: Path, scorer=None):
        """Run the orchestrator up through model_registry with stub agents."""
        import json as _json

        from agentic_mlops.contracts.evaluation import (
            EvaluationOutput,
            EvaluationRecommendation,
        )
        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow, _StepOutcome

        runs = tmp_path / "runs"
        eval_dir = runs / "wf_r" / "artifacts" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_report = {
            "success": True,
            "recommendation": "promote_candidate",
            "metrics": {"map50": 0.90, "map50_95": 0.60, "precision": 0.88, "recall": 0.82},
            "failed_checks": [],
            "passed_checks": ["map50"],
        }
        (eval_dir / "evaluation_report.json").write_text(_json.dumps(eval_report), encoding="utf-8")

        eval_output = EvaluationOutput(
            success=True,
            message="ok",
            recommendation=EvaluationRecommendation.PROMOTE_CANDIDATE,
        )
        (eval_dir / "evaluation_output.json").write_text(
            _json.dumps(eval_output.model_dump(mode="json")), encoding="utf-8"
        )

        wf = OrchestratorWorkflow(risk_scorer=scorer)
        original_dispatch = wf._dispatch

        def patched_dispatch(step, inp, output_root, step_outputs, azure_config):
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
                    key_outputs={"best_weights_path": "w.pt", "job_status": "completed"},
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
            workflow_id="wf_r",
            steps=["dataset_validation", "training", "evaluation", "approval", "model_registry"],
            runs_dir=str(runs),
            dry_run=True,
            dataset_path=str(tmp_path),
            data_yaml_path="data.yaml",
        )
        return wf.run(inp)

    def test_scorer_called_after_registry(self, tmp_path):
        fake = FakeModelRiskScorer()
        result = self._run_with_registry(tmp_path, scorer=fake)
        assert result.success
        assert len(fake.calls) == 1

    def test_risk_artifact_added_to_output(self, tmp_path):
        fake = FakeModelRiskScorer()
        result = self._run_with_registry(tmp_path, scorer=fake)
        assert any("risk_report.json" in a for a in result.artifacts)

    def test_real_scorer_writes_file(self, tmp_path):
        result = self._run_with_registry(tmp_path, scorer=None)
        assert result.success
        risk_path = tmp_path / "runs" / "wf_r" / "artifacts" / "model_registry" / "risk_report.json"
        assert risk_path.exists()
        data = json.loads(risk_path.read_text())
        assert "risk_score" in data
        assert "risk_level" in data

    def test_scorer_not_called_without_registry_step(self, tmp_path):
        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow, _StepOutcome

        runs = tmp_path / "runs"
        fake = FakeModelRiskScorer()
        wf = OrchestratorWorkflow(risk_scorer=fake)

        def patched(step, inp, output_root, step_outputs, azure_config):
            if step == "dataset_validation":
                return _StepOutcome(
                    success=True,
                    status_label="COMPLETED",
                    key_outputs={"dataset_path": str(tmp_path), "data_yaml_path": "d.yaml"},
                )
            return _StepOutcome(success=True, status_label="COMPLETED", key_outputs={})

        wf._dispatch = patched
        result = wf.run(
            OrchestratorInput(
                workflow_id="wf_nreg",
                steps=["dataset_validation"],
                runs_dir=str(runs),
                dry_run=True,
            )
        )
        assert result.success
        assert len(fake.calls) == 0
