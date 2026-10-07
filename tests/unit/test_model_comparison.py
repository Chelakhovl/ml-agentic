"""Unit tests for ModelComparator and ModelComparisonAgent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.agents.model_comparison import ModelComparisonAgent
from agentic_mlops.contracts.model_comparison import (
    ComparisonWeights,
    ModelComparisonInput,
)
from agentic_mlops.tools.model_comparator import ModelComparator, _normalise

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_report(tmp_path: Path, name: str, **metrics) -> Path:
    defaults = dict(
        map50=0.80,
        map50_95=0.55,
        precision=0.75,
        recall=0.72,
        success=True,
        recommendation="promote_candidate",
    )
    defaults.update(metrics)
    report = {
        "metrics": {
            "map50": defaults.pop("map50"),
            "map50_95": defaults.pop("map50_95"),
            "precision": defaults.pop("precision"),
            "recall": defaults.pop("recall"),
        },
        "recommendation": defaults.pop("recommendation", "promote_candidate"),
        "passed_checks": defaults.pop("passed_checks", []),
        "failed_checks": defaults.pop("failed_checks", []),
        "success": defaults.pop("success", True),
        **defaults,
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def _inp(reports: list[Path], **kwargs) -> ModelComparisonInput:
    return ModelComparisonInput(
        report_paths=[str(r) for r in reports],
        **kwargs,
    )


# ── _normalise ────────────────────────────────────────────────────────────────


class TestNormalise:
    def test_all_same_returns_half(self):
        assert _normalise([0.8, 0.8, 0.8]) == [0.5, 0.5, 0.5]

    def test_different_values(self):
        norm = _normalise([0.0, 0.5, 1.0])
        assert norm[0] == pytest.approx(0.0)
        assert norm[1] == pytest.approx(0.5)
        assert norm[2] == pytest.approx(1.0)

    def test_none_values_become_zero(self):
        norm = _normalise([0.0, None, 1.0])
        assert norm[1] == 0.0

    def test_single_value_returns_half(self):
        assert _normalise([0.9]) == [0.5]


# ── ModelComparator ───────────────────────────────────────────────────────────


class TestModelComparator:
    def test_empty_reports_fails(self):
        result = ModelComparator().compare(ModelComparisonInput(report_paths=[]))
        assert not result.success

    def test_missing_report_path_fails(self, tmp_path):
        result = ModelComparator().compare(
            _inp([tmp_path / "nonexistent.json", tmp_path / "also_missing.json"])
        )
        assert not result.success
        assert result.errors

    def test_two_models_ranked(self, tmp_path):
        a = _write_report(tmp_path, "model_a", map50=0.90)
        b = _write_report(tmp_path, "model_b", map50=0.70)
        result = ModelComparator().compare(_inp([a, b]))
        assert result.success
        assert result.rankings[0].model_name == "model_a"
        assert result.rankings[1].model_name == "model_b"
        assert result.rankings[0].rank == 1
        assert result.rankings[1].rank == 2

    def test_winner_is_rank_1(self, tmp_path):
        a = _write_report(tmp_path, "model_a", map50=0.90)
        b = _write_report(tmp_path, "model_b", map50=0.70)
        result = ModelComparator().compare(_inp([a, b]))
        assert result.winner == "model_a"

    def test_winner_report_path_correct(self, tmp_path):
        a = _write_report(tmp_path, "model_a", map50=0.90)
        b = _write_report(tmp_path, "model_b", map50=0.70)
        result = ModelComparator().compare(_inp([a, b]))
        assert result.winner_report_path == str(a)

    def test_model_names_override_stems(self, tmp_path):
        a = _write_report(tmp_path, "model_a")
        b = _write_report(tmp_path, "model_b")
        result = ModelComparator().compare(
            _inp([a, b], model_names=["Experiment-A", "Experiment-B"])
        )
        names = {e.model_name for e in result.rankings}
        assert "Experiment-A" in names
        assert "Experiment-B" in names

    def test_model_names_ignored_when_wrong_count(self, tmp_path):
        a = _write_report(tmp_path, "model_a")
        b = _write_report(tmp_path, "model_b")
        result = ModelComparator().compare(_inp([a, b], model_names=["OnlyOne"]))
        # Falls back to stem-based names
        assert result.success
        names = {e.model_name for e in result.rankings}
        assert "model_a" in names or "model_b" in names

    def test_total_models_correct(self, tmp_path):
        reports = [_write_report(tmp_path, f"m{i}") for i in range(3)]
        result = ModelComparator().compare(_inp(reports))
        assert result.total_models == 3

    def test_composite_score_in_0_1(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90, precision=0.85, recall=0.88)
        b = _write_report(tmp_path, "b", map50=0.70, precision=0.65, recall=0.60)
        result = ModelComparator().compare(_inp([a, b]))
        for e in result.rankings:
            assert 0.0 <= e.composite_score <= 1.0

    def test_latency_lower_is_better(self, tmp_path):
        a = _write_report(tmp_path, "slow", map50=0.80)
        b = _write_report(tmp_path, "fast", map50=0.80)
        result = ModelComparator().compare(_inp([a, b], measured_latency_ms=[500.0, 20.0]))
        # fast model should be ranked higher (same mAP50, better latency)
        assert result.winner == "fast"

    def test_model_size_lower_is_better(self, tmp_path):
        a = _write_report(tmp_path, "large", map50=0.80)
        b = _write_report(tmp_path, "small", map50=0.80)
        result = ModelComparator().compare(_inp([a, b], measured_model_size_mb=[500.0, 20.0]))
        assert result.winner == "small"

    def test_missing_latency_excluded_from_score(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.80)
        # No latency provided — scores should still be computed
        result = ModelComparator().compare(_inp([a, b]))
        assert result.success
        assert result.rankings[0].model_name == "a"

    def test_policy_applied(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90, recommendation="retrain")
        b = _write_report(tmp_path, "b", map50=0.80, recommendation="promote_candidate")
        result = ModelComparator().compare(_inp([a, b], promotion_policy_path="dummy_path"))
        # policy: only b passes (promote_candidate)
        assert result.policy_passing == 1
        # b passes, a does not
        b_entry = next(e for e in result.rankings if e.model_name == "b")
        a_entry = next(e for e in result.rankings if e.model_name == "a")
        assert b_entry.passes_policy is True
        assert a_entry.passes_policy is False

    def test_winner_is_highest_policy_passing(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90, recommendation="retrain")
        b = _write_report(tmp_path, "b", map50=0.80, recommendation="promote_candidate")
        result = ModelComparator().compare(_inp([a, b], promotion_policy_path="dummy_path"))
        assert result.winner == "b"  # a has higher score but fails policy

    def test_winner_none_when_all_fail_policy(self, tmp_path):
        a = _write_report(tmp_path, "a", recommendation="retrain")
        b = _write_report(tmp_path, "b", recommendation="collect_more_data")
        result = ModelComparator().compare(_inp([a, b], promotion_policy_path="dummy_path"))
        assert result.winner is None

    def test_no_policy_winner_is_rank_1(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.70)
        b = _write_report(tmp_path, "b", map50=0.90)
        result = ModelComparator().compare(_inp([a, b]))
        assert result.winner == "b"
        assert result.policy_passing is None

    def test_custom_weights_change_ranking(self, tmp_path):
        # model_a: great recall, mediocre precision
        # model_b: great precision, mediocre recall
        a = _write_report(tmp_path, "a", precision=0.60, recall=0.95)
        b = _write_report(tmp_path, "b", precision=0.95, recall=0.60)

        # Heavy weight on precision → b wins
        weights_p = ComparisonWeights(
            map50=0.0, map50_95=0.0, precision=1.0, recall=0.0, latency_ms=0.0, model_size_mb=0.0
        )
        result_p = ModelComparator().compare(_inp([a, b], weights=weights_p))
        assert result_p.winner == "b"

        # Heavy weight on recall → a wins
        weights_r = ComparisonWeights(
            map50=0.0, map50_95=0.0, precision=0.0, recall=1.0, latency_ms=0.0, model_size_mb=0.0
        )
        result_r = ModelComparator().compare(_inp([a, b], weights=weights_r))
        assert result_r.winner == "a"

    def test_single_model_score_is_half(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.85)
        result = ModelComparator().compare(_inp([a]))
        # Only one model — normalised scores all → 0.5 → composite = 0.5
        assert result.rankings[0].composite_score == pytest.approx(0.5)

    def test_notes_include_failed_checks(self, tmp_path):
        a = _write_report(tmp_path, "a", failed_checks=["precision_too_low"])
        b = _write_report(tmp_path, "b")
        result = ModelComparator().compare(_inp([a, b]))
        a_entry = next(e for e in result.rankings if e.model_name == "a")
        assert any("precision_too_low" in n for n in a_entry.notes)

    def test_report_metrics_captured(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.87, precision=0.83, recall=0.79)
        result = ModelComparator().compare(_inp([a]))
        entry = result.rankings[0]
        assert entry.map50 == pytest.approx(0.87)
        assert entry.precision == pytest.approx(0.83)
        assert entry.recall == pytest.approx(0.79)


# ── ModelComparisonAgent ──────────────────────────────────────────────────────


class TestModelComparisonAgent:
    def test_writes_json_report(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        agent = ModelComparisonAgent(artifacts_dir=tmp_path / "out")
        result = agent.run(_inp([a, b]))
        assert result.success
        assert result.comparison_report_path is not None
        assert Path(result.comparison_report_path).exists()

    def test_json_report_contains_rankings(self, tmp_path):
        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        out = tmp_path / "out"
        agent = ModelComparisonAgent(artifacts_dir=out)
        result = agent.run(_inp([a, b]))
        data = json.loads(Path(result.comparison_report_path).read_text())
        assert len(data["rankings"]) == 2
        assert data["winner"] == "a"

    def test_writes_md_report(self, tmp_path):
        a = _write_report(tmp_path, "a")
        b = _write_report(tmp_path, "b")
        agent = ModelComparisonAgent(artifacts_dir=tmp_path / "out")
        result = agent.run(_inp([a, b]))
        assert result.md_report_path is not None
        assert Path(result.md_report_path).exists()

    def test_md_contains_model_names(self, tmp_path):
        a = _write_report(tmp_path, "alpha")
        b = _write_report(tmp_path, "beta")
        out = tmp_path / "out"
        agent = ModelComparisonAgent(artifacts_dir=out)
        result = agent.run(_inp([a, b]))
        md = Path(result.md_report_path).read_text()
        assert "alpha" in md
        assert "beta" in md

    def test_artifacts_listed_in_output(self, tmp_path):
        a = _write_report(tmp_path, "a")
        b = _write_report(tmp_path, "b")
        agent = ModelComparisonAgent(artifacts_dir=tmp_path / "out")
        result = agent.run(_inp([a, b]))
        assert any("comparison_report.json" in p for p in result.artifacts)
        assert any("comparison_report.md" in p for p in result.artifacts)

    def test_failed_comparison_no_report_written(self, tmp_path):
        agent = ModelComparisonAgent(artifacts_dir=tmp_path / "out")
        result = agent.run(ModelComparisonInput(report_paths=[]))
        assert not result.success
        assert result.comparison_report_path is None

    def test_mlflow_logging(self, tmp_path):
        from agentic_mlops.integrations.mlflow_client import FakeMLflowClient

        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        fake_mlflow = FakeMLflowClient()
        run_id = fake_mlflow.start_run("comparison_test", "run_comparison")
        agent = ModelComparisonAgent(
            artifacts_dir=tmp_path / "out",
            mlflow_client=fake_mlflow,
            mlflow_run_id=run_id,
        )
        agent.run(_inp([a, b]))
        run = fake_mlflow.runs[run_id]
        assert "comparison.winner_map50" in run["metrics"]
        assert run["params"].get("comparison.winner") == "a"


# ── CLI integration ───────────────────────────────────────────────────────────


class TestCompareCli:
    def test_cli_basic(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compare-models",
                str(a),
                str(b),
                "--output-dir",
                str(tmp_path / "comparison_out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "model_a" in result.output or "a" in result.output

    def test_cli_rejects_single_report(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        a = _write_report(tmp_path, "a")
        runner = CliRunner()
        result = runner.invoke(app, ["compare-models", str(a), "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_cli_model_names(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compare-models",
                str(a),
                str(b),
                "--model-names",
                "ModelAlpha,ModelBeta",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0
        assert "ModelAlpha" in result.output or "ModelBeta" in result.output

    def test_cli_weights(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compare-models",
                str(a),
                str(b),
                "--weights",
                "map50=1.0,precision=0.0,recall=0.0,map50_95=0.0,latency_ms=0.0,model_size_mb=0.0",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0

    def test_cli_json_format(self, tmp_path):
        import json

        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        a = _write_report(tmp_path, "a", map50=0.90)
        b = _write_report(tmp_path, "b", map50=0.70)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compare-models",
                str(a),
                str(b),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
        # Logging lines (INFO:/DEBUG:) may prefix the JSON — find the first {
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert "rankings" in data
        assert "winner" in data
        assert data["total_models"] == 2
        names = [r["model_name"] for r in data["rankings"]]
        assert any("a" in n for n in names)
