"""Unit tests for EvaluationAgent (MVP Agent 3)."""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.evaluation import EvaluationAgent
from agentic_mlops.contracts.evaluation import (
    EvaluationInput,
    EvaluationMetrics,
    EvaluationMode,
    EvaluationOutput,
    EvaluationRecommendation,
    PerClassMetrics,
)
from tests.conftest import make_valid_dataset

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_input(
    dataset_root: Path,
    mode: EvaluationMode = EvaluationMode.LOCAL_DRY_RUN,
    training_status: str | None = "completed",
    promotion_policy_path: str | None = None,
) -> EvaluationInput:
    return EvaluationInput(
        dataset_path=str(dataset_root),
        data_yaml_path=str(dataset_root / "data.yaml"),
        weights_path=str(dataset_root / "best.pt"),
        mode=mode,
        training_status=training_status,  # type: ignore[arg-type]
        promotion_policy_path=promotion_policy_path,
    )


def _passing_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        map50=0.80,
        map50_95=0.55,
        precision=0.75,
        recall=0.75,
        per_class_metrics={
            "scratch": PerClassMetrics(precision=0.78, recall=0.73, map50=0.80, map50_95=0.55),
            "dent": PerClassMetrics(precision=0.76, recall=0.74, map50=0.79, map50_95=0.54),
            "crack": PerClassMetrics(precision=0.77, recall=0.72, map50=0.81, map50_95=0.56),
        },
    )


def _failing_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        map50=0.50,
        map50_95=0.30,
        precision=0.55,
        recall=0.50,
        per_class_metrics={
            "scratch": PerClassMetrics(precision=0.55, recall=0.50, map50=0.50, map50_95=0.30),
            "dent": PerClassMetrics(precision=0.55, recall=0.50, map50=0.50, map50_95=0.30),
            "crack": PerClassMetrics(precision=0.55, recall=0.50, map50=0.50, map50_95=0.30),
        },
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_blocked_on_failed_training(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    agent = EvaluationAgent(artifacts_dir=tmp_path / "eval_out")
    result = agent.run(_make_input(tmp_path, training_status="failed"))

    assert result.success is False
    assert result.recommendation is None
    assert any("blocked" in e.lower() for e in result.errors)


def test_blocked_on_cancelled_training(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    agent = EvaluationAgent(artifacts_dir=tmp_path / "eval_out")
    result = agent.run(_make_input(tmp_path, training_status="cancelled"))

    assert result.success is False
    assert any("blocked" in e.lower() for e in result.errors)


def test_dry_run_creates_evaluation_request_json(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    out_dir = tmp_path / "eval_out"
    agent = EvaluationAgent(artifacts_dir=out_dir)
    agent.run(_make_input(tmp_path))

    assert (out_dir / "evaluation_request.json").exists()


def test_dry_run_creates_evaluation_report_json(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    out_dir = tmp_path / "eval_out"
    agent = EvaluationAgent(artifacts_dir=out_dir)
    agent.run(_make_input(tmp_path))

    assert (out_dir / "evaluation_report.json").exists()


def test_dry_run_creates_evaluation_report_md(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    out_dir = tmp_path / "eval_out"
    agent = EvaluationAgent(artifacts_dir=out_dir)
    result = agent.run(_make_input(tmp_path))

    md = out_dir / "evaluation_report.md"
    assert md.exists()
    assert result.evaluation_report_path == str(md)


def test_passing_metrics_produce_promote_candidate(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    agent = EvaluationAgent(
        artifacts_dir=tmp_path / "eval_out",
        dry_run_override_metrics=_passing_metrics(),
    )
    result = agent.run(_make_input(tmp_path))

    assert result.recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE
    assert not result.failed_checks


def test_failing_metrics_produce_retrain(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    agent = EvaluationAgent(
        artifacts_dir=tmp_path / "eval_out",
        dry_run_override_metrics=_failing_metrics(),
    )
    result = agent.run(_make_input(tmp_path))

    assert result.recommendation in {
        EvaluationRecommendation.RETRAIN,
        EvaluationRecommendation.NEED_LABEL_REVIEW,
    }
    assert result.failed_checks


def test_critical_class_low_recall_produces_need_label_review(tmp_path: Path) -> None:
    """crack recall below critical_class_recall_min → NEED_LABEL_REVIEW."""
    make_valid_dataset(tmp_path)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "thresholds:\n"
        "  map50_min: 0.50\n"
        "  map50_95_min: 0.30\n"
        "  precision_min: 0.50\n"
        "  recall_min: 0.50\n"
        "critical_class_recall_min: 0.80\n"
        "critical_classes:\n"
        "  - crack\n",
        encoding="utf-8",
    )
    # crack recall = 0.78 < 0.80 → critical fail
    metrics = EvaluationMetrics(
        map50=0.80,
        map50_95=0.55,
        precision=0.75,
        recall=0.75,
        per_class_metrics={
            "scratch": PerClassMetrics(precision=0.78, recall=0.79, map50=0.80, map50_95=0.55),
            "dent": PerClassMetrics(precision=0.76, recall=0.79, map50=0.79, map50_95=0.54),
            "crack": PerClassMetrics(precision=0.77, recall=0.78, map50=0.81, map50_95=0.56),
        },
    )
    agent = EvaluationAgent(
        artifacts_dir=tmp_path / "eval_out",
        dry_run_override_metrics=metrics,
    )
    result = agent.run(_make_input(tmp_path, promotion_policy_path=str(policy_path)))

    assert result.recommendation == EvaluationRecommendation.NEED_LABEL_REVIEW


def test_output_is_structured_evaluation_output(tmp_path: Path) -> None:
    make_valid_dataset(tmp_path)
    agent = EvaluationAgent(artifacts_dir=tmp_path / "eval_out")
    result = agent.run(_make_input(tmp_path))

    assert isinstance(result, EvaluationOutput)
    assert isinstance(result.metrics, EvaluationMetrics)
    assert result.recommendation is not None
    assert isinstance(result.passed_checks, list)
    assert isinstance(result.failed_checks, list)
