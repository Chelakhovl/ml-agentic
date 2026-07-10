"""Unit tests for the Model Decision Agent, decider tool, and CLI command.

Coverage matrix:
    1.  Cannot read evaluation_report_path -> failed (success=False)
    2.  Evaluation success=False -> decision REJECT
    3.  Evaluation success=True but recommendation=None -> decision REJECT
    4.  Recommendation mapping: promote_candidate/retrain/need_label_review/
        collect_more_data/review_labels/needs_human_review/reject_candidate
    5.  Baseline improvement meets requirement -> stays PROMOTE, passed_checks noted
    6.  Baseline improvement below requirement -> PROMOTE downgraded to RETRAIN
    7.  require_improvement_over_baseline=False -> no downgrade even if improvement tiny
    8.  A non-PROMOTE decision is not further downgraded by a failing baseline check
    9.  Model size within budget -> no downgrade, passed_checks noted
   10.  Model size over budget -> PROMOTE downgraded to RETRAIN
   11.  model_path missing on disk -> reason recorded, no crash
   12.  Measured latency within budget -> no downgrade
   13.  Measured latency over budget -> PROMOTE downgraded to RETRAIN
   14.  No policy/baseline/runtime config -> pure recommendation mapping, no crash
   15.  ModelDecisionAgent writes decision_report.json / .md
   16.  ModelDecisionAgent logs to MLflow when enabled
   17.  CLI: model-decision command exists and succeeds (exit 0) for a PROMOTE decision
   18.  CLI: model-decision still exits 0 for a REJECT decision (decision != failure)
   19.  CLI: model-decision exits 1 when evaluation_report_path is missing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.agents.model_decision import ModelDecisionAgent
from agentic_mlops.contracts.model_decision import ModelDecision, ModelDecisionInput
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.model_decider import ModelDecider

# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_eval_report(
    path: Path,
    *,
    success: bool = True,
    recommendation: str | None = "promote_candidate",
    map50: float = 0.9,
    model_path: str | None = None,
    failed_checks: list[str] | None = None,
    message: str = "ok",
) -> str:
    data = {
        "success": success,
        "message": message,
        "recommendation": recommendation,
        "metrics": {
            "map50": map50,
            "map50_95": map50 * 0.7,
            "precision": 0.85,
            "recall": 0.8,
            "per_class_metrics": {},
        },
        "passed_checks": [],
        "failed_checks": failed_checks or [],
        "model_path": model_path,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ── 1-3. Structural / evaluation-failure cases ────────────────────────────────


def test_missing_evaluation_report_fails(tmp_path: Path) -> None:
    result = ModelDecider().decide(
        ModelDecisionInput(evaluation_report_path=str(tmp_path / "nope.json"))
    )
    assert result.success is False


def test_evaluation_not_successful_is_reject(tmp_path: Path) -> None:
    report = _write_eval_report(
        tmp_path / "eval.json", success=False, recommendation=None, message="training failed"
    )
    result = ModelDecider().decide(ModelDecisionInput(evaluation_report_path=report))

    assert result.success is True
    assert result.decision == ModelDecision.REJECT
    assert "training failed" in result.reasons[0]


def test_missing_recommendation_is_reject(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json", success=True, recommendation=None)
    result = ModelDecider().decide(ModelDecisionInput(evaluation_report_path=report))

    assert result.decision == ModelDecision.REJECT


# ── 4. Recommendation mapping ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        ("promote_candidate", ModelDecision.PROMOTE),
        ("retrain", ModelDecision.RETRAIN),
        ("need_label_review", ModelDecision.NEED_LABEL_REVIEW),
        ("collect_more_data", ModelDecision.NEED_MORE_DATA),
        ("review_labels", ModelDecision.NEED_LABEL_REVIEW),
        ("needs_human_review", ModelDecision.RETRAIN),
        ("reject_candidate", ModelDecision.REJECT),
    ],
)
def test_recommendation_mapping(
    tmp_path: Path, recommendation: str, expected: ModelDecision
) -> None:
    report = _write_eval_report(tmp_path / "eval.json", recommendation=recommendation)
    result = ModelDecider().decide(ModelDecisionInput(evaluation_report_path=report))

    assert result.decision == expected


# ── 5-8. Baseline comparison ───────────────────────────────────────────────────


def test_baseline_improvement_meets_requirement_stays_promote(tmp_path: Path) -> None:
    baseline = _write_eval_report(tmp_path / "baseline.json", map50=0.80)
    candidate = _write_eval_report(tmp_path / "candidate.json", map50=0.85)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "require_improvement_over_baseline: true\nbaseline_improvement_min_map50: 0.02\n",
        encoding="utf-8",
    )

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=candidate,
            baseline_report_path=baseline,
            promotion_policy_path=str(policy_path),
        )
    )

    assert result.decision == ModelDecision.PROMOTE
    assert result.map50_improvement == pytest.approx(0.05)
    assert any("improved" in c for c in result.passed_checks)


def test_baseline_improvement_below_requirement_downgrades_promote(tmp_path: Path) -> None:
    baseline = _write_eval_report(tmp_path / "baseline.json", map50=0.80)
    candidate = _write_eval_report(tmp_path / "candidate.json", map50=0.805)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "require_improvement_over_baseline: true\nbaseline_improvement_min_map50: 0.02\n",
        encoding="utf-8",
    )

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=candidate,
            baseline_report_path=baseline,
            promotion_policy_path=str(policy_path),
        )
    )

    assert result.decision == ModelDecision.RETRAIN
    assert any("below required" in r for r in result.reasons)


def test_baseline_check_disabled_by_policy_no_downgrade(tmp_path: Path) -> None:
    baseline = _write_eval_report(tmp_path / "baseline.json", map50=0.80)
    candidate = _write_eval_report(tmp_path / "candidate.json", map50=0.801)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("require_improvement_over_baseline: false\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=candidate,
            baseline_report_path=baseline,
            promotion_policy_path=str(policy_path),
        )
    )

    assert result.decision == ModelDecision.PROMOTE


def test_non_promote_decision_not_further_downgraded_by_baseline(tmp_path: Path) -> None:
    baseline = _write_eval_report(tmp_path / "baseline.json", map50=0.80)
    candidate = _write_eval_report(
        tmp_path / "candidate.json", map50=0.70, recommendation="need_label_review"
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "require_improvement_over_baseline: true\nbaseline_improvement_min_map50: 0.02\n",
        encoding="utf-8",
    )

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=candidate,
            baseline_report_path=baseline,
            promotion_policy_path=str(policy_path),
        )
    )

    assert result.decision == ModelDecision.NEED_LABEL_REVIEW


# ── 9-11. Model size budget ────────────────────────────────────────────────────


def test_model_size_within_budget_no_downgrade(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * (2 * 1024 * 1024))
    report = _write_eval_report(tmp_path / "eval.json", model_path=str(weights))
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text("runtime:\n  max_model_size_mb: 5.0\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(evaluation_report_path=report, evaluation_config_path=str(cfg_path))
    )

    assert result.decision == ModelDecision.PROMOTE
    assert result.model_size_mb == pytest.approx(2.0)
    assert any("Model size" in c for c in result.passed_checks)


def test_model_size_over_budget_downgrades_promote(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * (6 * 1024 * 1024))
    report = _write_eval_report(tmp_path / "eval.json", model_path=str(weights))
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text("runtime:\n  max_model_size_mb: 5.0\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(evaluation_report_path=report, evaluation_config_path=str(cfg_path))
    )

    assert result.decision == ModelDecision.RETRAIN
    assert any("exceeds max" in r for r in result.reasons)


def test_model_path_missing_on_disk_records_reason(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json", model_path=str(tmp_path / "nope.pt"))
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text("runtime:\n  max_model_size_mb: 5.0\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(evaluation_report_path=report, evaluation_config_path=str(cfg_path))
    )

    assert result.success is True
    assert any("not found" in r for r in result.reasons)


# ── 12-13. Latency budget ──────────────────────────────────────────────────────


def test_latency_within_budget_no_downgrade(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json")
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text("runtime:\n  max_latency_ms: 40\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=report,
            evaluation_config_path=str(cfg_path),
            measured_latency_ms=25.0,
        )
    )

    assert result.decision == ModelDecision.PROMOTE
    assert any("Latency" in c for c in result.passed_checks)


def test_latency_over_budget_downgrades_promote(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json")
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text("runtime:\n  max_latency_ms: 40\n", encoding="utf-8")

    result = ModelDecider().decide(
        ModelDecisionInput(
            evaluation_report_path=report,
            evaluation_config_path=str(cfg_path),
            measured_latency_ms=55.0,
        )
    )

    assert result.decision == ModelDecision.RETRAIN
    assert any("Measured latency" in r for r in result.reasons)


# ── 14. No optional config at all ─────────────────────────────────────────────


def test_bare_recommendation_mapping_without_extras(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json")
    result = ModelDecider().decide(ModelDecisionInput(evaluation_report_path=report))

    assert result.decision == ModelDecision.PROMOTE
    assert result.reasons == ["All promotion checks passed."]


# ── 15-16. ModelDecisionAgent ──────────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json")
    artifacts_dir = tmp_path / "artifacts"

    agent = ModelDecisionAgent(artifacts_dir=artifacts_dir)
    result = agent.run(ModelDecisionInput(evaluation_report_path=report))

    assert (artifacts_dir / "decision_report.json").exists()
    assert (artifacts_dir / "decision_report.md").exists()
    assert result.decision_report_path == str(artifacts_dir / "decision_report.json")

    data = json.loads((artifacts_dir / "decision_report.json").read_text(encoding="utf-8"))
    assert data["decision"] == "promote"


def test_agent_logs_to_mlflow_when_enabled(tmp_path: Path) -> None:
    report = _write_eval_report(tmp_path / "eval.json")
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = ModelDecisionAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(ModelDecisionInput(evaluation_report_path=report))

    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "model_decision"
    assert tags.get("model_decision") == "promote"
    assert len(client.runs[run_id]["artifacts"]) > 0


# ── 17-19. CLI ─────────────────────────────────────────────────────────────────


def test_cli_model_decision_promote_exits_0(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    report = _write_eval_report(tmp_path / "eval.json")

    result = CliRunner().invoke(app, ["model-decision", report])

    assert result.exit_code == 0, result.output
    assert "PROMOTE" in result.output


def test_cli_model_decision_reject_still_exits_0(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    report = _write_eval_report(tmp_path / "eval.json", success=False, recommendation=None)

    result = CliRunner().invoke(app, ["model-decision", report])

    assert result.exit_code == 0, result.output
    assert "REJECT" in result.output


def test_cli_model_decision_exits_1_on_missing_report(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(app, ["model-decision", str(tmp_path / "nope.json")])

    assert result.exit_code == 1
