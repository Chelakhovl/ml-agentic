"""Unit tests for HumanApprovalAgent (MVP Agent 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_mlops.agents.human_approval import HumanApprovalAgent
from agentic_mlops.cli.main import app
from agentic_mlops.contracts.approvals import (
    ApprovalAction,
    ApprovalInput,
    ApprovalStatus,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _write_eval_report(
    path: Path,
    recommendation: str | None = "promote_candidate",
    metrics: dict | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
    message: str = "Dry-run evaluation complete.",
) -> Path:
    report_path = path / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-06-12T12:00:00+00:00",
        "mode": "local_dry_run",
        "recommendation": recommendation,
        "metrics": metrics or {"map50": 0.86, "map50_95": 0.59, "precision": 0.84, "recall": 0.79},
        "passed_checks": passed_checks or ["map50 0.8620 >= 0.7500"],
        "failed_checks": failed_checks or [],
        "message": message,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _make_input(
    eval_report_path: Path,
    out_dir: Path,
    action: ApprovalAction,
    approver: str = "TestApprover",
    comment: str | None = None,
    force: bool = False,
) -> ApprovalInput:
    return ApprovalInput(
        evaluation_output_path=str(eval_report_path),
        approver=approver,
        output_dir=str(out_dir),
        interactive=False,
        action=action,
        comment=comment,
        force=force,
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_approve_model_produces_approved(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(_make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL))

    assert result.success is True
    assert result.status == ApprovalStatus.APPROVED
    assert result.action == ApprovalAction.APPROVE_MODEL


def test_reject_model_produces_rejected(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(_make_input(eval_path, tmp_path / "approval", ApprovalAction.REJECT_MODEL))

    assert result.success is True
    assert result.status == ApprovalStatus.REJECTED
    assert result.action == ApprovalAction.REJECT_MODEL


def test_request_retraining_produces_needs_retraining(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.REQUEST_RETRAINING)
    )

    assert result.success is True
    assert result.status == ApprovalStatus.NEEDS_RETRAINING


def test_request_more_data_produces_needs_more_data(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.REQUEST_MORE_DATA)
    )

    assert result.success is True
    assert result.status == ApprovalStatus.NEEDS_MORE_DATA


def test_request_label_review_produces_needs_label_review(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.REQUEST_LABEL_REVIEW)
    )

    assert result.success is True
    assert result.status == ApprovalStatus.NEEDS_LABEL_REVIEW


def test_failed_evaluation_cannot_be_approved(tmp_path: Path) -> None:
    """recommendation=None means blocked/failed evaluation — approve must be refused."""
    eval_path = _write_eval_report(
        tmp_path / "eval",
        recommendation=None,
        message="Evaluation blocked: upstream training status is 'failed'.",
    )
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(_make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL))

    assert result.success is False
    assert any("cannot approve" in e.lower() for e in result.errors)


def test_failed_evaluation_cannot_be_approved_even_with_force(tmp_path: Path) -> None:
    """--force must not override the failed-evaluation block."""
    eval_path = _write_eval_report(tmp_path / "eval", recommendation=None)
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL, force=True)
    )

    assert result.success is False
    assert any("cannot approve" in e.lower() for e in result.errors)


def test_risky_approval_blocked_without_force(tmp_path: Path) -> None:
    """retrain recommendation + approve_model + no --force → error."""
    eval_path = _write_eval_report(
        tmp_path / "eval",
        recommendation="retrain",
        failed_checks=["map50 0.50 < 0.75"],
    )
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL, force=False)
    )

    assert result.success is False
    assert any("force" in e.lower() for e in result.errors)


def test_risky_approval_succeeds_with_force(tmp_path: Path) -> None:
    """retrain recommendation + approve_model + --force → approved."""
    eval_path = _write_eval_report(
        tmp_path / "eval",
        recommendation="retrain",
        failed_checks=["map50 0.50 < 0.75"],
    )
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(
        _make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL, force=True)
    )

    assert result.success is True
    assert result.status == ApprovalStatus.APPROVED


def test_approval_decision_json_is_created(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    out_dir = tmp_path / "approval"
    agent = HumanApprovalAgent(artifacts_dir=out_dir)
    agent.run(_make_input(eval_path, out_dir, ApprovalAction.REJECT_MODEL))

    json_file = out_dir / "approval_decision.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["status"] == "rejected"
    assert data["action"] == "reject_model"
    assert data["approver"] == "TestApprover"


def test_approval_decision_md_is_created(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    out_dir = tmp_path / "approval"
    agent = HumanApprovalAgent(artifacts_dir=out_dir)
    result = agent.run(
        _make_input(
            eval_path, out_dir, ApprovalAction.REQUEST_RETRAINING, comment="Needs more data"
        )
    )

    md_file = out_dir / "approval_decision.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "NEEDS_RETRAINING" in content
    assert "Needs more data" in content
    assert result.generated_artifacts  # both json and md in list


def test_approval_request_built_from_eval_report(tmp_path: Path) -> None:
    eval_path = _write_eval_report(
        tmp_path / "eval",
        recommendation="promote_candidate",
        metrics={"map50": 0.90, "map50_95": 0.60, "precision": 0.88, "recall": 0.85},
    )
    agent = HumanApprovalAgent(artifacts_dir=tmp_path / "approval")
    result = agent.run(_make_input(eval_path, tmp_path / "approval", ApprovalAction.APPROVE_MODEL))

    assert result.approval_request is not None
    assert result.approval_request.recommendation == "promote_candidate"
    assert result.approval_request.key_metrics["map50"] == pytest.approx(0.90)


def test_cli_non_interactive_mode(tmp_path: Path) -> None:
    eval_path = _write_eval_report(tmp_path / "eval")
    out_dir = tmp_path / "approval"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "approve",
            "--evaluation-output",
            str(eval_path),
            "--output-dir",
            str(out_dir),
            "--approver",
            "CI",
            "--no-interactive",
            "--action",
            "request_retraining",
            "--comment",
            "Automated test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "approval_decision.json").exists()
    data = json.loads((out_dir / "approval_decision.json").read_text(encoding="utf-8"))
    assert data["status"] == "needs_retraining"
    assert data["approver"] == "CI"
