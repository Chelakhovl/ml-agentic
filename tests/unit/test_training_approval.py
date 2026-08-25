"""Unit tests for TrainingApprovalAgent (H4 gate)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_mlops.agents.training_approval import TrainingApprovalAgent
from agentic_mlops.cli.main import app
from agentic_mlops.contracts.training_approval import (
    TrainingApprovalAction,
    TrainingApprovalInput,
    TrainingApprovalStatus,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _write_dataset_report(
    path: Path,
    status: str = "passed",
    blocking_issues: list[str] | None = None,
    warnings: list[str] | None = None,
    num_images: int = 6,
) -> Path:
    report_path = path / "dataset_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "num_images": num_images,
        "num_labels": num_images,
        "blocking_issues": blocking_issues or [],
        "warnings": warnings or [],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _make_input(
    report_path: Path,
    out_dir: Path,
    action: TrainingApprovalAction,
    approver: str = "TestApprover",
    comment: str | None = None,
    force: bool = False,
) -> TrainingApprovalInput:
    return TrainingApprovalInput(
        dataset_report_path=str(report_path),
        approver=approver,
        output_dir=str(out_dir),
        interactive=False,
        action=action,
        comment=comment,
        force=force,
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_approve_training_produces_approved(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING)
    )

    assert result.success is True
    assert result.status == TrainingApprovalStatus.APPROVED
    assert result.action == TrainingApprovalAction.APPROVE_TRAINING


def test_reject_training_produces_rejected(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.REJECT_TRAINING)
    )

    assert result.success is True
    assert result.status == TrainingApprovalStatus.REJECTED


def test_cancel_produces_cancelled_and_unsuccessful(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(_make_input(report, tmp_path / "out", TrainingApprovalAction.CANCEL))

    assert result.success is False
    assert result.status == TrainingApprovalStatus.CANCELLED


def test_failed_validation_cannot_be_approved(tmp_path: Path) -> None:
    report = _write_dataset_report(
        tmp_path / "validation", status="failed", blocking_issues=["Missing label files"]
    )
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING)
    )

    assert result.success is False
    assert any("cannot approve" in e.lower() for e in result.errors)


def test_failed_validation_cannot_be_approved_even_with_force(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation", status="failed")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING, force=True)
    )

    assert result.success is False
    assert any("cannot approve" in e.lower() for e in result.errors)


def test_warning_status_blocked_without_force(tmp_path: Path) -> None:
    report = _write_dataset_report(
        tmp_path / "validation", status="warning", warnings=["Unusual aspect ratio"]
    )
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING, force=False)
    )

    assert result.success is False
    assert any("force" in e.lower() for e in result.errors)


def test_warning_status_succeeds_with_force(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation", status="warning")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(report, tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING, force=True)
    )

    assert result.success is True
    assert result.status == TrainingApprovalStatus.APPROVED


def test_non_interactive_requires_action(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        TrainingApprovalInput(
            dataset_report_path=str(report), output_dir=str(tmp_path / "out"), interactive=False
        )
    )
    assert result.success is False
    assert any("--action" in e for e in result.errors)


def test_unreadable_report_fails(tmp_path: Path) -> None:
    agent = TrainingApprovalAgent(artifacts_dir=tmp_path / "out")
    result = agent.run(
        _make_input(
            tmp_path / "nope.json", tmp_path / "out", TrainingApprovalAction.APPROVE_TRAINING
        )
    )
    assert result.success is False


def test_decision_json_is_created(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    out_dir = tmp_path / "out"
    agent = TrainingApprovalAgent(artifacts_dir=out_dir)
    agent.run(_make_input(report, out_dir, TrainingApprovalAction.REJECT_TRAINING))

    json_file = out_dir / "training_approval_decision.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["status"] == "rejected"
    assert data["action"] == "reject_training"
    assert data["approver"] == "TestApprover"


def test_decision_md_is_created(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    out_dir = tmp_path / "out"
    agent = TrainingApprovalAgent(artifacts_dir=out_dir)
    result = agent.run(
        _make_input(report, out_dir, TrainingApprovalAction.APPROVE_TRAINING, comment="Looks good")
    )

    md_file = out_dir / "training_approval_decision.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "APPROVED" in content
    assert "Looks good" in content
    assert result.generated_artifacts


def test_cli_non_interactive_mode(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "approve-training",
            str(report),
            "--output-dir",
            str(out_dir),
            "--approver",
            "CI",
            "--no-interactive",
            "--action",
            "reject_training",
            "--comment",
            "Automated test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "training_approval_decision.json").exists()
    data = json.loads((out_dir / "training_approval_decision.json").read_text(encoding="utf-8"))
    assert data["status"] == "rejected"
    assert data["approver"] == "CI"


def test_cli_invalid_action_exits_1(tmp_path: Path) -> None:
    report = _write_dataset_report(tmp_path / "validation")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "approve-training",
            str(report),
            "--no-interactive",
            "--action",
            "bogus",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid action" in result.output
