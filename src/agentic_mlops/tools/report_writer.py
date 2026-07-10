"""Utility for writing JSON and Markdown reports to the artifacts directory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.approvals import ApprovalOutput
from agentic_mlops.contracts.datasets import DatasetValidationOutput
from agentic_mlops.contracts.evaluation import EvaluationOutput
from agentic_mlops.contracts.training import TrainingOutput, training_mode_to_runner
from agentic_mlops.contracts.workflows import MVPWorkflowOutput
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class ReportWriter:
    """Writes structured reports to disk."""

    def write_dataset_validation_report(
        self,
        output: DatasetValidationOutput,
        artifacts_dir: Path,
        dataset_version: str = "unknown",
    ) -> tuple[Path, Path]:
        """Write dataset_quality_report.json and .md.

        Returns:
            (json_path, md_path)
        """
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "dataset_quality_report.json"
        md_path = artifacts_dir / "dataset_quality_report.md"

        payload: dict[str, Any] = {
            "dataset_version": dataset_version,
            "status": output.status,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "num_images": output.num_images,
            "num_labels": output.num_labels,
            "blocking_issues": output.blocking_issues,
            "warnings": output.warnings,
            "class_distribution": output.class_distribution,
            "recommendation": (
                "proceed_to_training"
                if output.status == "passed"
                else "fix_dataset_before_training"
            ),
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_dataset_report_md(payload), encoding="utf-8")

        logger.info("Reports written", extra={"json": str(json_path), "md": str(md_path)})
        return json_path, md_path


    def write_training_report(
        self,
        output: TrainingOutput,
        artifacts_dir: Path,
    ) -> Path:
        """Write training_report.md. Returns the path."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        md_path = artifacts_dir / "training_report.md"
        md_path.write_text(_training_report_md(output), encoding="utf-8")
        logger.info("Training report written", extra={"md": str(md_path)})
        return md_path


    def write_evaluation_report(
        self,
        output: EvaluationOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write evaluation_report.json and evaluation_report.md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "evaluation_report.json"
        md_path = artifacts_dir / "evaluation_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "mode": output.mode,
            "runner": output.runner,
            "model_path": output.model_path,
            "started_at": output.started_at,
            "completed_at": output.completed_at,
            "recommendation": output.recommendation,
            "metrics": output.metrics.model_dump(),
            "passed_checks": output.passed_checks,
            "failed_checks": output.failed_checks,
            "reasons": output.reasons,
            "artifacts": output.artifacts,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_evaluation_report_md(payload, output), encoding="utf-8")
        logger.info(
            "Evaluation reports written",
            extra={"json": str(json_path), "md": str(md_path)},
        )
        return json_path, md_path

    def write_approval_decision(
        self,
        output: ApprovalOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write approval_decision.json and approval_decision.md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "approval_decision.json"
        md_path = artifacts_dir / "approval_decision.md"

        payload: dict[str, Any] = {
            "status": output.status,
            "action": output.action,
            "approver": output.approver,
            "timestamp": output.timestamp,
            "evaluation_output_path": output.evaluation_output_path,
            "comment": output.comment,
            "approval_request": (
                output.approval_request.model_dump() if output.approval_request else None
            ),
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_approval_decision_md(output), encoding="utf-8")
        logger.info(
            "Approval decision written",
            extra={"json": str(json_path), "md": str(md_path)},
        )
        return json_path, md_path


    def write_workflow_summary(
        self,
        output: MVPWorkflowOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write workflow_summary.json and workflow_summary.md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "workflow_summary.json"
        md_path = artifacts_dir / "workflow_summary.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "workflow_status": str(output.workflow_status),
            "success": output.success,
            "message": output.message,
            "steps": [s.model_dump() for s in output.steps],
            "errors": output.errors,
        }
        if output.registration_status is not None:
            payload["registration_status"] = output.registration_status
        if output.registered_model_name is not None:
            payload["registered_model_name"] = output.registered_model_name
        if output.registered_model_version is not None:
            payload["registered_model_version"] = output.registered_model_version
        if output.registered_model_path is not None:
            payload["registered_model_path"] = output.registered_model_path

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_workflow_summary_md(output), encoding="utf-8")
        logger.info(
            "Workflow summary written",
            extra={"json": str(json_path), "md": str(md_path)},
        )
        return json_path, md_path


def _dataset_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Quality Report",
        "",
        f"**Status:** `{report['status'].upper()}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Dataset version:** {report['dataset_version']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Images | {report['num_images']} |",
        f"| Label files | {report['num_labels']} |",
        f"| Blocking issues | {len(report['blocking_issues'])} |",
        f"| Warnings | {len(report['warnings'])} |",
        "",
    ]

    if report["blocking_issues"]:
        lines += ["## Blocking Issues", ""]
        for issue in report["blocking_issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    if report["warnings"]:
        lines += ["## Warnings", ""]
        for warn in report["warnings"]:
            lines.append(f"- {warn}")
        lines.append("")

    if report["class_distribution"]:
        lines += ["## Class Distribution", "", "| Class | Count |", "|-------|-------|"]
        total = sum(report["class_distribution"].values())
        for cls, count in sorted(report["class_distribution"].items()):
            pct = count / total * 100 if total else 0
            lines.append(f"| {cls} | {count} ({pct:.1f}%) |")
        lines.append("")

    lines += [f"**Recommendation:** `{report['recommendation']}`", ""]
    return "\n".join(lines)


def _training_report_md(output: TrainingOutput) -> str:
    status_label = "COMPLETED" if output.success else "FAILED"
    lines = [
        "# Training Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {datetime.now(tz=UTC).isoformat()}  ",
        f"**Runner:** `{training_mode_to_runner(output.mode)}`",
        "",
        "## Job Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Job ID | `{output.job_id or 'N/A'}` |",
        f"| MLflow Run ID | `{output.mlflow_run_id or 'N/A'}` |",
        f"| Job Status | `{output.job_status}` |",
        f"| Best Weights | `{output.best_weights_path or 'N/A'}` |",
        "",
    ]

    if output.training_plan_path:
        lines += [
            "## Dry-Run Plan",
            "",
            f"Training plan written to: `{output.training_plan_path}`",
            "",
            "> No actual training was performed in dry-run mode.",
            "",
        ]

    if output.training_artifacts:
        lines += ["## Artifacts", ""]
        for artifact in output.training_artifacts:
            lines.append(f"- **{artifact.name}** (`{artifact.artifact_type}`): `{artifact.path}`")
        lines.append("")

    if output.errors:
        lines += ["## Errors", ""]
        for err in output.errors:
            lines.append(f"- {err}")
        lines.append("")

    lines.append(f"**Message:** {output.message}")
    return "\n".join(lines)


def _evaluation_report_md(report: dict[str, Any], output: EvaluationOutput) -> str:
    rec = report.get("recommendation") or "N/A"
    rec_color_map = {
        "promote_candidate": "PASS",
        "retrain": "FAIL",
        "need_label_review": "WARN",
        "collect_more_data": "WARN",
        "review_labels": "WARN",
        "needs_human_review": "WARN",
        "reject_candidate": "FAIL",
    }
    rec_label = rec_color_map.get(rec, rec)
    m = report["metrics"]
    lines = [
        "# Evaluation Report",
        "",
        f"**Status:** `{rec_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Mode:** `{report['mode']}`  ",
        f"**Runner:** `{report.get('runner') or 'N/A'}`  ",
        f"**Recommendation:** `{rec}`",
        "",
        "## Job Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Model path | `{report.get('model_path') or 'N/A'}` |",
        f"| Started at | {report.get('started_at') or 'N/A'} |",
        f"| Completed at | {report.get('completed_at') or 'N/A'} |",
        "",
        "## Global Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@0.5 | {m['map50']:.4f} |",
        f"| mAP@0.5:0.95 | {m['map50_95']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        "",
    ]

    if m.get("per_class_metrics"):
        lines += [
            "## Per-Class Metrics",
            "",
            "| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
            "|-------|-----------|--------|---------|--------------|",
        ]
        for cls, pcm in sorted(m["per_class_metrics"].items()):
            lines.append(
                f"| {cls} | {pcm['precision']:.4f} | {pcm['recall']:.4f}"
                f" | {pcm['map50']:.4f} | {pcm['map50_95']:.4f} |"
            )
        lines.append("")

    if report.get("passed_checks"):
        lines += ["## Passed Checks", ""]
        for chk in report["passed_checks"]:
            lines.append(f"- PASS {chk}")
        lines.append("")

    if report.get("failed_checks"):
        lines += ["## Failed Checks", ""]
        for chk in report["failed_checks"]:
            lines.append(f"- FAIL {chk}")
        lines.append("")

    if report.get("reasons"):
        lines += ["## Reasons", ""]
        for r in report["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if report.get("artifacts"):
        lines += ["## Artifacts", ""]
        for a in report["artifacts"]:
            lines.append(f"- `{a}`")
        lines.append("")

    lines.append(f"**Message:** {output.message}")
    return "\n".join(lines)


def _approval_decision_md(output: ApprovalOutput) -> str:
    req = output.approval_request
    status_label = (output.status or "N/A").upper()
    lines = [
        "# Human Approval Decision",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Action:** `{output.action or 'N/A'}`  ",
        f"**Approver:** {output.approver or 'N/A'}  ",
        f"**Timestamp:** {output.timestamp}",
        "",
    ]

    if req:
        lines += [
            "## Model Candidate",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Candidate model | `{req.candidate_model}` |",
            f"| Dataset | `{req.dataset_version_or_path}` |",
            f"| Evaluation status | `{req.evaluation_status}` |",
            f"| Recommendation | `{req.recommendation or 'N/A (blocked)'}` |",
            "",
        ]

        if req.key_metrics:
            lines += [
                "## Evaluation Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
            for k, v in req.key_metrics.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

        if req.reasons:
            lines += ["## Reasons / Issues", ""]
            for r in req.reasons:
                lines.append(f"- {r}")
            lines.append("")

        if req.next_actions:
            lines += ["## Next Actions", ""]
            for a in req.next_actions:
                lines.append(f"- {a}")
            lines.append("")

    if output.comment:
        lines += ["## Human Comment", "", output.comment, ""]

    lines.append(f"**Message:** {output.message}")
    return "\n".join(lines)


def _workflow_summary_md(output: MVPWorkflowOutput) -> str:
    status_label = str(output.workflow_status).upper()
    lines = [
        "# MVP Workflow Summary",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {datetime.now(tz=UTC).isoformat()}",
        "",
        "## Steps",
        "",
        "| Step | Status | Success |",
        "|------|--------|---------|",
    ]
    for step in output.steps:
        ok = "yes" if step.success else "no"
        lines.append(f"| {step.step} | {step.status} | {ok} |")
    lines.append("")

    for step in output.steps:
        if step.errors:
            lines += [f"### {step.step.capitalize()} Errors", ""]
            for err in step.errors:
                lines.append(f"- {err}")
            lines.append("")

    if output.errors:
        lines += ["## Workflow Errors", ""]
        for err in output.errors:
            lines.append(f"- {err}")
        lines.append("")

    # MLflow tracking section
    lines += ["## MLflow Tracking", ""]
    if output.mlflow_run_id:
        lines += [
            "| Field | Value |",
            "|-------|-------|",
            "| Enabled | yes |",
            f"| Experiment | `{output.mlflow_experiment_name or 'N/A'}` |",
            f"| Run ID | `{output.mlflow_run_id}` |",
            f"| Tracking URI | `{output.mlflow_tracking_uri or 'N/A'}` |",
            "",
        ]
    else:
        lines += ["MLflow: disabled", ""]

    lines.append(f"**Message:** {output.message}")
    return "\n".join(lines)
