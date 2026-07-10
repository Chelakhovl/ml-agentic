"""Utility for writing JSON and Markdown reports to the artifacts directory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.annotation import AnnotationOutput
from agentic_mlops.contracts.approvals import ApprovalOutput
from agentic_mlops.contracts.data_intake import DataIntakeOutput
from agentic_mlops.contracts.dataset_structuring import DatasetStructuringOutput
from agentic_mlops.contracts.dataset_versioning import DatasetVersioningOutput
from agentic_mlops.contracts.datasets import DatasetValidationOutput
from agentic_mlops.contracts.deployment import DeploymentOutput
from agentic_mlops.contracts.evaluation import EvaluationOutput
from agentic_mlops.contracts.label_qa import LabelQAOutput
from agentic_mlops.contracts.model_decision import ModelDecisionOutput
from agentic_mlops.contracts.monitoring import MonitoringOutput
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


    def write_data_intake_report(
        self,
        output: DataIntakeOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write dataset_manifest.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "dataset_manifest.json"
        md_path = artifacts_dir / "dataset_manifest.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "status": output.status,
            "dataset_name": output.dataset_name,
            "source": output.source,
            "num_files": output.num_files,
            "valid_images": output.valid_images,
            "corrupted_images": output.corrupted_images,
            "duplicate_groups": output.duplicate_groups,
            "unexpected_format_files": output.unexpected_format_files,
            "pillow_available": output.pillow_available,
            "images": [r.model_dump() for r in output.image_records],
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_data_intake_report_md(payload), encoding="utf-8")

        logger.info(
            "Data intake manifest written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_annotation_report(
        self,
        output: AnnotationOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write pseudo_label_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "pseudo_label_report.json"
        md_path = artifacts_dir / "pseudo_label_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "pseudo_labels_path": output.pseudo_labels_path,
            "review_queue_path": output.review_queue_path,
            "num_images_processed": output.num_images_processed,
            "num_images_skipped_existing": output.num_images_skipped_existing,
            "high_confidence_count": output.high_confidence_count,
            "medium_confidence_count": output.medium_confidence_count,
            "low_confidence_count": output.low_confidence_count,
            "records": [r.model_dump() for r in output.records],
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_annotation_report_md(payload), encoding="utf-8")

        logger.info(
            "Pseudo-label report written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_deployment_report(
        self,
        output: DeploymentOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write deployment_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "deployment_report.json"
        md_path = artifacts_dir / "deployment_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "status": output.status,
            "endpoint_name": output.endpoint_name,
            "exported_model_path": output.exported_model_path,
            "release": output.release,
            "smoke_test_results": output.smoke_test_results,
            "block_reason": output.block_reason,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_deployment_report_md(payload), encoding="utf-8")

        logger.info(
            "Deployment report written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_model_decision_report(
        self,
        output: ModelDecisionOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write decision_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "decision_report.json"
        md_path = artifacts_dir / "decision_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "decision": output.decision,
            "required_approval_gate": output.required_approval_gate,
            "reasons": output.reasons,
            "passed_checks": output.passed_checks,
            "failed_checks": output.failed_checks,
            "metrics": output.metrics.model_dump() if output.metrics else None,
            "baseline_metrics": (
                output.baseline_metrics.model_dump() if output.baseline_metrics else None
            ),
            "map50_improvement": output.map50_improvement,
            "model_size_mb": output.model_size_mb,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_model_decision_report_md(payload), encoding="utf-8")

        logger.info(
            "Model decision report written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_dataset_versioning_report(
        self,
        output: DatasetVersioningOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write dataset_version_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "dataset_version_report.json"
        md_path = artifacts_dir / "dataset_version_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "status": output.status,
            "dataset_name": output.dataset_name,
            "version": output.version,
            "dataset_version_path": output.dataset_version_path,
            "hash": output.hash,
            "lineage": output.lineage.model_dump(mode="json") if output.lineage else None,
            "block_reason": output.block_reason,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_dataset_versioning_report_md(payload), encoding="utf-8")

        logger.info(
            "Dataset version report written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_dataset_structuring_report(
        self,
        output: DatasetStructuringOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write split_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "split_report.json"
        md_path = artifacts_dir / "split_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "structured_dataset_path": output.structured_dataset_path,
            "data_yaml_path": output.data_yaml_path,
            "num_images": output.num_images,
            "num_labels": output.num_labels,
            "split_counts": output.split_counts,
            "classes": output.classes,
            "assignments": [a.model_dump() for a in output.assignments],
            "warnings": output.warnings,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_dataset_structuring_report_md(payload), encoding="utf-8")

        logger.info(
            "Dataset structuring report written",
            extra={"json": str(json_path), "md": str(md_path)},
        )
        return json_path, md_path

    def write_label_qa_report(
        self,
        output: LabelQAOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write label_quality_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "label_quality_report.json"
        md_path = artifacts_dir / "label_quality_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "status": output.status,
            "label_quality_score": output.label_quality_score,
            "num_images_checked": output.num_images_checked,
            "num_labels_checked": output.num_labels_checked,
            "reference_model_used": output.reference_model_used,
            "class_distribution": output.class_distribution,
            "suspicious_samples": [s.model_dump() for s in output.suspicious_samples],
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_label_qa_report_md(payload), encoding="utf-8")

        logger.info(
            "Label QA report written", extra={"json": str(json_path), "md": str(md_path)}
        )
        return json_path, md_path

    def write_monitoring_report(
        self,
        output: MonitoringOutput,
        artifacts_dir: Path,
    ) -> tuple[Path, Path]:
        """Write monitoring_report.json and .md. Returns (json_path, md_path)."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "monitoring_report.json"
        md_path = artifacts_dir / "monitoring_report.md"

        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "success": output.success,
            "status": output.status,
            "endpoint_name": output.endpoint_name,
            "model_version": output.model_version,
            "window_start": output.window_start,
            "window_end": output.window_end,
            "total_predictions": output.total_predictions,
            "metrics": output.metrics,
            "class_distribution": output.class_distribution,
            "drift_detected": output.drift_detected,
            "new_classes_detected": output.new_classes_detected,
            "triggered_alerts": output.triggered_alerts,
            "recommended_action": output.recommended_action,
            "requires_human_review": output.requires_human_review,
            "hard_samples_manifest_path": output.hard_samples_manifest_path,
            "num_hard_samples": len(output.hard_samples),
            "warnings": output.warnings,
            "message": output.message,
        }

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_monitoring_report_md(payload), encoding="utf-8")

        logger.info(
            "Monitoring report written", extra={"json": str(json_path), "md": str(md_path)}
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


def _annotation_report_md(report: dict[str, Any]) -> str:
    status_label = "SUCCESS" if report["success"] else "FAILED"
    lines = [
        "# Pseudo-Label Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Pseudo labels path:** `{report['pseudo_labels_path'] or 'N/A'}`  ",
        f"**Review queue:** `{report['review_queue_path'] or 'N/A'}`",
        "",
        "## Confidence Routing",
        "",
        "| Bucket | Count |",
        "|--------|-------|",
        f"| High (auto-candidate, sample-audited) | {report['high_confidence_count']} |",
        f"| Medium (human review) | {report['medium_confidence_count']} |",
        f"| Low (hard sample / expert review) | {report['low_confidence_count']} |",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Images processed | {report['num_images_processed']} |",
        f"| Images skipped (already labeled) | {report['num_images_skipped_existing']} |",
        "",
    ]

    review_records = [
        r for r in report["records"] if r["bucket"] in ("medium", "low")
    ]
    if review_records:
        lines += [
            "## Review Queue (medium/low confidence)",
            "",
            "| Image | Bucket | Detections | Min Conf | Mean Conf |",
            "|-------|--------|------------|----------|-----------|",
        ]
        for r in review_records[:200]:
            min_c = f"{r['min_confidence']:.2f}" if r["min_confidence"] is not None else "N/A"
            mean_c = f"{r['mean_confidence']:.2f}" if r["mean_confidence"] is not None else "N/A"
            lines.append(
                f"| {r['image']} | {r['bucket']} | {r['num_detections']} | {min_c} | {mean_c} |"
            )
        if len(review_records) > 200:
            lines.append(f"| ... | ... | ... | ... | {len(review_records) - 200} more not shown |")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _deployment_report_md(report: dict[str, Any]) -> str:
    status_label = str(report["status"]).upper()
    lines = [
        "# Deployment Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Endpoint:** {report['endpoint_name'] or 'N/A'}  ",
        f"**Release:** {report['release'] if report['release'] is not None else 'N/A'}",
        "",
    ]

    if report["block_reason"]:
        lines += ["## Blocked", "", report["block_reason"], ""]

    if report["exported_model_path"]:
        lines += [f"**Exported model:** `{report['exported_model_path']}`", ""]

    if report["smoke_test_results"]:
        lines += ["## Smoke Tests", ""]
        for chk in report["smoke_test_results"]:
            lines.append(f"- {chk}")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _monitoring_report_md(report: dict[str, Any]) -> str:
    status_label = str(report["status"]).upper()
    lines = [
        "# Monitoring Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Endpoint:** {report['endpoint_name'] or 'N/A'}  ",
        f"**Model version:** {report['model_version'] or 'N/A'}  ",
        f"**Window:** {report['window_start'] or 'N/A'} -> {report['window_end'] or 'N/A'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Predictions in window | {report['total_predictions']} |",
        f"| Hard samples | {report['num_hard_samples']} |",
        f"| Drift detected | {'yes' if report['drift_detected'] else 'no'} |",
        f"| Requires human review | {'yes' if report['requires_human_review'] else 'no'} |",
        f"| Recommended action | `{report['recommended_action']}` |",
        "",
    ]

    m = report["metrics"]
    if m:
        lines += ["## Metrics", "", "| Metric | Value |", "|--------|-------|"]
        for k, v in m.items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    if report["class_distribution"]:
        lines += ["## Class Distribution", "", "| Class | Proportion |", "|-------|------------|"]
        for cls, prop in sorted(report["class_distribution"].items()):
            lines.append(f"| {cls} | {prop:.2%} |")
        lines.append("")

    if report["new_classes_detected"]:
        lines += ["## New Classes Detected", ""]
        for cls in report["new_classes_detected"]:
            lines.append(f"- {cls}")
        lines.append("")

    if report["triggered_alerts"]:
        lines += ["## Triggered Alerts", ""]
        for alert in report["triggered_alerts"]:
            lines.append(f"- {alert}")
        lines.append("")

    if report["warnings"]:
        lines += ["## Warnings", ""]
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _model_decision_report_md(report: dict[str, Any]) -> str:
    decision_label = str(report["decision"]).upper() if report["decision"] else "N/A"
    lines = [
        "# Model Decision Report",
        "",
        f"**Decision:** `{decision_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Required approval gate:** `{report['required_approval_gate']}`",
        "",
    ]

    m = report.get("metrics")
    if m:
        lines += [
            "## Candidate Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@0.5 | {m['map50']:.4f} |",
            f"| mAP@0.5:0.95 | {m['map50_95']:.4f} |",
            f"| Precision | {m['precision']:.4f} |",
            f"| Recall | {m['recall']:.4f} |",
            "",
        ]

    if report.get("baseline_metrics") is not None:
        b = report["baseline_metrics"]
        improvement = report.get("map50_improvement")
        lines += [
            "## Baseline Comparison",
            "",
            "| Metric | Baseline | Candidate | Δ mAP50 |",
            "|--------|----------|-----------|---------|",
            (
                f"| mAP@0.5 | {b['map50']:.4f} | {m['map50']:.4f} | "
                f"{improvement:+.4f} |"
            ),
            "",
        ]

    if report.get("model_size_mb") is not None:
        lines += [f"**Model size:** {report['model_size_mb']:.1f} MB", ""]

    if report["reasons"]:
        lines += ["## Reasons", ""]
        for r in report["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if report["passed_checks"]:
        lines += ["## Passed Checks", ""]
        for chk in report["passed_checks"]:
            lines.append(f"- PASS {chk}")
        lines.append("")

    if report["failed_checks"]:
        lines += ["## Failed Checks", ""]
        for chk in report["failed_checks"]:
            lines.append(f"- FAIL {chk}")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _dataset_versioning_report_md(report: dict[str, Any]) -> str:
    status_label = str(report["status"]).upper()
    lines = [
        "# Dataset Version Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Dataset:** {report['dataset_name'] or 'N/A'}  ",
        f"**Version:** {report['version'] if report['version'] is not None else 'N/A'}  ",
        f"**Hash:** `{report['hash'] or 'N/A'}`",
        "",
    ]

    if report["block_reason"]:
        lines += ["## Blocked", "", report["block_reason"], ""]

    lineage = report["lineage"]
    if lineage:
        lines += [
            "## Lineage",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Parent version | {lineage.get('parent_version') or 'N/A'} |",
            f"| Workflow ID | {lineage.get('workflow_id') or 'N/A'} |",
            f"| Approved by | {lineage.get('approved_by') or 'N/A'} |",
            f"| Validation status | {lineage.get('validation_status') or 'N/A'} |",
            f"| Label QA status | {lineage.get('label_qa_status') or 'N/A'} |",
            f"| Registered at | {lineage.get('registered_at') or 'N/A'} |",
            "",
        ]
        if lineage.get("classes"):
            lines += ["## Classes", ""]
            for i, cls in enumerate(lineage["classes"]):
                lines.append(f"{i}. {cls}")
            lines.append("")
        if lineage.get("source_batches"):
            lines += ["## Source Batches", ""]
            for b in lineage["source_batches"]:
                lines.append(f"- {b}")
            lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _dataset_structuring_report_md(report: dict[str, Any]) -> str:
    status_label = "SUCCESS" if report["success"] else "FAILED"
    lines = [
        "# Dataset Split Report",
        "",
        f"**Status:** `{status_label}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Structured dataset path:** `{report['structured_dataset_path'] or 'N/A'}`  ",
        f"**data.yaml:** `{report['data_yaml_path'] or 'N/A'}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Images | {report['num_images']} |",
        f"| Label files with objects | {report['num_labels']} |",
        "",
    ]

    if report["split_counts"]:
        lines += ["## Split Counts", "", "| Split | Count |", "|-------|-------|"]
        for split, count in sorted(report["split_counts"].items()):
            lines.append(f"| {split} | {count} |")
        lines.append("")

    if report["classes"]:
        lines += ["## Classes", ""]
        for i, cls in enumerate(report["classes"]):
            lines.append(f"{i}. {cls}")
        lines.append("")

    if report["warnings"]:
        lines += ["## Warnings", ""]
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _data_intake_report_md(report: dict[str, Any]) -> str:
    dupes = report["duplicate_groups"]
    corrupted = report["corrupted_images"]
    unexpected = report["unexpected_format_files"]
    lines = [
        "# Dataset Manifest",
        "",
        f"**Status:** `{report['status'].upper()}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Dataset name:** {report['dataset_name'] or 'N/A'}  ",
        f"**Source:** {report['source'] or 'N/A (needs human confirmation)'}  ",
        "**Pillow available:** "
        + ("yes" if report["pillow_available"] else "no (best-effort checks only)"),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Files scanned | {report['num_files']} |",
        f"| Valid images | {report['valid_images']} |",
        f"| Corrupted images | {len(corrupted)} |",
        f"| Duplicate groups | {len(dupes)} |",
        f"| Unexpected format files | {len(unexpected)} |",
        "",
    ]

    if corrupted:
        lines += ["## Corrupted Images", ""]
        for c in corrupted[:200]:
            lines.append(f"- {c}")
        if len(corrupted) > 200:
            lines.append(f"- ... {len(corrupted) - 200} more not shown")
        lines.append("")

    if dupes:
        lines += ["## Duplicate Groups", ""]
        for group in dupes[:200]:
            lines.append(f"- {', '.join(group)}")
        if len(dupes) > 200:
            lines.append(f"- ... {len(dupes) - 200} more not shown")
        lines.append("")

    if unexpected:
        lines += ["## Unexpected Format Files", ""]
        for u in unexpected[:200]:
            lines.append(f"- {u}")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
    return "\n".join(lines)


def _label_qa_report_md(report: dict[str, Any]) -> str:
    samples = report["suspicious_samples"]
    lines = [
        "# Label Quality Report",
        "",
        f"**Status:** `{report['status'].upper()}`  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Quality score:** {report['label_quality_score']:.4f}  ",
        f"**Reference model used:** {'yes' if report['reference_model_used'] else 'no'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Images checked | {report['num_images_checked']} |",
        f"| Label files checked | {report['num_labels_checked']} |",
        f"| Suspicious samples | {len(samples)} |",
        "",
    ]

    if report["class_distribution"]:
        lines += ["## Class Distribution", "", "| Class | Count |", "|-------|-------|"]
        total = sum(report["class_distribution"].values())
        for cls, count in sorted(report["class_distribution"].items()):
            pct = count / total * 100 if total else 0
            lines.append(f"| {cls} | {count} ({pct:.1f}%) |")
        lines.append("")

    if samples:
        lines += ["## Suspicious Samples", "", "| Image | Split | Issue | Message |",
                   "|-------|-------|-------|---------|"]
        for s in samples[:200]:  # keep the report readable for large datasets
            lines.append(
                f"| {s['image'] or 'N/A'} | {s['split'] or 'N/A'} "
                f"| {s['issue_type']} | {s['message']} |"
            )
        if len(samples) > 200:
            lines.append(f"| ... | ... | ... | {len(samples) - 200} more not shown |")
        lines.append("")

    lines.append(f"**Message:** {report['message']}")
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
