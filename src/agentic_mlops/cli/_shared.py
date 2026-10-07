"""Shared helpers for the Agentic MLOps CLI."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from agentic_mlops.contracts.mlflow_config import MLflowConfig
    from agentic_mlops.integrations.artifact_store import ArtifactStore
    from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase

console = Console()


class OutputFormat(str, enum.Enum):
    text = "text"
    json = "json"


# ── MLflow helpers ─────────────────────────────────────────────────────────────


def _resolve_mlflow(
    mlflow_config_path: str | None,
    enable_mlflow: bool,
) -> tuple[MLflowTrackingClientBase, MLflowConfig]:
    """Build (client, config) pair from CLI flags.

    Priority: explicit --enable-mlflow flag > config file > disabled default.
    """
    from agentic_mlops.contracts.mlflow_config import MLflowConfig  # noqa: PLC0415
    from agentic_mlops.integrations.mlflow_client import (  # noqa: PLC0415
        LocalMLflowTrackingClient,
        NoOpMLflowTrackingClient,
    )

    if mlflow_config_path:
        cfg = MLflowConfig.from_yaml(mlflow_config_path)
    else:
        cfg = MLflowConfig(enabled=False)

    # CLI flag takes precedence over config file
    if enable_mlflow:
        cfg = cfg.model_copy(update={"enabled": True})

    if cfg.enabled:
        try:
            client: MLflowTrackingClientBase = LocalMLflowTrackingClient(cfg.tracking_uri)
        except RuntimeError as exc:
            console.print(f"[red]MLflow error: {exc}[/red]")
            raise typer.Exit(code=1)
    else:
        client = NoOpMLflowTrackingClient()

    return client, cfg


def _make_artifact_store(azure_config_path: str | None) -> ArtifactStore:
    """Return AzureBlobArtifactStore when storage is enabled, else NoOpArtifactStore."""
    from agentic_mlops.integrations.artifact_store import (  # noqa: PLC0415
        AzureBlobArtifactStore,
        NoOpArtifactStore,
    )

    if azure_config_path:
        try:
            from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415

            cfg = AzureMLConfig.from_yaml(azure_config_path)
            if cfg.storage.enabled:
                return AzureBlobArtifactStore(cfg)
        except Exception as exc:
            console.print(f"[yellow]Artifact store init failed (blob disabled): {exc}[/yellow]")
    return NoOpArtifactStore()


# ── Pretty-print helpers ───────────────────────────────────────────────────────


def _print_data_intake_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "passed": "green",
        "needs_human_source_approval": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    console.print(
        f"\n[bold {status_color}]Data Intake: {str(result.status).upper()}[/bold {status_color}]"
    )
    console.print(
        f"Files: {result.num_files}  |  Valid images: {result.valid_images}  |  "
        f"Corrupted: {len(result.corrupted_images)}  |  "
        f"Duplicate groups: {len(result.duplicate_groups)}"
    )
    if not result.pillow_available:
        console.print(
            "[yellow]Pillow not installed — corruption checks are best-effort only "
            "(pip install -e '.[vision]')[/yellow]"
        )

    if result.warnings:
        console.print("\n[bold yellow]Needs Review[/bold yellow]")
        for w in result.warnings:
            console.print(f"  [yellow]WARN[/yellow] {w}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.manifest_path:
        console.print(f"\nManifest saved to: [bold]{result.manifest_path}[/bold]")


def _print_dataset_structuring_result(result) -> None:  # type: ignore[type-arg]
    status_color = "green" if result.success else "red"
    label = "SUCCESS" if result.success else "FAILED"
    console.print(f"\n[bold {status_color}]Dataset Structuring: {label}[/bold {status_color}]")
    console.print(f"Images: {result.num_images}  |  Labelled: {result.num_labels}")

    if result.split_counts:
        table = Table(title="Split Counts", show_header=True)
        table.add_column("Split", style="cyan")
        table.add_column("Count", justify="right")
        for split, count in sorted(result.split_counts.items()):
            table.add_row(split, str(count))
        console.print(table)

    if result.warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")
        for w in result.warnings:
            console.print(f"  [yellow]WARN[/yellow] {w}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.data_yaml_path:
        console.print(f"\ndata.yaml: [bold]{result.data_yaml_path}[/bold]")
    if result.split_report_path:
        console.print(f"Split report: [bold]{result.split_report_path}[/bold]")


def _print_annotation_result(result) -> None:  # type: ignore[type-arg]
    status_color = "green" if result.success else "red"
    label = "SUCCESS" if result.success else "FAILED"
    console.print(f"\n[bold {status_color}]Pseudo-Label: {label}[/bold {status_color}]")
    console.print(
        f"Processed: {result.num_images_processed}  |  "
        f"Skipped (already labeled): {result.num_images_skipped_existing}"
    )

    table = Table(title="Confidence Routing", show_header=True)
    table.add_column("Bucket", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("high (auto-candidate)", str(result.high_confidence_count))
    table.add_row("medium (human review)", str(result.medium_confidence_count))
    table.add_row("low (hard sample)", str(result.low_confidence_count))
    console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.pseudo_labels_path:
        console.print(f"\nPseudo-labels: [bold]{result.pseudo_labels_path}[/bold]")
    if result.review_queue_path:
        console.print(f"Review queue: [bold]{result.review_queue_path}[/bold]")


def _print_validation_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "passed": "green",
        "warning": "yellow",
        "failed": "red",
    }.get(result.status, "white")

    console.print(
        f"\n[bold {status_color}]Dataset Validation: {result.status.upper()}[/bold {status_color}]"
    )
    console.print(f"Images: {result.num_images}  |  Label files: {result.num_labels}")

    if result.blocking_issues:
        console.print("\n[bold red]Blocking Issues[/bold red]")
        for issue in result.blocking_issues:
            console.print(f"  [red]FAIL[/red] {issue}")

    if result.warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")
        for warn in result.warnings:
            console.print(f"  [yellow]WARN[/yellow] {warn}")

    if result.class_distribution:
        table = Table(title="Class Distribution", show_header=True)
        table.add_column("Class", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("%", justify="right")
        total = sum(result.class_distribution.values())
        for cls, count in sorted(result.class_distribution.items()):
            pct = count / total * 100 if total else 0
            table.add_row(cls, str(count), f"{pct:.1f}%")
        console.print(table)

    if result.report_path:
        console.print(f"\nReport saved to: [bold]{result.report_path}[/bold]")


def _print_label_qa_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "passed": "green",
        "review_required": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    console.print(
        f"\n[bold {status_color}]Label QA: {str(result.status).upper()}[/bold {status_color}]"
    )
    console.print(
        f"Images: {result.num_images_checked}  |  Labels: {result.num_labels_checked}  |  "
        f"Quality score: {result.label_quality_score:.4f}"
    )

    if result.suspicious_samples:
        table = Table(title="Suspicious Samples", show_header=True)
        table.add_column("Image", style="cyan")
        table.add_column("Split")
        table.add_column("Issue")
        for s in result.suspicious_samples[:50]:
            table.add_row(s.image or "N/A", s.split or "N/A", str(s.issue_type))
        console.print(table)
        if len(result.suspicious_samples) > 50:
            console.print(f"  ... {len(result.suspicious_samples) - 50} more not shown")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.report_path:
        console.print(f"\nReport saved to: [bold]{result.report_path}[/bold]")


def _print_dataset_versioning_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "registered": "green",
        "deduplicated": "yellow",
        "blocked": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Dataset Versioning: {label}[/bold {status_color}]")
    if result.dataset_name:
        console.print(f"Dataset name : {result.dataset_name}")
    if result.version is not None:
        console.print(f"Version      : {result.version}")
    if result.hash:
        console.print(f"Hash         : {result.hash[:16]}...")
    if result.dataset_version_path:
        console.print(f"Path         : [bold]{result.dataset_version_path}[/bold]")
    if result.block_reason:
        console.print(f"\n[yellow]Blocked: {result.block_reason}[/yellow]")
    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")


def _print_model_decision_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "promote": "green",
        "reject": "red",
        "retrain": "yellow",
        "need_more_data": "yellow",
        "need_label_review": "yellow",
    }.get(str(result.decision), "white")

    label = str(result.decision).upper() if result.decision else "N/A"
    console.print(f"\n[bold {status_color}]Model Decision: {label}[/bold {status_color}]")
    console.print(f"Approval gate: {result.required_approval_gate}")

    if result.metrics:
        m = result.metrics
        table = Table(title="Candidate Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("mAP@0.5", f"{m.map50:.4f}")
        table.add_row("mAP@0.5:0.95", f"{m.map50_95:.4f}")
        table.add_row("Precision", f"{m.precision:.4f}")
        table.add_row("Recall", f"{m.recall:.4f}")
        console.print(table)

    if result.map50_improvement is not None:
        console.print(f"mAP50 vs baseline: {result.map50_improvement:+.4f}")

    if result.reasons:
        console.print("\n[bold]Reasons[/bold]")
        for r in result.reasons:
            console.print(f"  - {r}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.decision_report_path:
        console.print(f"\nDecision report: [bold]{result.decision_report_path}[/bold]")


def _print_deployment_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "deployed_to_staging": "green",
        "deployed_to_production": "green",
        "blocked": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Deployment: {label}[/bold {status_color}]")
    if result.endpoint_name:
        console.print(f"Endpoint : {result.endpoint_name}")
    if result.release is not None:
        console.print(f"Release  : {result.release}")
    if result.exported_model_path:
        console.print(f"Exported : [bold]{result.exported_model_path}[/bold]")
    if result.scoring_uri:
        console.print(f"Scoring URI: [bold]{result.scoring_uri}[/bold]")
        console.print(f"Azure deployment: {result.azure_deployment_name}")
    if result.canary_percentage != 100:
        console.print(f"Canary   : {result.canary_percentage}% traffic")

    if result.smoke_test_results:
        console.print("\n[bold]Smoke Tests[/bold]")
        for chk in result.smoke_test_results:
            console.print(f"  - {chk}")

    if result.block_reason:
        console.print(f"\n[yellow]Blocked: {result.block_reason}[/yellow]")
    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.deployment_report_path:
        console.print(f"\nDeployment report: [bold]{result.deployment_report_path}[/bold]")


def _print_monitoring_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "completed": "green",
        "alerts_triggered": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Monitoring: {label}[/bold {status_color}]")
    if result.endpoint_name:
        console.print(f"Endpoint : {result.endpoint_name}")
    console.print(f"Predictions in window: {result.total_predictions}")
    console.print(f"Recommended action   : {result.recommended_action}")
    console.print(f"Requires human review : {'yes' if result.requires_human_review else 'no'}")

    if result.metrics:
        table = Table(title="Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in result.metrics.items():
            table.add_row(k, f"{v:.4f}")
        console.print(table)

    if result.triggered_alerts:
        console.print("\n[bold yellow]Triggered Alerts[/bold yellow]")
        for alert in result.triggered_alerts:
            console.print(f"  - {alert}")

    if result.hard_samples:
        console.print(f"\nHard samples mined: {len(result.hard_samples)}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.monitoring_report_path:
        console.print(f"\nMonitoring report: [bold]{result.monitoring_report_path}[/bold]")


def _print_evaluation_result(result) -> None:  # type: ignore[type-arg]
    rec = result.recommendation or "N/A"
    rec_color = {
        "promote_candidate": "green",
        "retrain": "red",
        "need_label_review": "yellow",
    }.get(rec, "white")

    status_label = "PASS" if result.success else "FAIL"
    console.print(
        f"\n[bold {rec_color}]Evaluation: {status_label} — {rec.upper()}[/bold {rec_color}]"
    )

    if result.metrics:
        m = result.metrics
        table = Table(title="Global Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("mAP@0.5", f"{m.map50:.4f}")
        table.add_row("mAP@0.5:0.95", f"{m.map50_95:.4f}")
        table.add_row("Precision", f"{m.precision:.4f}")
        table.add_row("Recall", f"{m.recall:.4f}")
        console.print(table)

    if result.passed_checks:
        console.print("\n[bold green]Passed Checks[/bold green]")
        for chk in result.passed_checks:
            console.print(f"  [green]PASS[/green] {chk}")

    if result.failed_checks:
        console.print("\n[bold red]Failed Checks[/bold red]")
        for chk in result.failed_checks:
            console.print(f"  [red]FAIL[/red] {chk}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.evaluation_report_path:
        console.print(f"\nReport saved to: [bold]{result.evaluation_report_path}[/bold]")


def _print_training_result(result) -> None:  # type: ignore[type-arg]
    status_color = "green" if result.success else "red"
    label = "COMPLETED" if result.success else "FAILED/BLOCKED"
    console.print(f"\n[bold {status_color}]Training: {label}[/bold {status_color}]")
    console.print(f"Mode: {result.mode}  |  Job status: {result.job_status}")

    if result.job_id:
        console.print(f"Job ID:          {result.job_id}")
    if result.mlflow_run_id:
        console.print(f"MLflow Run ID:   {result.mlflow_run_id}")
    if result.training_plan_path:
        console.print(f"Training plan:   [bold]{result.training_plan_path}[/bold]")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.report_path:
        console.print(f"\nReport saved to: [bold]{result.report_path}[/bold]")


def _print_approval_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "approved": "green",
        "rejected": "yellow",
        "needs_retraining": "yellow",
        "needs_more_data": "yellow",
        "needs_label_review": "yellow",
        "cancelled": "red",
        "pending": "white",
    }.get(str(result.status), "white")

    label = str(result.status).upper().replace("_", " ")
    console.print(f"\n[bold {status_color}]Approval: {label}[/bold {status_color}]")
    console.print(f"Action:   {result.action or 'N/A'}")
    console.print(f"Approver: {result.approver or 'N/A'}")

    if result.comment:
        console.print(f"Comment:  {result.comment}")

    if result.approval_request:
        req = result.approval_request
        console.print(f"Model:    {req.candidate_model}")
        console.print(f"Dataset:  {req.dataset_version_or_path}")
        if req.next_actions:
            console.print("\n[bold]Next Actions[/bold]")
            for a in req.next_actions:
                console.print(f"  - {a}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.generated_artifacts:
        artifacts_parent = Path(result.generated_artifacts[0]).parent
        console.print(f"\nArtifacts saved to: [bold]{artifacts_parent}[/bold]")


def _print_training_approval_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "approved": "green",
        "rejected": "yellow",
        "cancelled": "red",
        "pending": "white",
    }.get(str(result.status), "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Training Approval (H4): {label}[/bold {status_color}]")
    console.print(f"Action:   {result.action or 'N/A'}")
    console.print(f"Approver: {result.approver or 'N/A'}")

    if result.comment:
        console.print(f"Comment:  {result.comment}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.generated_artifacts:
        artifacts_parent = Path(result.generated_artifacts[0]).parent
        console.print(f"\nArtifacts saved to: [bold]{artifacts_parent}[/bold]")


def _print_registry_result(result) -> None:  # type: ignore[type-arg]
    from agentic_mlops.contracts.model_registry import RegistrationStatus  # noqa: PLC0415

    status_color = {
        RegistrationStatus.REGISTERED: "green",
        RegistrationStatus.SKIPPED: "yellow",
        RegistrationStatus.BLOCKED: "yellow",
        RegistrationStatus.FAILED: "red",
    }.get(result.status, "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Model Registry: {label}[/bold {status_color}]")

    if result.model_name:
        console.print(f"Model name : {result.model_name}")
    if result.version is not None:
        console.print(f"Version    : {result.version}")
    if result.registry_path:
        console.print(f"Path       : [bold]{result.registry_path}[/bold]")
    if result.block_reason:
        console.print(f"\n[yellow]Blocked: {result.block_reason}[/yellow]")
    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")


def _print_orchestrator_result(result) -> None:  # type: ignore[type-arg]
    status_color = {
        "completed": "green",
        "pending_approval": "yellow",
        "blocked": "yellow",
        "failed": "red",
    }.get(str(result.status), "white")

    label = str(result.status).upper()
    console.print(f"\n[bold {status_color}]Orchestrator: {label}[/bold {status_color}]")
    console.print(f"Workflow ID   : {result.workflow_id}")
    console.print(f"Current state : {result.current_state}")

    table = Table(title="Steps", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Status")
    table.add_column("OK?", justify="center")
    for step in result.steps:
        ok_str = "[green]yes[/green]" if step.success else "[red]no[/red]"
        table.add_row(step.step, step.status, ok_str)
    console.print(table)

    if result.pending_approval_id:
        console.print(
            f"\n[yellow]Awaiting human approval — pending_approval_id: "
            f"{result.pending_approval_id}[/yellow]"
        )
        console.print(
            "Re-run with --resume once a decision has been made "
            "(set approval_action in the config, or interactive_approval: true)."
        )

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.state_path:
        console.print(f"\nState file : [bold]{result.state_path}[/bold]")
        console.print(f"Audit log  : [bold]{result.audit_log_path}[/bold]")

    if result.mlflow_run_id:
        console.print(f"\nMLflow experiment : [bold]{result.mlflow_experiment_name}[/bold]")
        console.print(f"MLflow run ID     : [bold]{result.mlflow_run_id}[/bold]")
        console.print(f"MLflow tracking   : [bold]{result.mlflow_tracking_uri}[/bold]")


def _print_mvp_summary(result) -> None:  # type: ignore[type-arg]
    status_color = "green" if result.success else "red"
    label = str(result.workflow_status).upper()
    console.print(f"\n[bold {status_color}]MVP Workflow: {label}[/bold {status_color}]")

    table = Table(title="Workflow Steps", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Status")
    table.add_column("OK?", justify="center")
    for step in result.steps:
        ok_str = "[green]yes[/green]" if step.success else "[red]no[/red]"
        table.add_row(step.step, step.status, ok_str)
    console.print(table)

    if result.mlflow_run_id:
        console.print(f"\nMLflow experiment : [bold]{result.mlflow_experiment_name}[/bold]")
        console.print(f"MLflow run ID     : [bold]{result.mlflow_run_id}[/bold]")
        console.print(f"MLflow tracking   : [bold]{result.mlflow_tracking_uri}[/bold]")

    if result.registration_status:
        reg_color = "green" if result.registration_status == "registered" else "yellow"
        console.print(
            f"\nModel Registry: [{reg_color}]{result.registration_status.upper()}[/{reg_color}]"
        )
        if result.registered_model_name:
            console.print(f"  Name   : {result.registered_model_name}")
        if result.registered_model_version is not None:
            console.print(f"  Version: {result.registered_model_version}")
        if result.registered_model_path:
            console.print(f"  Path   : [bold]{result.registered_model_path}[/bold]")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]FAIL[/red] {err}")

    if result.workflow_summary_path:
        summary_dir = Path(result.workflow_summary_path).parent
        console.print(f"\nWorkflow artifacts: [bold]{summary_dir}[/bold]")
