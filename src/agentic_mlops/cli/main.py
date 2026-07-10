"""CLI entry-point for the Agentic MLOps workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from agentic_mlops.observability.logging import configure_logging

if TYPE_CHECKING:
    from agentic_mlops.contracts.mlflow_config import MLflowConfig
    from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase

app = typer.Typer(
    name="agentic-mlops",
    help="Agentic MLOps workflow for YOLO object detection.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _setup(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-formatted log lines"),
) -> None:
    configure_logging(level=log_level, json_format=json_logs)


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


@app.command("validate-dataset")
def validate_dataset(
    dataset_path: str = typer.Argument(..., help="Path to the YOLO dataset root directory"),
    data_yaml: str = typer.Option(None, "--data-yaml", help="Override path to data.yaml"),
    output_dir: str = typer.Option(
        None, "--output-dir", help="Where to save reports (default: <dataset_path>/validation_out)"
    ),
    fail_on_warnings: bool = typer.Option(
        False, "--fail-on-warnings", help="Exit code 1 if warnings are found"
    ),
) -> None:
    """Validate a YOLO dataset and write a quality report."""
    from agentic_mlops.agents.dataset_validation import DatasetValidationAgent  # noqa: PLC0415
    from agentic_mlops.contracts.datasets import DatasetValidationInput  # noqa: PLC0415

    artifacts_dir = Path(output_dir) if output_dir else Path(dataset_path) / "validation_out"

    agent = DatasetValidationAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DatasetValidationInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml,
            fail_on_warnings=fail_on_warnings,
        )
    )

    _print_validation_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("data-intake")
def data_intake(
    raw_data_path: str = typer.Argument(..., help="Path to the raw image directory"),
    dataset_name: str = typer.Option(..., "--dataset-name", help="Name for this raw dataset"),
    source: str = typer.Option(
        None,
        "--source",
        help="Data provenance/source (e.g. 'internal_camera_batch') — omitting it forces "
        "needs_human_source_approval",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the manifest (default: <raw_data_path>/intake_out)",
    ),
    expected_formats: str = typer.Option(
        "jpg,jpeg,png", "--expected-formats", help="Comma-separated list of accepted extensions"
    ),
    min_files: int = typer.Option(1, "--min-files", help="Minimum file count required to pass"),
    corrupted_ratio_threshold: float = typer.Option(
        0.05, "--corrupted-ratio-threshold", help="Corrupted-file ratio that fails intake"
    ),
    duplicate_ratio_threshold: float = typer.Option(
        0.20, "--duplicate-ratio-threshold", help="Duplicate-file ratio that requires human review"
    ),
) -> None:
    """Scan a raw image directory and write a dataset_manifest."""
    from agentic_mlops.agents.data_intake import DataIntakeAgent  # noqa: PLC0415
    from agentic_mlops.contracts.data_intake import DataIntakeInput  # noqa: PLC0415

    artifacts_dir = Path(output_dir) if output_dir else Path(raw_data_path) / "intake_out"

    agent = DataIntakeAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DataIntakeInput(
            raw_data_path=raw_data_path,
            dataset_name=dataset_name,
            source=source,
            expected_formats=[f.strip() for f in expected_formats.split(",") if f.strip()],
            min_files=min_files,
            corrupted_ratio_threshold=corrupted_ratio_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
        )
    )

    _print_data_intake_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("label-qa")
def label_qa(
    dataset_path: str = typer.Argument(..., help="Path to the YOLO dataset root directory"),
    data_yaml: str = typer.Option(None, "--data-yaml", help="Override path to data.yaml"),
    output_dir: str = typer.Option(
        None, "--output-dir", help="Where to save reports (default: <dataset_path>/label_qa_out)"
    ),
    reference_model: str = typer.Option(
        None,
        "--reference-model",
        help="Path to a reference YOLO model (.pt) — enables the disagreement check",
    ),
    reference_model_confidence: float = typer.Option(
        0.25, "--reference-model-confidence", help="Confidence threshold for the reference model"
    ),
    review_required_threshold: int = typer.Option(
        5,
        "--review-required-threshold",
        help="Suspicious sample count that flips status from passed to review_required",
    ),
) -> None:
    """Check label quality on a YOLO dataset and write a suspicious-samples report."""
    from agentic_mlops.agents.label_qa import LabelQAAgent  # noqa: PLC0415
    from agentic_mlops.contracts.label_qa import LabelQAInput  # noqa: PLC0415

    artifacts_dir = Path(output_dir) if output_dir else Path(dataset_path) / "label_qa_out"

    agent = LabelQAAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        LabelQAInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml,
            reference_model_path=reference_model,
            reference_model_confidence=reference_model_confidence,
            review_required_threshold=review_required_threshold,
        )
    )

    _print_label_qa_result(result)

    if not result.success or result.status == "review_required":
        raise typer.Exit(code=1)


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


@app.command("train")
def train(
    dataset_path: str = typer.Option(..., "--dataset-path", help="YOLO dataset root directory"),
    data_yaml: str = typer.Option(..., "--data-yaml", help="Path to data.yaml"),
    training_config: str = typer.Option(
        ..., "--training-config", help="Path to training config YAML"
    ),
    output_dir: str = typer.Option(
        None, "--output-dir", help="Where to save artifacts (default: <dataset_path>/training_out)"
    ),
    runner: str = typer.Option(
        None,
        "--runner",
        help="Override runner: fake (dry-run) | local-yolo (Ultralytics) | azure-ml (Azure ML)",
    ),
    azure_config: str = typer.Option(
        None, "--azure-config", help="Path to Azure ML config YAML (required for --runner azure-ml)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Alias for --runner fake: write plan, skip training"
    ),
    dataset_validation_status: str = typer.Option(
        None,
        "--dataset-validation-status",
        help="Pass the upstream validation status (passed|warning|failed). "
             "Training is blocked when 'failed'.",
    ),
    mlflow_config: str = typer.Option(
        None, "--mlflow-config", help="Path to MLflow config YAML"
    ),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--disable-mlflow",
        help="Enable MLflow tracking (default: disabled)",
    ),
) -> None:
    """Build a training plan (--runner fake) or run real YOLO training (local-yolo / azure-ml)."""
    from agentic_mlops.agents.training import TrainingAgent  # noqa: PLC0415
    from agentic_mlops.contracts.training import (  # noqa: PLC0415
        TrainingConfig,
        TrainingInput,
        TrainingMode,
    )

    cfg = TrainingConfig.from_yaml(training_config)

    if dry_run or runner == "fake":
        cfg.mode = TrainingMode.LOCAL_DRY_RUN
    elif runner == "local-yolo":
        cfg.mode = TrainingMode.LOCAL_TRAIN
    elif runner == "azure-ml":
        if not azure_config:
            console.print("[red]--azure-config is required when --runner azure-ml[/red]")
            raise typer.Exit(code=1)
        cfg.mode = TrainingMode.AZURE_TRAIN
    elif runner is not None:
        console.print(
            f"[red]Unknown runner '{runner}'. Valid values: fake, local-yolo, azure-ml[/red]"
        )
        raise typer.Exit(code=1)

    artifacts_dir = Path(output_dir) if output_dir else Path(dataset_path) / "training_out"

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)

    run_id: str | None = None
    if mlflow_cfg.enabled:
        run_name = f"{mlflow_cfg.run_name_prefix}-train"
        run_id = mlflow_client.start_run(mlflow_cfg.experiment_name, run_name)

    azure_runner = None
    if cfg.mode == TrainingMode.AZURE_TRAIN:
        from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415
        from agentic_mlops.tools.training_runner import AzureMLTrainingRunner  # noqa: PLC0415

        azure_runner = AzureMLTrainingRunner(AzureMLConfig.from_yaml(azure_config))

    agent = TrainingAgent(
        artifacts_dir=artifacts_dir,
        mlflow_client=mlflow_client if run_id else None,
        mlflow_run_id=run_id,
        azure_runner=azure_runner,
    )
    result = agent.run(
        TrainingInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml,
            training_config=cfg,
            dataset_validation_status=dataset_validation_status,  # type: ignore[arg-type]
        )
    )

    if run_id:
        final_status = "FINISHED" if result.success else "FAILED"
        mlflow_client.end_run(run_id, status=final_status)

    _print_training_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    dataset_path: str = typer.Option(..., "--dataset-path", help="YOLO dataset root directory"),
    data_yaml: str = typer.Option(..., "--data-yaml", help="Path to data.yaml"),
    weights_path: str = typer.Option(
        None, "--weights-path", help="Path to model weights (.pt)"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save artifacts (default: <dataset_path>/evaluation_out)",
    ),
    runner: str = typer.Option(
        None,
        "--runner",
        help="Override runner: fake (dry-run) | local-yolo (Ultralytics) | azure-ml (Azure ML)",
    ),
    azure_config: str = typer.Option(
        None, "--azure-config", help="Path to Azure ML config YAML (required for --runner azure-ml)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Alias for --runner fake: use fake metrics"
    ),
    training_status: str = typer.Option(
        None,
        "--training-status",
        help=(
            "Upstream training status (completed|failed|cancelled)."
            " Blocked when failed/cancelled."
        ),
    ),
    promotion_policy: str = typer.Option(
        None, "--promotion-policy", help="Path to promotion_policy.yaml"
    ),
    training_output: str = typer.Option(
        None,
        "--training-output",
        help="Path to training_output.json; extracts best_weights_path automatically",
    ),
    evaluation_config: str = typer.Option(
        None, "--evaluation-config", help="Path to evaluation config YAML"
    ),
    mlflow_config: str = typer.Option(
        None, "--mlflow-config", help="Path to MLflow config YAML"
    ),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--disable-mlflow",
        help="Enable MLflow tracking (default: disabled)",
    ),
) -> None:
    """Evaluate a YOLO model and apply the promotion policy."""
    import json as _json  # noqa: PLC0415

    from agentic_mlops.agents.evaluation import EvaluationAgent  # noqa: PLC0415
    from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMode  # noqa: PLC0415

    # Resolve weights_path from --training-output if not provided directly
    resolved_weights = weights_path
    if training_output and not resolved_weights:
        data = _json.loads(Path(training_output).read_text(encoding="utf-8"))
        resolved_weights = data.get("best_weights_path") or "dry_run"

    if not resolved_weights:
        resolved_weights = "dry_run"

    if dry_run or runner == "fake":
        mode = EvaluationMode.LOCAL_DRY_RUN
    elif runner == "local-yolo":
        mode = EvaluationMode.LOCAL_EVAL
    elif runner == "azure-ml":
        if not azure_config:
            console.print("[red]--azure-config is required when --runner azure-ml[/red]")
            raise typer.Exit(code=1)
        mode = EvaluationMode.AZURE_EVAL
    elif runner is not None:
        console.print(
            f"[red]Unknown runner '{runner}'. Valid values: fake, local-yolo, azure-ml[/red]"
        )
        raise typer.Exit(code=1)
    else:
        mode = EvaluationMode.LOCAL_DRY_RUN

    artifacts_dir = Path(output_dir) if output_dir else Path(dataset_path) / "evaluation_out"

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)

    run_id: str | None = None
    if mlflow_cfg.enabled:
        run_name = f"{mlflow_cfg.run_name_prefix}-evaluate"
        run_id = mlflow_client.start_run(mlflow_cfg.experiment_name, run_name)

    azure_runner = None
    if mode == EvaluationMode.AZURE_EVAL:
        from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415
        from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner  # noqa: PLC0415

        azure_runner = AzureMLEvaluationRunner(AzureMLConfig.from_yaml(azure_config))

    agent = EvaluationAgent(
        artifacts_dir=artifacts_dir,
        mlflow_client=mlflow_client if run_id else None,
        mlflow_run_id=run_id,
        azure_runner=azure_runner,
    )
    result = agent.run(
        EvaluationInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml,
            weights_path=resolved_weights,
            mode=mode,
            training_status=training_status,  # type: ignore[arg-type]
            promotion_policy_path=promotion_policy,
            evaluation_config_path=evaluation_config,
        )
    )

    if run_id:
        final_status = "FINISHED" if result.success else "FAILED"
        mlflow_client.end_run(run_id, status=final_status)

    _print_evaluation_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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


@app.command("approve")
def approve(
    evaluation_output: str = typer.Option(
        ..., "--evaluation-output", help="Path to evaluation_report.json"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save approval artifacts (default: <eval_dir>/../approval_out)",
    ),
    approver: str = typer.Option(None, "--approver", help="Name of the approver"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Prompt for decision interactively"
    ),
    action: str = typer.Option(
        None,
        "--action",
        help=(
            "Action in non-interactive mode: approve_model | reject_model |"
            " request_retraining | request_more_data | request_label_review | cancel"
        ),
    ),
    comment: str = typer.Option(None, "--comment", help="Optional human comment"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force approve a non-promoted candidate in non-interactive mode.",
    ),
    candidate_model: str = typer.Option(
        "unknown", "--candidate-model", help="Model name or ID"
    ),
    dataset_version: str = typer.Option(
        "unknown", "--dataset-version", help="Dataset version or path"
    ),
) -> None:
    """Review evaluation output and record a human approval decision."""
    from agentic_mlops.agents.human_approval import HumanApprovalAgent  # noqa: PLC0415
    from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalInput  # noqa: PLC0415

    parsed_action: ApprovalAction | None = None
    if action:
        try:
            parsed_action = ApprovalAction(action)
        except ValueError:
            valid = ", ".join(a.value for a in ApprovalAction)
            console.print(f"[red]Invalid action '{action}'. Valid values: {valid}[/red]")
            raise typer.Exit(code=1)

    out_dir = output_dir or str(Path(evaluation_output).parent.parent / "approval_out")

    agent = HumanApprovalAgent(artifacts_dir=Path(out_dir))
    result = agent.run(
        ApprovalInput(
            evaluation_output_path=evaluation_output,
            approver=approver,
            output_dir=out_dir,
            interactive=interactive,
            action=parsed_action,
            comment=comment,
            force=force,
            candidate_model=candidate_model,
            dataset_version_or_path=dataset_version,
        )
    )

    _print_approval_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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


@app.command("register-model")
def register_model(
    model_name: str = typer.Option(..., "--model-name", help="Registry model name"),
    training_output: str = typer.Option(
        ..., "--training-output", help="Path to training_output.json"
    ),
    evaluation_output: str = typer.Option(
        ..., "--evaluation-output", help="Path to evaluation_output.json"
    ),
    approval_decision: str = typer.Option(
        ..., "--approval-decision", help="Path to approval_decision.json"
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry", "--registry-dir", help="Root of local model registry"
    ),
    backend: str = typer.Option(
        "local", "--backend", help="Registry backend: local | mlflow | azure_ml"
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help="Path to Azure ML config YAML (required for --backend azure_ml)",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save registration artifacts (default: <registry_dir>/registration_out)",
    ),
) -> None:
    """Register an approved YOLO model into the local model registry."""
    from agentic_mlops.agents.model_registry import ModelRegistryAgent  # noqa: PLC0415
    from agentic_mlops.contracts.model_registry import (  # noqa: PLC0415
        ModelRegistrationInput,
        RegistryBackend,
    )

    try:
        parsed_backend = RegistryBackend(backend)
    except ValueError:
        valid = ", ".join(b.value for b in RegistryBackend)
        console.print(f"[red]Invalid backend '{backend}'. Valid values: {valid}[/red]")
        raise typer.Exit(code=1)

    registry_client = None
    if parsed_backend == RegistryBackend.AZURE_ML:
        if not azure_config:
            console.print("[red]--azure-config is required when --backend azure_ml[/red]")
            raise typer.Exit(code=1)
        from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415
        from agentic_mlops.integrations.model_registry import (  # noqa: PLC0415
            AzureMLModelRegistryClient,
        )

        registry_client = AzureMLModelRegistryClient(AzureMLConfig.from_yaml(azure_config))

    artifacts_dir = (
        Path(output_dir)
        if output_dir
        else Path(registry_dir) / "registration_out"
    )

    agent = ModelRegistryAgent(artifacts_dir=artifacts_dir, registry_client=registry_client)
    result = agent.run(
        ModelRegistrationInput(
            model_name=model_name,
            training_output_path=training_output,
            evaluation_output_path=evaluation_output,
            approval_decision_path=approval_decision,
            registry_dir=registry_dir,
            backend=parsed_backend,
        )
    )

    _print_registry_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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


@app.command("run-mvp")
def run_mvp(
    dataset_path: str = typer.Option(..., "--dataset-path", help="YOLO dataset root directory"),
    data_yaml: str = typer.Option(..., "--data-yaml", help="Path to data.yaml"),
    training_config: str = typer.Option(
        ..., "--training-config", help="Path to training config YAML"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", help="Root directory for all workflow artifacts"
    ),
    evaluation_config: str = typer.Option(
        None, "--evaluation-config", help="Path to promotion_policy.yaml"
    ),
    approver: str = typer.Option(None, "--approver", help="Name of the approver"),
    approval_action: str = typer.Option(
        None,
        "--approval-action",
        help=(
            "Approval action in non-interactive mode: "
            "approve_model | reject_model | request_retraining | "
            "request_more_data | request_label_review | cancel"
        ),
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Use dry-run mode"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive approval prompt"
    ),
    force_approve: bool = typer.Option(
        False, "--force-approve", help="Force approve non-promoted candidate"
    ),
    fail_on_warnings: bool = typer.Option(
        False, "--fail-on-warnings", help="Treat dataset warnings as failures"
    ),
    training_runner: str = typer.Option(
        None, "--training-runner", help="Override training runner: fake | local-yolo | azure-ml"
    ),
    evaluation_runner: str = typer.Option(
        None,
        "--evaluation-runner",
        help="Override evaluation runner: fake | local-yolo | azure-ml",
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help=(
            "Path to Azure ML config YAML — required when --training-runner azure-ml, "
            "--evaluation-runner azure-ml, or --registry-backend azure_ml"
        ),
    ),
    mlflow_config: str = typer.Option(
        None, "--mlflow-config", help="Path to MLflow config YAML"
    ),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--disable-mlflow",
        help="Enable MLflow tracking (default: disabled)",
    ),
    register_approved_model: bool = typer.Option(
        False,
        "--register-approved-model/--no-register-approved-model",
        help="Register the model after approval (local registry by default)",
    ),
    model_name: str = typer.Option(
        "yolo-model", "--model-name", help="Model name for the registry"
    ),
    registry_backend: str = typer.Option(
        "local", "--registry-backend", help="Registry backend: local | mlflow | azure_ml"
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry",
        "--registry-dir",
        help="Root directory for the local model registry",
    ),
) -> None:
    """Run the end-to-end MVP workflow: validate -> train -> evaluate -> approve -> register."""
    from agentic_mlops.contracts.approvals import ApprovalAction  # noqa: PLC0415
    from agentic_mlops.contracts.model_registry import RegistryBackend  # noqa: PLC0415
    from agentic_mlops.contracts.workflows import MVPWorkflowInput  # noqa: PLC0415
    from agentic_mlops.workflows.mvp_workflow import MVPWorkflow  # noqa: PLC0415

    parsed_action: ApprovalAction | None = None
    if approval_action:
        try:
            parsed_action = ApprovalAction(approval_action)
        except ValueError:
            valid = ", ".join(a.value for a in ApprovalAction)
            console.print(
                f"[red]Invalid approval-action '{approval_action}'. Valid: {valid}[/red]"
            )
            raise typer.Exit(code=1)

    try:
        parsed_backend = RegistryBackend(registry_backend)
    except ValueError:
        valid_b = ", ".join(b.value for b in RegistryBackend)
        console.print(
            f"[red]Invalid registry-backend '{registry_backend}'. Valid: {valid_b}[/red]"
        )
        raise typer.Exit(code=1)

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)

    workflow = MVPWorkflow(
        mlflow_client=mlflow_client if mlflow_cfg.enabled else None,
        mlflow_config=mlflow_cfg if mlflow_cfg.enabled else None,
    )
    result = workflow.run(
        MVPWorkflowInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml,
            training_config_path=training_config,
            evaluation_config_path=evaluation_config,
            output_dir=output_dir,
            approver=approver,
            approval_action=parsed_action,
            dry_run=dry_run,
            interactive_approval=interactive,
            force_approve=force_approve,
            fail_on_warnings=fail_on_warnings,
            training_runner=training_runner,
            evaluation_runner=evaluation_runner,
            azure_config_path=azure_config,
            register_approved_model=register_approved_model,
            model_name=model_name,
            registry_backend=parsed_backend,
            registry_dir=registry_dir,
        )
    )

    _print_mvp_summary(result)

    if not result.success:
        raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
