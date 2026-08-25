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
    from agentic_mlops.integrations.artifact_store import ArtifactStore
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


@app.command("structure-dataset")
def structure_dataset(
    raw_data_path: str = typer.Argument(..., help="Path to raw images (+ labels)"),
    output_dataset_path: str = typer.Option(
        ..., "--output-dataset-path", help="Where to write the structured YOLO dataset"
    ),
    classes: str = typer.Option(
        ..., "--classes", help="Comma-separated ordered class list (defines data.yaml names)"
    ),
    label_format: str = typer.Option(
        "yolo", "--label-format", help="Source label format: yolo | coco | voc"
    ),
    coco_annotations: str = typer.Option(
        None,
        "--coco-annotations",
        help="Path to a COCO annotations JSON file (required when --label-format coco)",
    ),
    voc_annotations_dir: str = typer.Option(
        None,
        "--voc-annotations-dir",
        help=(
            "Directory containing Pascal VOC XML files (optional when --label-format voc; "
            "defaults to <raw_data_path>/Annotations/ or <raw_data_path>/)"
        ),
    ),
    split_strategy: str = typer.Option(
        "random", "--split-strategy", help="Split strategy: random | grouped_by_source"
    ),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio"),
    group_by_regex: str = typer.Option(
        None,
        "--group-by-regex",
        help="Regex with one capturing group to extract a source/video id from filenames "
        "(used only with --split-strategy grouped_by_source)",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for the split"),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the split report (default: <output-dataset-path>/structuring_out)",
    ),
) -> None:
    """Structure raw images (+ YOLO/COCO labels) into a split YOLO dataset."""
    from agentic_mlops.agents.dataset_structuring import DatasetStructuringAgent  # noqa: PLC0415
    from agentic_mlops.contracts.dataset_structuring import (  # noqa: PLC0415
        DatasetStructuringInput,
        LabelFormat,
        SplitStrategy,
    )

    try:
        parsed_format = LabelFormat(label_format)
    except ValueError:
        valid = ", ".join(f.value for f in LabelFormat)
        console.print(f"[red]Invalid label-format '{label_format}'. Valid values: {valid}[/red]")
        raise typer.Exit(code=1)

    try:
        parsed_strategy = SplitStrategy(split_strategy)
    except ValueError:
        valid_s = ", ".join(s.value for s in SplitStrategy)
        console.print(
            f"[red]Invalid split-strategy '{split_strategy}'. Valid values: {valid_s}[/red]"
        )
        raise typer.Exit(code=1)

    artifacts_dir = (
        Path(output_dir) if output_dir else Path(output_dataset_path) / "structuring_out"
    )

    agent = DatasetStructuringAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DatasetStructuringInput(
            raw_data_path=raw_data_path,
            output_dataset_path=output_dataset_path,
            classes=[c.strip() for c in classes.split(",") if c.strip()],
            label_format=parsed_format,
            coco_annotations_path=coco_annotations,
            voc_annotations_dir=voc_annotations_dir,
            split_strategy=parsed_strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_by_regex=group_by_regex,
            seed=seed,
        )
    )

    _print_dataset_structuring_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("pseudo-label")
def pseudo_label(
    images_path: str = typer.Argument(..., help="Path to unlabeled/partially-labeled images"),
    model_path: str = typer.Option(
        ..., "--model-path", help="Path to an approved YOLO model (.pt)"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save pseudo-labels/report (default: <images_path>/pseudo_label_out)",
    ),
    auto_candidate_threshold: float = typer.Option(
        0.90,
        "--auto-candidate-threshold",
        help="Min-detection confidence for the high-confidence bucket",
    ),
    human_review_threshold: float = typer.Option(
        0.50,
        "--human-review-threshold",
        help="Min-detection confidence for the medium bucket (below -> low)",
    ),
    imgsz: int = typer.Option(640, "--imgsz", help="Inference image size"),
    device: str = typer.Option("cpu", "--device", help="Inference device (cpu, 0, 0,1, ...)"),
    existing_labels_path: str = typer.Option(
        None,
        "--existing-labels-path",
        help="Directory of existing human labels — images already labeled there are skipped",
    ),
) -> None:
    """Pre-label images with a YOLO model and route them into confidence buckets."""
    from agentic_mlops.agents.annotation import AnnotationAgent  # noqa: PLC0415
    from agentic_mlops.contracts.annotation import (  # noqa: PLC0415
        AnnotationInput,
        ConfidenceThresholds,
    )

    artifacts_dir = Path(output_dir) if output_dir else Path(images_path) / "pseudo_label_out"

    agent = AnnotationAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        AnnotationInput(
            images_path=images_path,
            model_path=model_path,
            confidence_thresholds=ConfidenceThresholds(
                auto_candidate=auto_candidate_threshold,
                human_review=human_review_threshold,
            ),
            imgsz=imgsz,
            device=device,
            existing_labels_path=existing_labels_path,
        )
    )

    _print_annotation_result(result)

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


@app.command("version-dataset")
def version_dataset(
    dataset_path: str = typer.Argument(..., help="Path to a structured YOLO dataset"),
    dataset_name: str = typer.Option(..., "--dataset-name", help="Registry name for this dataset"),
    registry_dir: str = typer.Option(
        "outputs/dataset_registry", "--registry-dir", help="Root of the local dataset registry"
    ),
    parent_version: int = typer.Option(
        None, "--parent-version", help="Version this one derives from, if any"
    ),
    workflow_id: str = typer.Option(
        None, "--workflow-id", help="Workflow that produced this dataset"
    ),
    validation_report: str = typer.Option(
        None,
        "--validation-report",
        help="Path to dataset_quality_report.json — blocks registration if status='failed'",
    ),
    label_quality_report: str = typer.Option(
        None,
        "--label-quality-report",
        help="Path to label_quality_report.json — blocks registration if status='failed'",
    ),
    approved_by: str = typer.Option(None, "--approved-by", help="Name of the approver"),
    source_batches: str = typer.Option(
        None, "--source-batches", help="Comma-separated list of source batch identifiers"
    ),
    backend: str = typer.Option(
        "local", "--backend", help="Dataset registry backend: local (default) | azure_ml"
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help="Path to Azure ML config YAML — required for --backend azure_ml",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the version report (default: <registry-dir>/version_out)",
    ),
) -> None:
    """Register a structured YOLO dataset as a new version with lineage."""
    from agentic_mlops.agents.dataset_versioning import DatasetVersioningAgent  # noqa: PLC0415
    from agentic_mlops.contracts.dataset_versioning import (  # noqa: PLC0415
        DatasetRegistryBackend,
        DatasetVersioningInput,
    )

    try:
        parsed_backend = DatasetRegistryBackend(backend)
    except ValueError:
        valid = ", ".join(b.value for b in DatasetRegistryBackend)
        console.print(f"[red]Invalid backend '{backend}'. Valid values: {valid}[/red]")
        raise typer.Exit(code=1)

    registry_client = None
    if parsed_backend == DatasetRegistryBackend.AZURE_ML:
        if not azure_config:
            console.print("[red]--azure-config is required when --backend azure_ml[/red]")
            raise typer.Exit(code=1)
        from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415
        from agentic_mlops.integrations.dataset_registry import (  # noqa: PLC0415
            AzureMLDatasetRegistryClient,
        )

        registry_client = AzureMLDatasetRegistryClient(AzureMLConfig.from_yaml(azure_config))

    artifacts_dir = Path(output_dir) if output_dir else Path(registry_dir) / "version_out"

    agent = DatasetVersioningAgent(artifacts_dir=artifacts_dir, registry_client=registry_client)
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            registry_dir=registry_dir,
            parent_version=parent_version,
            workflow_id=workflow_id,
            validation_report_path=validation_report,
            label_quality_report_path=label_quality_report,
            approved_by=approved_by,
            source_batches=(
                [b.strip() for b in source_batches.split(",") if b.strip()]
                if source_batches
                else []
            ),
            backend=parsed_backend,
        )
    )

    _print_dataset_versioning_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("model-decision")
def model_decision(
    evaluation_report: str = typer.Argument(..., help="Path to evaluation_report.json"),
    promotion_policy: str = typer.Option(
        None,
        "--promotion-policy",
        help="Path to promotion_policy.yaml (for baseline-improvement requirement)",
    ),
    evaluation_config: str = typer.Option(
        None,
        "--evaluation-config",
        help="Path to evaluation config YAML (for max_latency_ms / max_model_size_mb)",
    ),
    baseline_report: str = typer.Option(
        None,
        "--baseline-report",
        help="Path to a baseline evaluation_report.json to compare against",
    ),
    auto_baseline: bool = typer.Option(
        False,
        "--auto-baseline/--no-auto-baseline",
        help=(
            "Auto-resolve the best registered model as baseline from --registry-dir. "
            "Ignored when --baseline-report is also set."
        ),
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry",
        "--registry-dir",
        help="Local model registry root (used when --auto-baseline is set)",
    ),
    measured_latency_ms: float = typer.Option(
        None, "--measured-latency-ms", help="Externally-measured inference latency, in ms"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the decision report (default: <evaluation_report dir>/decision_out)",
    ),
) -> None:
    """Turn an evaluation report into an explainable promote/reject/retrain decision."""
    from agentic_mlops.agents.model_decision import ModelDecisionAgent  # noqa: PLC0415
    from agentic_mlops.contracts.model_decision import ModelDecisionInput  # noqa: PLC0415

    artifacts_dir = (
        Path(output_dir) if output_dir else Path(evaluation_report).parent / "decision_out"
    )

    # Auto-resolve baseline from registry when requested and no explicit path given
    resolved_baseline = baseline_report
    if not resolved_baseline and auto_baseline:
        from agentic_mlops.tools.baseline_resolver import BaselineResolver  # noqa: PLC0415

        resolver = BaselineResolver()
        resolved = resolver.resolve(Path(registry_dir))
        if resolved is not None:
            synthetic = resolver.write_synthetic_report(resolved, output_dir=artifacts_dir)
            resolved_baseline = str(synthetic)
            console.print(
                f"[dim]Auto-baseline: {resolved.model_name} v{resolved.version} "
                f"(mAP50={resolved.map50:.4f})[/dim]"
            )
        else:
            console.print("[yellow]--auto-baseline: no registered models found; skipping.[/yellow]")

    agent = ModelDecisionAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        ModelDecisionInput(
            evaluation_report_path=evaluation_report,
            promotion_policy_path=promotion_policy,
            evaluation_config_path=evaluation_config,
            baseline_report_path=resolved_baseline,
            measured_latency_ms=measured_latency_ms,
        )
    )

    _print_model_decision_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("compare-models")
def compare_models(
    report_paths: list[str] = typer.Argument(
        ..., help="Paths to evaluation_report.json files to compare (two or more)"
    ),
    model_names: str = typer.Option(
        None,
        "--model-names",
        help="Comma-separated model labels (default: path stems)",
    ),
    promotion_policy: str = typer.Option(
        None,
        "--promotion-policy",
        help="Path to promotion_policy.yaml — marks which models pass the policy",
    ),
    weights: str = typer.Option(
        None,
        "--weights",
        help=("Comma-separated metric=value weights, e.g. " "'map50=0.5,recall=0.3,precision=0.2'"),
    ),
    latency_ms: str = typer.Option(
        None,
        "--latency-ms",
        help="Comma-separated measured latency in ms (index-aligned with report paths)",
    ),
    model_size_mb: str = typer.Option(
        None,
        "--model-size-mb",
        help="Comma-separated measured model size in MB (index-aligned with report paths)",
    ),
    output_dir: str = typer.Option(
        "outputs/comparison",
        "--output-dir",
        help="Where to write comparison_report.json/md",
    ),
) -> None:
    """Rank multiple trained models by a weighted composite score.

    Accepts two or more evaluation_report.json files produced by
    ``agentic-mlops evaluate``. Prints a ranked table and writes
    comparison_report.json and comparison_report.md.
    """
    from agentic_mlops.agents.model_comparison import ModelComparisonAgent  # noqa: PLC0415
    from agentic_mlops.contracts.model_comparison import (  # noqa: PLC0415
        ComparisonWeights,
        ModelComparisonInput,
    )

    if len(report_paths) < 2:
        console.print("[red]Provide at least two report paths to compare.[/red]")
        raise typer.Exit(code=1)

    parsed_names: list[str] | None = (
        [n.strip() for n in model_names.split(",")] if model_names else None
    )

    parsed_weights: ComparisonWeights | None = None
    if weights:
        try:
            wdict = dict(kv.split("=") for kv in weights.split(","))
            parsed_weights = ComparisonWeights(**{k: float(v) for k, v in wdict.items()})
        except Exception as exc:
            console.print(f"[red]Invalid --weights format: {exc}[/red]")
            raise typer.Exit(code=1)

    parsed_latency: list[float | None] | None = None
    if latency_ms:
        parsed_latency = [float(x.strip()) if x.strip() else None for x in latency_ms.split(",")]

    parsed_size: list[float | None] | None = None
    if model_size_mb:
        parsed_size = [float(x.strip()) if x.strip() else None for x in model_size_mb.split(",")]

    inp = ModelComparisonInput(
        report_paths=report_paths,
        model_names=parsed_names,
        promotion_policy_path=promotion_policy,
        weights=parsed_weights,
        measured_latency_ms=parsed_latency,
        measured_model_size_mb=parsed_size,
        output_dir=output_dir,
    )

    agent = ModelComparisonAgent(artifacts_dir=Path(output_dir))
    result = agent.run(inp)

    if result.success:
        console.print(f"\n[bold]Model Comparison — {result.total_models} models[/bold]")
        console.print(f"Winner: [green]{result.winner or '(none)'}[/green]")
        if result.policy_passing is not None:
            console.print(f"Policy passing: {result.policy_passing}/{result.total_models}")
        console.print("")
        for r in result.rankings:
            policy_mark = " ✓" if r.passes_policy else (" ✗" if r.passes_policy is False else "")
            map50 = f"{r.map50:.3f}" if r.map50 is not None else "—"
            console.print(
                f"  #{r.rank} {r.model_name}{policy_mark}  "
                f"mAP50={map50}  score={r.composite_score:.3f}"
            )
        if result.comparison_report_path:
            console.print(f"\nReport: {result.comparison_report_path}")
    else:
        console.print(f"[red]Comparison failed: {result.message}[/red]")
        for err in result.errors:
            console.print(f"  [red]• {err}[/red]")

    if not result.success:
        raise typer.Exit(code=1)


@app.command("deploy-model")
def deploy_model(
    model_path: str = typer.Argument(
        None, help="Path to a registered model's weights (best.pt) — required for --backend local"
    ),
    model_name: str = typer.Option(
        ..., "--model-name", help="Model name for the deployment record"
    ),
    model_version: int = typer.Option(
        None, "--model-version", help="Registered model version, if known"
    ),
    target: str = typer.Option(
        "staging", "--target", help="Deployment target: staging | production"
    ),
    export_format: str = typer.Option(
        "onnx",
        "--export-format",
        help="Export format: onnx (requires the onnx package) | pt (passthrough)",
    ),
    deployment_dir: str = typer.Option(
        "outputs/deployments", "--deployment-dir", help="Root of the local deployment registry"
    ),
    endpoint_name: str = typer.Option(
        None, "--endpoint-name", help="Endpoint name (default: <model-name>-<target>)"
    ),
    production_approval: str = typer.Option(
        None,
        "--production-approval",
        help=(
            "Path to an approval_decision.json with status='approved' — "
            "required for --target production"
        ),
    ),
    rollback_plan: str = typer.Option(
        None, "--rollback-plan", help="Rollback plan text — required for --target production"
    ),
    backend: str = typer.Option(
        "local", "--backend", help="Deployment backend: local | azure_ml | docker | aks"
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help="Path to Azure ML config YAML — required for --backend azure_ml",
    ),
    azure_model_name: str = typer.Option(
        None,
        "--azure-model-name",
        help="Registered Azure ML Model asset name — required for --backend azure_ml",
    ),
    azure_model_version: int = typer.Option(
        None,
        "--azure-model-version",
        help="Registered Azure ML Model asset version — required for --backend azure_ml",
    ),
    docker_config: str = typer.Option(
        None,
        "--docker-config",
        help="Path to Docker config YAML — required for --backend docker or aks",
    ),
    aks_config: str = typer.Option(
        None,
        "--aks-config",
        help="Path to AKS config YAML — required for --backend aks",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the deployment report (default: <deployment-dir>/deploy_out)",
    ),
) -> None:
    """Export, smoke-test, and deploy a registered model to staging or production.

    --backend local (default) writes a versioned local release directory.
    --backend azure_ml deploys an already-registered Azure ML Model asset to a
    real Managed Online Endpoint via the Azure ML SDK v2.
    """
    from agentic_mlops.agents.deployment import DeploymentAgent  # noqa: PLC0415
    from agentic_mlops.contracts.deployment import (  # noqa: PLC0415
        DeploymentBackend,
        DeploymentInput,
        DeploymentTarget,
        ExportFormat,
    )

    try:
        parsed_target = DeploymentTarget(target)
    except ValueError:
        valid = ", ".join(t.value for t in DeploymentTarget)
        console.print(f"[red]Invalid target '{target}'. Valid values: {valid}[/red]")
        raise typer.Exit(code=1)

    try:
        parsed_format = ExportFormat(export_format)
    except ValueError:
        valid_f = ", ".join(f.value for f in ExportFormat)
        console.print(
            f"[red]Invalid export-format '{export_format}'. Valid values: {valid_f}[/red]"
        )
        raise typer.Exit(code=1)

    try:
        parsed_backend = DeploymentBackend(backend)
    except ValueError:
        valid_b = ", ".join(b.value for b in DeploymentBackend)
        console.print(f"[red]Invalid backend '{backend}'. Valid values: {valid_b}[/red]")
        raise typer.Exit(code=1)

    deployer = None
    if parsed_backend == DeploymentBackend.AZURE_ML:
        if not azure_config:
            console.print("[red]--azure-config is required when --backend azure_ml[/red]")
            raise typer.Exit(code=1)
        from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415
        from agentic_mlops.integrations.azure_ml_online_endpoint import (  # noqa: PLC0415
            AzureMLOnlineEndpointDeployer,
        )
        from agentic_mlops.tools.deployer import ModelDeployer  # noqa: PLC0415

        azure_deployer = AzureMLOnlineEndpointDeployer(AzureMLConfig.from_yaml(azure_config))
        deployer = ModelDeployer(azure_deployer=azure_deployer)
    elif parsed_backend in (DeploymentBackend.DOCKER, DeploymentBackend.AKS):
        if not docker_config:
            console.print(
                f"[red]--docker-config is required when --backend {parsed_backend.value}[/red]"
            )
            raise typer.Exit(code=1)
        if parsed_backend == DeploymentBackend.AKS and not aks_config:
            console.print("[red]--aks-config is required when --backend aks[/red]")
            raise typer.Exit(code=1)
        if not model_path:
            console.print(
                f"[red]model_path is required when --backend {parsed_backend.value}[/red]"
            )
            raise typer.Exit(code=1)
        from agentic_mlops.contracts.docker import AksConfig, DockerConfig  # noqa: PLC0415
        from agentic_mlops.integrations.docker_client import DockerClient  # noqa: PLC0415
        from agentic_mlops.tools.deployer import ModelDeployer  # noqa: PLC0415

        d_cfg = DockerConfig.from_yaml(docker_config)
        d_client = DockerClient()
        a_cfg = AksConfig.from_yaml(aks_config) if aks_config else None
        a_client = None
        if parsed_backend == DeploymentBackend.AKS:
            from agentic_mlops.integrations.aks_client import AksClient  # noqa: PLC0415

            a_client = AksClient(a_cfg)
        deployer = ModelDeployer(
            docker_client=d_client,
            docker_config=d_cfg,
            aks_client=a_client,
            aks_config=a_cfg,
        )
    elif not model_path:
        console.print("[red]model_path is required when --backend local[/red]")
        raise typer.Exit(code=1)

    artifacts_dir = Path(output_dir) if output_dir else Path(deployment_dir) / "deploy_out"

    agent = DeploymentAgent(artifacts_dir=artifacts_dir, deployer=deployer)
    result = agent.run(
        DeploymentInput(
            model_path=model_path,
            model_name=model_name,
            model_version=model_version,
            target=parsed_target,
            export_format=parsed_format,
            deployment_dir=deployment_dir,
            endpoint_name=endpoint_name,
            production_approval_path=production_approval,
            rollback_plan=rollback_plan,
            backend=parsed_backend,
            azure_model_name=azure_model_name,
            azure_model_version=azure_model_version,
        )
    )

    _print_deployment_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("monitor")
def monitor(
    predictions_log: str = typer.Argument(
        None,
        help=(
            "Path to a JSON-Lines predictions log (required when --source local, "
            "the default; omit when --source azure-monitor)"
        ),
    ),
    endpoint_name: str = typer.Option(..., "--endpoint-name", help="Deployed endpoint name"),
    source: str = typer.Option(
        "local",
        "--source",
        help="Log source: 'local' (read a JSONL file) or 'azure-monitor' (query App Insights)",
    ),
    app_insights_workspace_id: str = typer.Option(
        None,
        "--app-insights-workspace-id",
        help="App Insights Application ID (GUID). Required when --source azure-monitor",
    ),
    model_version: str = typer.Option(
        "", "--model-version", help="Deployed model version, if known"
    ),
    monitoring_window: str = typer.Option(
        "24h", "--monitoring-window", help="Window ending at the log's latest timestamp: 24h/7d/30m"
    ),
    baseline_class_distribution: str = typer.Option(
        None,
        "--baseline-class-distribution",
        help="Path to a JSON {class: count_or_proportion} baseline for drift detection",
    ),
    critical_classes: str = typer.Option(
        None, "--critical-classes", help="Comma-separated classes to watch for representation drop"
    ),
    low_confidence_threshold: float = typer.Option(
        0.5, "--low-confidence-threshold", help="Per-image weakest-detection floor for hard samples"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the monitoring report (default: <predictions-log dir>/monitoring_out)",
    ),
) -> None:
    """Analyze a predictions log for drift/latency/confidence issues and recommend an action."""
    from agentic_mlops.agents.monitoring import MonitoringAgent  # noqa: PLC0415
    from agentic_mlops.contracts.monitoring import MonitoringInput  # noqa: PLC0415

    # Normalize source value: CLI uses "azure-monitor", contract uses "azure_monitor"
    source_norm = source.replace("-", "_")
    if source_norm not in ("local", "azure_monitor"):
        console.print(f"[red]Invalid --source '{source}'. Use 'local' or 'azure-monitor'.[/red]")
        raise typer.Exit(code=1)

    log_client = None
    if source_norm == "local":
        if not predictions_log:
            console.print("[red]predictions_log argument is required when --source local[/red]")
            raise typer.Exit(code=1)
        artifacts_dir = (
            Path(output_dir) if output_dir else Path(predictions_log).parent / "monitoring_out"
        )
    else:
        if not app_insights_workspace_id:
            console.print(
                "[red]--app-insights-workspace-id is required when --source azure-monitor[/red]"
            )
            raise typer.Exit(code=1)
        from agentic_mlops.integrations.appinsights_log_client import (  # noqa: PLC0415
            ApplicationInsightsLogClient,
        )
        from agentic_mlops.tools.monitor import _parse_window  # noqa: PLC0415

        window_seconds = _parse_window(monitoring_window)
        if window_seconds is None:
            console.print(
                f"[red]Invalid --monitoring-window '{monitoring_window}'. "
                "Expected e.g. '24h', '7d', '30m'.[/red]"
            )
            raise typer.Exit(code=1)
        try:
            log_client = ApplicationInsightsLogClient(
                workspace_id=app_insights_workspace_id,
                endpoint_name=endpoint_name,
                window_seconds=window_seconds,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        artifacts_dir = Path(output_dir) if output_dir else Path(".") / "monitoring_out"

    agent = MonitoringAgent(artifacts_dir=artifacts_dir, log_client=log_client)
    result = agent.run(
        MonitoringInput(
            endpoint_name=endpoint_name,
            model_version=model_version,
            source=source_norm,
            predictions_log_path=predictions_log or "",
            app_insights_workspace_id=app_insights_workspace_id,
            monitoring_window=monitoring_window,
            baseline_class_distribution_path=baseline_class_distribution,
            critical_classes=(
                [c.strip() for c in critical_classes.split(",") if c.strip()]
                if critical_classes
                else []
            ),
            low_confidence_threshold=low_confidence_threshold,
        )
    )

    _print_monitoring_result(result)

    if not result.success:
        raise typer.Exit(code=1)


@app.command("ingest-hard-samples")
def ingest_hard_samples(
    hard_samples_manifest: str = typer.Argument(
        ..., help="Path to hard_samples_manifest.json written by the monitor command"
    ),
    images_source_dir: str = typer.Option(
        ..., "--images-source-dir", help="Directory containing the original inference images"
    ),
    dataset_name: str = typer.Option(
        ..., "--dataset-name", help="Dataset name forwarded to DataIntakeAgent"
    ),
    source: str = typer.Option(
        None,
        "--source",
        help="Provenance label (default: 'hard_sample_mining')",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Artifacts directory (default: <manifest_dir>/hard_sample_ingestion_out)",
    ),
    expected_formats: str = typer.Option(
        "jpg,jpeg,png", "--expected-formats", help="Comma-separated accepted extensions"
    ),
    min_files: int = typer.Option(1, "--min-files"),
    corrupted_ratio_threshold: float = typer.Option(0.05, "--corrupted-ratio-threshold"),
    duplicate_ratio_threshold: float = typer.Option(0.20, "--duplicate-ratio-threshold"),
) -> None:
    """Stage hard samples from a monitoring manifest and run data intake over them."""
    from agentic_mlops.agents.hard_sample_ingestion import HardSampleIngestionAgent  # noqa: PLC0415
    from agentic_mlops.contracts.hard_sample_ingestion import (
        HardSampleIngestionInput,  # noqa: PLC0415
    )

    artifacts_dir = (
        Path(output_dir)
        if output_dir
        else Path(hard_samples_manifest).parent / "hard_sample_ingestion_out"
    )

    agent = HardSampleIngestionAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=hard_samples_manifest,
            images_source_dir=images_source_dir,
            dataset_name=dataset_name,
            source=source,
            expected_formats=[f.strip() for f in expected_formats.split(",") if f.strip()],
            min_files=min_files,
            corrupted_ratio_threshold=corrupted_ratio_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
        )
    )

    if result.success:
        console.print(
            f"[green]Hard sample ingestion complete.[/green] "
            f"Staged: {result.num_images_found}/{result.num_hard_samples_in_manifest} images. "
            f"Intake status: {result.intake_status}."
        )
    else:
        console.print(f"[red]Hard sample ingestion failed:[/red] {result.message}")

    for w in result.warnings:
        console.print(f"[yellow]  warning:[/yellow] {w}")

    if not result.success:
        raise typer.Exit(code=1)


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
    mlflow_config: str = typer.Option(None, "--mlflow-config", help="Path to MLflow config YAML"),
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
    weights_path: str = typer.Option(None, "--weights-path", help="Path to model weights (.pt)"),
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
    mlflow_config: str = typer.Option(None, "--mlflow-config", help="Path to MLflow config YAML"),
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
    candidate_model: str = typer.Option("unknown", "--candidate-model", help="Model name or ID"),
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


@app.command("approve-training")
def approve_training(
    dataset_report: str = typer.Argument(..., help="Path to dataset_quality_report.json"),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save approval artifacts (default: <report dir>/training_approval_out)",
    ),
    approver: str = typer.Option(None, "--approver", help="Name of the approver"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Prompt for decision interactively"
    ),
    action: str = typer.Option(
        None,
        "--action",
        help="Action in non-interactive mode: approve_training | reject_training | cancel",
    ),
    comment: str = typer.Option(None, "--comment", help="Optional human comment"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force approve training on a dataset with warnings, in non-interactive mode.",
    ),
) -> None:
    """H4 gate: review a dataset validation report and approve/reject starting training."""
    from agentic_mlops.agents.training_approval import TrainingApprovalAgent  # noqa: PLC0415
    from agentic_mlops.contracts.training_approval import (  # noqa: PLC0415
        TrainingApprovalAction,
        TrainingApprovalInput,
    )

    parsed_action: TrainingApprovalAction | None = None
    if action:
        try:
            parsed_action = TrainingApprovalAction(action)
        except ValueError:
            valid = ", ".join(a.value for a in TrainingApprovalAction)
            console.print(f"[red]Invalid action '{action}'. Valid values: {valid}[/red]")
            raise typer.Exit(code=1)

    out_dir = output_dir or str(Path(dataset_report).parent / "training_approval_out")

    agent = TrainingApprovalAgent(artifacts_dir=Path(out_dir))
    result = agent.run(
        TrainingApprovalInput(
            dataset_report_path=dataset_report,
            approver=approver,
            output_dir=out_dir,
            interactive=interactive,
            action=parsed_action,
            comment=comment,
            force=force,
        )
    )

    _print_training_approval_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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

    artifacts_dir = Path(output_dir) if output_dir else Path(registry_dir) / "registration_out"

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
        None,
        "--training-runner",
        help="Override training runner: fake | local-yolo | azure-ml | azure-ml-pipeline",
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
    mlflow_config: str = typer.Option(None, "--mlflow-config", help="Path to MLflow config YAML"),
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
            console.print(f"[red]Invalid approval-action '{approval_action}'. Valid: {valid}[/red]")
            raise typer.Exit(code=1)

    try:
        parsed_backend = RegistryBackend(registry_backend)
    except ValueError:
        valid_b = ", ".join(b.value for b in RegistryBackend)
        console.print(f"[red]Invalid registry-backend '{registry_backend}'. Valid: {valid_b}[/red]")
        raise typer.Exit(code=1)

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)
    artifact_store = _make_artifact_store(azure_config)

    workflow = MVPWorkflow(
        mlflow_client=mlflow_client if mlflow_cfg.enabled else None,
        mlflow_config=mlflow_cfg if mlflow_cfg.enabled else None,
        artifact_store=artifact_store,
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


@app.command("run-workflow")
def run_workflow(
    workflow_id: str = typer.Option(..., "--workflow-id", help="Unique ID for this workflow run"),
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to an orchestrator config YAML (see configs/orchestrator.example.yaml)",
    ),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Root directory for state.json/audit_log.jsonl/artifacts"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume a previously started workflow_id, skipping completed steps"
    ),
    resume_from_step: str = typer.Option(
        None,
        "--resume-from-step",
        help=(
            "Rewind a saved workflow to the named step and re-run from there. "
            "All prior step outputs are kept; the named step and everything after "
            "it are re-run. Implies --resume. "
            "Valid step names: data_intake, dataset_structuring, dataset_validation, "
            "dataset_versioning, training_approval, training, evaluation, "
            "model_decision, approval, model_registry, deployment."
        ),
    ),
    trigger: str = typer.Option("manual", "--trigger", help="What triggered this run"),
    mlflow_config: str = typer.Option(None, "--mlflow-config", help="Path to MLflow config YAML"),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--disable-mlflow",
        help="Enable MLflow tracking as one parent run across all steps (default: disabled)",
    ),
) -> None:
    """Run the full, configurable Orchestrator pipeline (any subset of PIPELINE_STEPS).

    Unlike `run-mvp` (fixed 5-step chain), this reads step selection and every
    step's config from a single YAML file, persists state.json/audit_log.jsonl
    under --runs-dir/<workflow-id>/, and supports --resume after a pause (e.g.
    at the human approval gate) or a failure. --resume-from-step <name> rewinds
    a saved workflow to re-run from a specific step without replaying earlier ones
    (prior outputs are preserved so downstream steps still have their context).
    With --enable-mlflow, the whole run is tracked as one parent MLflow run.
    """
    from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

    try:
        inp = OrchestratorInput.from_yaml(
            config,
            workflow_id=workflow_id,
            runs_dir=runs_dir,
            resume=resume,
            resume_from_step=resume_from_step or None,
            trigger=trigger,
        )
    except Exception as exc:
        console.print(f"[red]Cannot load orchestrator config '{config}': {exc}[/red]")
        raise typer.Exit(code=1)

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)
    artifact_store = _make_artifact_store(inp.azure_config_path)

    result = OrchestratorWorkflow(
        mlflow_client=mlflow_client if mlflow_cfg.enabled else None,
        mlflow_config=mlflow_cfg if mlflow_cfg.enabled else None,
        artifact_store=artifact_store,
    ).run(inp)

    _print_orchestrator_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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


@app.command("serve")
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Directory containing workflow run state files"
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry", "--registry-dir", help="Local model registry root directory"
    ),
    datasets_dir: str = typer.Option(
        "outputs/dataset_registry",
        "--dataset-registry-dir",
        help="Local dataset registry root directory",
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development mode)"),
    password: str = typer.Option(
        "",
        "--password",
        envvar="DASHBOARD_PASSWORD",
        help="HTTP Basic Auth password. Omit (or leave DASHBOARD_PASSWORD unset) to disable auth.",
    ),
    user: str = typer.Option(
        "admin",
        "--user",
        envvar="DASHBOARD_USER",
        help="HTTP Basic Auth username (default: admin).",
    ),
    slack_signing_secret: str = typer.Option(
        "",
        "--slack-signing-secret",
        envvar="SLACK_SIGNING_SECRET",
        help=(
            "Slack signing secret for validating interactive button callbacks "
            "at POST /webhooks/approve/slack. Get it from your Slack app's "
            "'Basic Information' page. Also readable from the "
            "SLACK_SIGNING_SECRET environment variable."
        ),
    ),
    teams_signing_secret: str = typer.Option(
        "",
        "--teams-signing-secret",
        envvar="TEAMS_SIGNING_SECRET",
        help=(
            "Teams connector security token (base64-encoded symmetric key) for "
            "validating interactive HttpPOST button callbacks at "
            "POST /webhooks/approve/teams. Get it from your Teams connector "
            "configuration page. Also readable from the TEAMS_SIGNING_SECRET "
            "environment variable."
        ),
    ),
) -> None:
    """Start the MLOps web dashboard (requires the 'web' extra: pip install -e '.[web]')."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]The 'web' extra is required. Install with:[/red]\n" "  pip install -e '.[web]'"
        )
        raise typer.Exit(code=1) from None

    from agentic_mlops.web.app import create_app

    web_app = create_app(
        runs_dir=runs_dir,
        registry_dir=registry_dir,
        datasets_dir=datasets_dir,
        dashboard_user=user,
        dashboard_password=password,
        slack_signing_secret=slack_signing_secret,
        teams_signing_secret=teams_signing_secret,
    )
    console.print(f"[bold green]MLOps Dashboard[/bold green]  http://{host}:{port}")
    console.print(f"  Runs dir     : [bold]{runs_dir}[/bold]")
    console.print(f"  Registry dir : [bold]{registry_dir}[/bold]")
    console.print(f"  Datasets dir : [bold]{datasets_dir}[/bold]")
    if password:
        console.print(f"  Auth         : Basic Auth (user: [bold]{user}[/bold])")
    else:
        console.print(
            "  Auth         : [yellow]disabled[/yellow] — set DASHBOARD_PASSWORD to enable"
        )
    uvicorn.run(web_app, host=host, port=port, reload=reload)


@app.command("cost-report")
def cost_report(
    workflow_id: str = typer.Argument(..., help="Workflow ID to generate a cost report for"),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Directory containing workflow run state files"
    ),
    pricing_config: str = typer.Option(
        None,
        "--pricing-config",
        help=(
            "YAML file with ComputePricing config (rates: {instance_type: usd/hr}). "
            "Uses built-in Azure ML default rates when omitted."
        ),
    ),
    output_file: str = typer.Option(
        None,
        "--output-file",
        help="Write cost_summary.json to this path instead of the workflow artifacts dir.",
    ),
) -> None:
    """Estimate the compute cost of a completed workflow run.

    Reads training/evaluation timing from the workflow's state.json and outputs a
    cost summary.  Rates default to built-in Azure ML pricing; override with
    --pricing-config.
    """
    import json as _json

    from agentic_mlops.contracts.cost import ComputePricing
    from agentic_mlops.tools.cost_tracker import CostTracker

    workflow_dir = Path(runs_dir) / workflow_id
    state_path = workflow_dir / "state.json"
    if not state_path.exists():
        console.print(f"[red]State file not found:[/red] {state_path}")
        raise typer.Exit(code=1)

    state = _json.loads(state_path.read_text(encoding="utf-8"))
    step_outputs: dict = state.get("step_outputs", {})

    pricing = ComputePricing()
    if pricing_config:
        try:
            pricing = ComputePricing.from_yaml(pricing_config)
        except Exception as exc:
            console.print(f"[red]Cannot load pricing config:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    tracker = CostTracker(pricing)

    # Record training cost
    training = step_outputs.get("training")
    if training and not training.get("skipped"):
        tracker.record(
            step="training",
            runner=state.get("training_runner") or "unknown",
            compute_type=training.get("compute_type") or "unknown",
            started_at=training.get("started_at"),
            completed_at=training.get("completed_at"),
        )

    # Record evaluation cost
    evaluation = step_outputs.get("evaluation")
    if evaluation and not evaluation.get("skipped"):
        tracker.record(
            step="evaluation",
            runner=state.get("evaluation_runner") or "unknown",
            compute_type=evaluation.get("compute_type") or "unknown",
            started_at=evaluation.get("started_at"),
            completed_at=evaluation.get("completed_at"),
        )

    summary = tracker.summary(workflow_id)

    # Write JSON
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = workflow_dir / "artifacts" / "cost_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(summary.model_dump(), indent=2), encoding="utf-8")

    # Print table
    console.print(f"\n[bold]Cost Report — Workflow:[/bold] {workflow_id}")
    console.print(f"  Currency : {summary.currency}")
    console.print(f"  Steps    : {len(summary.entries)}")
    for entry in summary.entries:
        dur = f"{entry.duration_seconds:.0f}s" if entry.duration_seconds is not None else "—"
        console.print(
            f"  {entry.step:20s}  runner={entry.runner:18s}  "
            f"compute={entry.compute_type:22s}  "
            f"duration={dur:8s}  "
            f"cost={entry.estimated_cost:.4f} {entry.currency}"
        )
    console.print(
        f"\n  [bold]Total estimated cost:[/bold] {summary.total_cost:.4f} {summary.currency}"
    )
    console.print(f"\n  Report written to: {out_path}")


@app.command("status")
def status(
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    registry_dir: str = typer.Option(
        "outputs/model_registry", "--registry-dir", help="Model registry root"
    ),
    dataset_registry_dir: str = typer.Option(
        "outputs/dataset_registry", "--dataset-registry-dir", help="Dataset registry root"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Max recent workflows to show"),
) -> None:
    """Show a quick health snapshot: workflows, models, datasets, monitoring alerts."""
    from agentic_mlops.web import reader  # noqa: PLC0415

    _rd = Path(runs_dir)
    _mod = Path(registry_dir)
    _ds = Path(dataset_registry_dir)

    _WF_ICON = {
        "completed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "running": "[cyan]⟳[/cyan]",
        "pending_approval": "[yellow]⏳[/yellow]",
        "blocked": "[red]⊘[/red]",
    }
    _ACTION_COLOR = {
        "no_action": "green",
        "notify_ops": "yellow",
        "need_more_data": "yellow",
        "model_review": "red",
        "create_retraining_request": "red",
    }

    def _short_dt(ts: str | None) -> str:
        if not ts:
            return "—"
        return ts[:16].replace("T", " ")

    console.print()

    # ── Workflows ──────────────────────────────────────────────────────────────
    workflows = reader.list_workflows(_rd)
    pending = [w for w in workflows if w.get("status") == "pending_approval"]
    recent = workflows[:limit]

    console.print(
        f"  [bold]Workflows[/bold]  "
        f"[dim]({len(workflows)} total"
        + (f", [yellow]{len(pending)} awaiting approval[/yellow]" if pending else "")
        + ")[/dim]"
    )
    if recent:
        for w in recent:
            st = w.get("status", "?")
            icon = _WF_ICON.get(st, "·")
            wid = w.get("workflow_id", "?")
            state = w.get("current_state", "")
            started = _short_dt(w.get("started_at"))
            pend = w.get("pending_approval_id")
            pending_id = f"  [dim]→ {pend}[/dim]" if pend else ""
            console.print(f"    {icon}  {wid:<24} {st:<20} {state:<30} {started}{pending_id}")
    else:
        console.print("    [dim]No workflows found.[/dim]")

    console.print()

    # ── Models ─────────────────────────────────────────────────────────────────
    models = reader.list_models(_mod)
    console.print(f"  [bold]Models[/bold]  [dim]({len(models)} registered)[/dim]")
    if models:
        for m in models[:3]:
            name = m.get("model_name", "?")
            ver = m.get("version", "?")
            lineage = m.get("lineage") or {}
            map50 = lineage.get("map50")
            map50_str = f"mAP50={map50:.4f}" if isinstance(map50, int | float) else ""
            approved_by = lineage.get("approved_by") or m.get("approved_by", "")
            reg_at = _short_dt(m.get("registered_at"))
            by_str = f"by {approved_by}" if approved_by else ""
            meta = "  ".join(filter(None, [map50_str, by_str, reg_at]))
            console.print(f"    [green]✓[/green]  {name}  v{ver}  [dim]{meta}[/dim]")
    else:
        console.print("    [dim]No registered models.[/dim]")

    console.print()

    # ── Datasets ───────────────────────────────────────────────────────────────
    datasets = reader.list_datasets(_ds)
    console.print(f"  [bold]Datasets[/bold]  [dim]({len(datasets)} registered)[/dim]")
    if datasets:
        for d in datasets[:3]:
            name = d.get("dataset_name", "?")
            ver = d.get("version", "?")
            reg_at = _short_dt(d.get("registered_at"))
            console.print(f"    [green]✓[/green]  {name}  v{ver}  [dim]{reg_at}[/dim]")
    else:
        console.print("    [dim]No registered datasets.[/dim]")

    console.print()

    # ── Monitoring ─────────────────────────────────────────────────────────────
    mon_reports = reader.list_monitoring_reports(_rd)
    active_alerts = [
        r for r in mon_reports if r.get("recommended_action", "no_action") != "no_action"
    ]
    console.print(
        f"  [bold]Monitoring[/bold]  [dim]({len(mon_reports)} report(s)"
        + (f", [red]{len(active_alerts)} alert(s)[/red]" if active_alerts else "")
        + ")[/dim]"
    )
    if active_alerts:
        for r in active_alerts[:3]:
            action = r.get("recommended_action", "")
            color = _ACTION_COLOR.get(action, "white")
            endpoint = r.get("endpoint_name", "?")
            wid = r.get("workflow_id", "?")
            ts = _short_dt(r.get("generated_at"))
            console.print(
                f"    [{color}]![/{color}]  {endpoint}  [{color}]{action}[/{color}]"
                f"  [dim]{wid}  {ts}[/dim]"
            )
    elif mon_reports:
        latest = mon_reports[0]
        ts = _short_dt(latest.get("generated_at"))
        console.print(
            f"    [green]✓[/green]  [dim]Latest: "
            f"{latest.get('endpoint_name', '?')}  {ts}  no alerts[/dim]"
        )
    else:
        console.print("    [dim]No monitoring reports found.[/dim]")

    console.print()

    # ── Summary ────────────────────────────────────────────────────────────────
    issues = len([w for w in workflows if w.get("status") == "failed"]) + len(active_alerts)
    if pending:
        console.print(
            f"  [yellow bold]{len(pending)} workflow(s) waiting for human approval.[/yellow bold]  "
            "Run [dim]agentic-mlops run-workflow ... --resume[/dim] after setting the action."
        )
    if issues == 0 and not pending:
        console.print("  [green bold]All systems nominal.[/green bold]")
    console.print()


@app.command("show-state")
def show_state(
    workflow_id: str = typer.Argument(
        None,
        help="Workflow ID to inspect. Omit to list all available workflow IDs.",
    ),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    audit: bool = typer.Option(
        False,
        "--audit",
        help="Also print the last audit log events.",
    ),
    audit_lines: int = typer.Option(
        20,
        "--audit-lines",
        help="Number of recent audit events to show with --audit.",
    ),
) -> None:
    """Show the detailed state of a workflow run (step table, artifacts, approval info).

    Without WORKFLOW_ID, lists all workflow IDs found under --runs-dir.
    """
    from agentic_mlops.tools.state_inspector import WorkflowStateInspector  # noqa: PLC0415

    inspector = WorkflowStateInspector()

    if workflow_id is None:
        ids = inspector.list_workflows(runs_dir)
        if not ids:
            console.print(f"[dim]No workflows found in {runs_dir}.[/dim]")
            return
        console.print(f"\n  [bold]Workflows in {runs_dir}[/bold]\n")
        for wid in ids:
            snap = inspector.inspect(runs_dir, wid, max_audit_events=0)
            if snap is None:
                continue
            _STATUS_COLOR = {
                "completed": "green",
                "failed": "red",
                "running": "cyan",
                "pending_approval": "yellow",
                "blocked": "red",
            }
            color = _STATUS_COLOR.get(snap.status, "white")
            started = (snap.started_at or "")[:16].replace("T", " ")
            console.print(
                f"    [{color}]{snap.status:<20}[/{color}]  " f"{wid:<30}  [dim]{started}[/dim]"
            )
        console.print()
        return

    snap = inspector.inspect(runs_dir, workflow_id, max_audit_events=audit_lines)
    if snap is None:
        console.print(
            f"[red]No state file found for workflow '{workflow_id}' " f"in {runs_dir}.[/red]"
        )
        raise typer.Exit(code=1)

    _STATUS_COLOR = {
        "completed": "green",
        "failed": "red",
        "running": "cyan",
        "pending_approval": "yellow",
        "blocked": "red",
    }
    _STEP_ICON = {
        "completed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "skipped": "[dim]–[/dim]",
        "running": "[cyan]⟳[/cyan]",
        "pending": "[dim]·[/dim]",
    }

    color = _STATUS_COLOR.get(snap.status, "white")
    started = (snap.started_at or "—")[:19].replace("T", " ")
    updated = (snap.updated_at or "—")[:19].replace("T", " ")

    console.print()
    console.print(f"  [bold]Workflow:[/bold]  {snap.workflow_id}")
    console.print(
        f"  [bold]Status:[/bold]   [{color}]{snap.status}[/{color}]"
        f"  [dim]state={snap.current_state}[/dim]"
    )
    console.print(f"  [bold]Started:[/bold]  {started}    [bold]Updated:[/bold]  {updated}")
    if snap.mlflow_run_id:
        console.print(f"  [bold]MLflow:[/bold]   [dim]{snap.mlflow_run_id}[/dim]")
    if snap.pending_approval_id:
        console.print(f"  [bold yellow]Pending approval:[/bold yellow]  {snap.pending_approval_id}")
        console.print(
            "  [dim]→ Set the action in your orchestrator config and re-run with "
            "--resume (or use --resume-from-step)[/dim]"
        )

    console.print()
    console.print(
        f"  [bold]Steps[/bold]  [dim]("
        f"{snap.num_completed} completed"
        + (f", {snap.num_failed} failed" if snap.num_failed else "")
        + (f", {snap.num_skipped} skipped" if snap.num_skipped else "")
        + f" of {len(snap.step_summaries)})[/dim]"
    )
    console.print()

    for ss in snap.step_summaries:
        icon = _STEP_ICON.get(ss.status, "·")
        status_str = f"[dim]{ss.status}[/dim]" if ss.status == "pending" else ss.status
        console.print(f"    {icon}  {ss.name:<28}  {status_str}")
        for art in ss.artifacts[:2]:
            console.print(f"           [dim]{art}[/dim]")

    if audit and snap.audit_events:
        console.print()
        n = len(snap.audit_events)
        console.print(f"  [bold]Audit log[/bold]  [dim](last {n} events)[/dim]")
        console.print()
        for ev in snap.audit_events:
            ts = (ev.get("timestamp") or "")[:19].replace("T", " ")
            event_name = ev.get("event", "?")
            step = ev.get("step", "")
            detail = f"  [dim]{step}[/dim]" if step else ""
            console.print(f"    [dim]{ts}[/dim]  {event_name}{detail}")

    console.print(f"\n  [dim]State file: {snap.state_path}[/dim]")
    console.print()


@app.command("diff-runs")
def diff_runs(
    wf_id_a: str = typer.Argument(..., help="First (baseline) workflow ID"),
    wf_id_b: str = typer.Argument(..., help="Second (target) workflow ID"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
) -> None:
    """Compare two workflow runs: step outcomes and evaluation metrics."""
    from agentic_mlops.tools.run_diff import WorkflowRunDiffer

    diff = WorkflowRunDiffer().diff(runs_dir, wf_id_a, wf_id_b)
    if diff is None:
        missing = []
        from pathlib import Path as _P

        if not (_P(runs_dir) / wf_id_a / "state.json").exists():
            missing.append(wf_id_a)
        if not (_P(runs_dir) / wf_id_b / "state.json").exists():
            missing.append(wf_id_b)
        console.print(f"[red]Workflow(s) not found:[/red] {', '.join(missing)}")
        raise typer.Exit(1)

    _STATUS_COLOUR = {
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
        "running": "yellow",
        "pending": "dim",
    }

    def _sc(s: str | None) -> str:
        if s is None:
            return "[dim]—[/dim]"
        c = _STATUS_COLOUR.get(s, "white")
        return f"[{c}]{s}[/{c}]"

    console.print(f"\n[bold]diff-runs:[/bold] [cyan]{wf_id_a}[/cyan]  →  [cyan]{wf_id_b}[/cyan]")
    console.print("─" * 60)

    # Status
    sa_col = _sc(diff.status_a)
    sb_col = _sc(diff.status_b)
    changed_mark = "  [yellow]←[/yellow]" if diff.status_a != diff.status_b else ""
    console.print(f"  Status:  {sa_col}  →  {sb_col}{changed_mark}")
    if diff.started_at_a or diff.started_at_b:
        ta = (diff.started_at_a or "")[:19].replace("T", " ")
        tb = (diff.started_at_b or "")[:19].replace("T", " ")
        console.print(f"  Started: [dim]{ta}[/dim]  →  [dim]{tb}[/dim]")

    # Steps
    if diff.step_diffs or diff.steps_only_in_a or diff.steps_only_in_b:
        console.print("\n[bold]Steps[/bold]")
        for sd in diff.step_diffs:
            arrow = f"{_sc(sd.status_a)}  →  {_sc(sd.status_b)}"
            marker = "  [yellow]←[/yellow]" if sd.changed else ""
            console.print(f"  {sd.name:<28} {arrow}{marker}")
        for s in diff.steps_only_in_a:
            console.print(f"  [dim]{s:<28} only in {wf_id_a}[/dim]")
        for s in diff.steps_only_in_b:
            console.print(f"  [dim]{s:<28} only in {wf_id_b}[/dim]")

    # Metrics
    if diff.metric_diffs:
        console.print("\n[bold]Metrics[/bold]")
        current_step = ""
        for md in diff.metric_diffs:
            step, key = md.metric.split(".", 1)
            if step != current_step:
                console.print(f"  [bold dim]{step}[/bold dim]")
                current_step = step

            def _fmt(v: float | None) -> str:
                return f"{v:.4f}" if v is not None else "[dim]—[/dim]"

            val_str = f"{_fmt(md.value_a)}  →  {_fmt(md.value_b)}"
            if md.delta is not None:
                sign = "+" if md.delta >= 0 else ""
                arrow = "↑" if md.improved else ("↓" if md.improved is False else "")
                colour = "green" if md.improved else ("red" if md.improved is False else "dim")
                delta_str = f"  [{colour}]({sign}{md.delta:.4f} {arrow})[/{colour}]"
            else:
                delta_str = ""
            console.print(f"    {key:<20} {val_str}{delta_str}")

    if not diff.has_changes and not diff.metric_diffs:
        console.print("\n  [dim]No differences.[/dim]")
    console.print()


@app.command("prune-runs")
def prune_runs(
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    keep_last: int = typer.Option(
        None,
        "--keep-last",
        help="Keep only the N most-recent runs; delete the rest.",
    ),
    older_than_days: float = typer.Option(
        None,
        "--older-than-days",
        help="Delete runs whose started_at is older than N days.",
    ),
    statuses: list[str] = typer.Option(
        None,
        "--status",
        help="Only delete runs with this status (repeatable). E.g. --status completed --status failed.",  # noqa: E501
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Show what would be deleted without actually deleting.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Delete old workflow run directories by age or count."""
    from agentic_mlops.tools.run_pruner import WorkflowPruner

    if keep_last is None and older_than_days is None:
        console.print("[red]Error:[/red] Specify --keep-last or --older-than-days.")
        raise typer.Exit(1)

    try:
        pruner = WorkflowPruner(
            keep_last=keep_last,
            older_than_days=older_than_days,
            statuses=list(statuses) if statuses else None,
            dry_run=True,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    preview = pruner.prune(runs_dir)

    if not preview.deleted:
        console.print("\n  [dim]Nothing to prune.[/dim]\n")
        return

    console.print(f"\n[bold]prune-runs[/bold]  {'(dry run)' if dry_run else ''}")
    console.print("─" * 50)
    for wf in preview.deleted:
        console.print(f"  [red]delete[/red]  {wf}")
    for wf in preview.kept:
        console.print(f"  [green]keep  [/green]  {wf}")
    if preview.skipped:
        for wf in preview.skipped:
            console.print(f"  [yellow]skip  [/yellow]  {wf}  (unreadable state)")
    console.print(f"\n  {len(preview.deleted)} run(s) would be deleted.")

    if dry_run:
        console.print()
        return

    if not yes:
        confirm = typer.confirm(f"\nDelete {len(preview.deleted)} run(s)?", default=False)
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    real_pruner = WorkflowPruner(
        keep_last=keep_last,
        older_than_days=older_than_days,
        statuses=list(statuses) if statuses else None,
        dry_run=False,
    )
    result = real_pruner.prune(runs_dir)
    console.print(f"\n  [green]Deleted {result.deleted_count} run(s).[/green]\n")


@app.command("tag-run")
def tag_run(
    workflow_id: str = typer.Argument(..., help="Workflow ID to tag"),
    tags: list[str] = typer.Argument(
        None,
        help="Tags as key=value pairs (e.g. env=prod version=2). " "Omit to show current tags.",
    ),
    remove: list[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Tag key(s) to remove (repeatable).",
    ),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
) -> None:
    """Add, remove, or display tags on a workflow run.

    Set tags:   agentic-mlops tag-run wf_001 env=prod dataset=v3
    Remove tag: agentic-mlops tag-run wf_001 --remove env
    Show tags:  agentic-mlops tag-run wf_001
    """
    from agentic_mlops.tools.run_tagger import WorkflowRunTagger

    tagger = WorkflowRunTagger()

    try:
        if remove:
            current = tagger.remove(runs_dir, workflow_id, list(remove))
        elif tags:
            parsed: dict[str, str] = {}
            for item in tags:
                if "=" not in item:
                    console.print(f"[red]Error:[/red] Tag '{item}' must be in key=value format.")
                    raise typer.Exit(1)
                k, _, v = item.partition("=")
                parsed[k.strip()] = v.strip()
            current = tagger.add(runs_dir, workflow_id, parsed)
        else:
            current = tagger.get(runs_dir, workflow_id)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Tags for[/bold] [cyan]{workflow_id}[/cyan]")
    console.print("─" * 40)
    if current:
        for k, v in sorted(current.items()):
            console.print(f"  [green]{k}[/green] = {v}")
    else:
        console.print("  [dim](no tags)[/dim]")
    console.print()


@app.command("doctor")
def doctor(
    config: str = typer.Option(
        None,
        "--config",
        help="Orchestrator config YAML to validate (can be passed multiple times)",
        show_default=False,
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help="Azure ML config YAML to validate",
    ),
    mlflow_config: str = typer.Option(
        None,
        "--mlflow-config",
        help="MLflow config YAML to validate",
    ),
    runs_dir: str = typer.Option(
        "runs",
        "--runs-dir",
        help="Workflow runs directory to check for write access",
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry",
        "--registry-dir",
        help="Model registry directory to check for write access",
    ),
    dataset_registry_dir: str = typer.Option(
        "outputs/dataset_registry",
        "--dataset-registry-dir",
        help="Dataset registry directory to check for write access",
    ),
    no_tools: bool = typer.Option(
        False,
        "--no-tools",
        help="Skip external CLI tool checks (docker, az)",
    ),
) -> None:
    """Run a local self-check — verify dependencies, configs, and directory access.

    Checks required and optional Python packages, validates any YAML config files
    you pass, confirms output directories are writable, and probes for optional
    external tools (docker, az CLI).  No network calls are made.
    """
    from agentic_mlops.contracts.doctor import CheckStatus  # noqa: PLC0415
    from agentic_mlops.tools.doctor import SystemDoctor  # noqa: PLC0415

    config_paths: list[tuple[Path, str]] = []
    if config:
        config_paths.append((Path(config), "orchestrator config"))
    if azure_config:
        config_paths.append((Path(azure_config), "azure_ml config"))
    if mlflow_config:
        config_paths.append((Path(mlflow_config), "mlflow config"))

    directory_paths: list[tuple[Path, str]] = [
        (Path(runs_dir), "runs_dir"),
        (Path(registry_dir), "registry_dir"),
        (Path(dataset_registry_dir), "dataset_registry_dir"),
    ]

    report = SystemDoctor().check(
        config_paths=config_paths,
        directory_paths=directory_paths,
        check_docker=not no_tools,
        check_az=not no_tools,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    _STATUS_ICON = {
        CheckStatus.OK: "[green]✓[/green]",
        CheckStatus.WARNING: "[yellow]⚠[/yellow]",
        CheckStatus.ERROR: "[red]✗[/red]",
    }
    _CATEGORY_HEADER = {
        "packages": "Python Packages",
        "configs": "Config Files",
        "directories": "Directories",
        "tools": "External Tools",
    }

    console.print("\n[bold]agentic-mlops doctor[/bold]\n")

    current_cat = ""
    for chk in report.checks:
        if chk.category != current_cat:
            current_cat = chk.category
            console.print(f"  [bold]{_CATEGORY_HEADER.get(current_cat, current_cat)}[/bold]")
        icon = _STATUS_ICON[chk.status]
        console.print(f"    {icon}  {chk.message}")

    console.print()
    ok_s = f"[green]{report.num_ok} ok[/green]"
    warn_s = f"[yellow]{report.num_warnings} warnings[/yellow]"
    err_s = f"[red]{report.num_errors} errors[/red]"
    console.print(f"  Summary: {ok_s}  {warn_s}  {err_s}")

    if report.overall_status == CheckStatus.OK:
        console.print("  [green bold]All checks passed.[/green bold]\n")
    elif report.overall_status == CheckStatus.WARNING:
        console.print(
            "  [yellow]Some optional components are missing — see warnings above.[/yellow]\n"
        )
        raise typer.Exit(code=0)
    else:
        console.print("  [red bold]One or more required checks failed.[/red bold]\n")
        raise typer.Exit(code=1)


@app.command("lint-config")
def lint_config(
    config: str = typer.Argument(
        ...,
        help="Orchestrator config YAML to validate",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 even when the only issues are warnings (no errors).",
    ),
) -> None:
    """Validate an orchestrator YAML config without running anything.

    Checks step names, required fields, file path existence, step dependency order,
    and Azure backend / runner consistency.  No agents are started and no side effects
    are produced.

    Exit codes: 0 = ok (or warnings-only without --strict); 1 = errors present
    (or warnings with --strict).
    """
    from agentic_mlops.contracts.doctor import CheckStatus  # noqa: PLC0415
    from agentic_mlops.tools.config_linter import ConfigLinter  # noqa: PLC0415

    report = ConfigLinter().lint(config)

    _STATUS_ICON = {
        CheckStatus.OK: "[green]✓[/green]",
        CheckStatus.WARNING: "[yellow]⚠[/yellow]",
        CheckStatus.ERROR: "[red]✗[/red]",
    }

    console.print(f"\n[bold]agentic-mlops lint-config[/bold]  {config}\n")

    current_cat = ""
    for chk in report.checks:
        if chk.category != current_cat:
            current_cat = chk.category
            console.print(f"  [bold]{chk.category}[/bold]")
        icon = _STATUS_ICON[chk.status]
        console.print(f"    {icon}  {chk.message}")
        if chk.detail:
            console.print(f"         [dim]{chk.detail}[/dim]")

    console.print()
    ok_s = f"[green]{report.num_ok} ok[/green]"
    warn_s = f"[yellow]{report.num_warnings} warnings[/yellow]"
    err_s = f"[red]{report.num_errors} errors[/red]"
    console.print(f"  Summary: {ok_s}  {warn_s}  {err_s}")

    if report.overall_status == CheckStatus.OK:
        console.print("  [green bold]Config is valid.[/green bold]\n")
    elif report.overall_status == CheckStatus.WARNING:
        console.print("  [yellow]Config is valid with warnings — review above.[/yellow]\n")
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("  [red bold]Config has errors — fix them before running.[/red bold]\n")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
