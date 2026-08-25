"""Data-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import (
    _print_annotation_result,
    _print_data_intake_result,
    _print_dataset_structuring_result,
    _print_dataset_versioning_result,
    _print_label_qa_result,
    _print_validation_result,
    console,
)


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
