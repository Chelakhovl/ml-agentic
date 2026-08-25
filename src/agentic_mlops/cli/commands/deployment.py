"""Deployment-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import (
    _print_deployment_result,
    _print_monitoring_result,
    console,
)


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
    canary_percentage: int = typer.Option(
        100,
        "--canary-percentage",
        min=1,
        max=100,
        help=(
            "Percentage of traffic to route to the new deployment (1–100). "
            "Only meaningful for --backend azure_ml (routes partial traffic to new deployment). "
            "For local/docker/aks, recorded in the deployment manifest as metadata."
        ),
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
            canary_percentage=canary_percentage,
        )
    )

    _print_deployment_result(result)

    if not result.success:
        raise typer.Exit(code=1)


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
