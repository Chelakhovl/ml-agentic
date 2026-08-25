"""Training-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import (
    _make_artifact_store,
    _print_approval_result,
    _print_evaluation_result,
    _print_mvp_summary,
    _print_registry_result,
    _print_training_approval_result,
    _print_training_result,
    _resolve_mlflow,
    console,
)


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
