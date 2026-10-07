"""End-to-end MVP workflow: Validation → Training → Evaluation → Approval → Registry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.agents.dataset_validation import DatasetValidationAgent
from agentic_mlops.agents.evaluation import EvaluationAgent
from agentic_mlops.agents.human_approval import HumanApprovalAgent
from agentic_mlops.agents.model_registry import ModelRegistryAgent
from agentic_mlops.agents.training import TrainingAgent
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalInput, ApprovalStatus
from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.contracts.datasets import DatasetValidationInput
from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMode
from agentic_mlops.contracts.mlflow_config import MLflowConfig
from agentic_mlops.contracts.model_registry import ModelRegistrationInput, RegistryBackend
from agentic_mlops.contracts.training import (
    TrainingConfig,
    TrainingInput,
    TrainingMode,
)
from agentic_mlops.contracts.workflows import (
    MVPWorkflowInput,
    MVPWorkflowOutput,
    MVPWorkflowStatus,
    MVPWorkflowStepResult,
)
from agentic_mlops.integrations.artifact_store import ArtifactStore, NoOpArtifactStore
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.integrations.model_registry import (
    AzureMLModelRegistryClient,
    ModelRegistryClientBase,
)
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.observability.tracing import BaseTracer, NoOpTracer
from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner
from agentic_mlops.tools.pipeline_runner import AzureMLPipelineRunner
from agentic_mlops.tools.report_writer import ReportWriter
from agentic_mlops.tools.training_runner import AzureMLTrainingRunner


class MVPWorkflow:
    """Chains all four MVP agents sequentially; stops on first failure.

    Pass ``mlflow_client`` + ``mlflow_config`` to enable experiment tracking.
    User-injected ``_*_factory`` kwargs are used as-is (for tests); default
    factories automatically inject MLflow when a parent run is active.
    """

    def __init__(
        self,
        *,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_config: MLflowConfig | None = None,
        _validation_factory: Callable[[Path], Any] | None = None,
        _training_factory: Callable[[Path], Any] | None = None,
        _evaluation_factory: Callable[[Path], Any] | None = None,
        _approval_factory: Callable[[Path], Any] | None = None,
        _registry_factory: Callable[[Path], Any] | None = None,
        artifact_store: ArtifactStore | None = None,
        tracer: BaseTracer | None = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        # None = use default factory with MLflow injection; non-None = test override
        self._user_validation_factory = _validation_factory
        self._user_training_factory = _training_factory
        self._user_evaluation_factory = _evaluation_factory
        self._user_approval_factory = _approval_factory
        self._user_registry_factory = _registry_factory
        self._mlflow = mlflow_client
        self._mlflow_config = mlflow_config
        self._report_writer = ReportWriter()
        self._artifact_store: ArtifactStore = artifact_store or NoOpArtifactStore()
        self._tracer: BaseTracer = tracer if tracer is not None else NoOpTracer()

    def _wire_tracer(self, agent: Any) -> Any:
        """Inject the workflow-level tracer into any BaseAgent after construction."""
        agent._tracer = self._tracer
        return agent

    def run(self, inp: MVPWorkflowInput) -> MVPWorkflowOutput:
        output_dir = Path(inp.output_dir)
        steps: list[MVPWorkflowStepResult] = []
        all_artifacts: list[str] = []

        # ── Resolve Azure ML runners/client up front (fail fast, before any step) ──
        try:
            azure_pipeline_runner = self._maybe_azure_pipeline_runner(inp)
            azure_train_runner = self._maybe_azure_train_runner(inp)
            azure_eval_runner = self._maybe_azure_eval_runner(inp)
            azure_registry_client = self._maybe_azure_registry_client(inp)
        except ValueError as exc:
            return self._fail(output_dir, steps, all_artifacts, str(exc), None)

        # ── Create parent MLflow run ──────────────────────────────────────────
        mlflow_run_id: str | None = None
        if self._mlflow and self._mlflow_config and self._mlflow_config.enabled:
            run_name = (
                f"{self._mlflow_config.run_name_prefix}-"
                f"{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
            )
            mlflow_run_id = self._mlflow.start_run(
                self._mlflow_config.experiment_name,
                run_name,
            )
            self._mlflow.log_tags(
                mlflow_run_id,
                {
                    "workflow_name": "mvp_local_yolo",
                    "training_runner": inp.training_runner or "fake",
                    "evaluation_runner": inp.evaluation_runner or "fake",
                },
            )

        # ── Build factories (inject MLflow + Azure ML runners into defaults only) ──
        val_factory = self._user_validation_factory or self._default_val_factory(mlflow_run_id)
        train_factory = self._user_training_factory or self._default_train_factory(
            mlflow_run_id, azure_train_runner
        )
        eval_factory = self._user_evaluation_factory or self._default_eval_factory(
            mlflow_run_id, azure_eval_runner
        )
        approval_factory = self._user_approval_factory or self._default_approval_factory(
            mlflow_run_id
        )
        registry_factory = self._user_registry_factory or self._default_registry_factory(
            mlflow_run_id, azure_registry_client
        )

        # ── Step 1: Dataset Validation ────────────────────────────────────────
        val_dir = output_dir / "validation"
        try:
            val_agent = val_factory(val_dir)
            val_result = self._wire_tracer(val_agent).run_traced(
                DatasetValidationInput(
                    dataset_path=inp.dataset_path,
                    data_yaml_path=inp.data_yaml_path,
                    fail_on_warnings=inp.fail_on_warnings,
                )
            )
            step = MVPWorkflowStepResult(
                step="validation",
                status=val_result.status if val_result.success else "failed",
                success=val_result.success,
                errors=val_result.errors,
                artifacts=val_result.artifacts,
            )
            steps.append(step)
            all_artifacts.extend(val_result.artifacts)
            if not val_result.success:
                return self._fail(
                    output_dir,
                    steps,
                    all_artifacts,
                    "Dataset validation failed.",
                    mlflow_run_id,
                )
        except Exception as exc:
            return self._exception_fail(
                output_dir, steps, all_artifacts, "validation", exc, mlflow_run_id
            )

        # ── Step 2: Training  /  Step 3: Evaluation ──────────────────────────
        # Pipeline mode runs both steps as a single Azure ML PipelineJob.
        train_dir = output_dir / "training"
        eval_dir = output_dir / "evaluation"

        if azure_pipeline_runner is not None:
            try:
                cfg = TrainingConfig.from_yaml(inp.training_config_path)
                cfg.mode = TrainingMode.AZURE_PIPELINE
                training_inp = TrainingInput(
                    dataset_path=inp.dataset_path,
                    data_yaml_path=inp.data_yaml_path,
                    training_config=cfg,
                    dataset_validation_status=val_result.status,
                )
                evaluation_inp = EvaluationInput(
                    dataset_path=inp.dataset_path,
                    data_yaml_path=inp.data_yaml_path,
                    weights_path="pipeline",
                    mode=EvaluationMode.AZURE_PIPELINE_EVAL,
                    training_status=None,
                    promotion_policy_path=inp.evaluation_config_path,
                    evaluation_config_path=inp.evaluation_config_path,
                )
                train_result, eval_result = azure_pipeline_runner.run(
                    training_inp, evaluation_inp, train_dir, eval_dir
                )
            except Exception as exc:
                return self._exception_fail(
                    output_dir, steps, all_artifacts, "pipeline", exc, mlflow_run_id
                )

            for step_name, result in (("training", train_result), ("evaluation", eval_result)):
                step = MVPWorkflowStepResult(
                    step=step_name,
                    status="completed" if result.success else "failed",
                    success=result.success,
                    errors=result.errors,
                    artifacts=result.artifacts,
                )
                steps.append(step)
                all_artifacts.extend(result.artifacts)

            self._artifact_store.upload_directory(train_dir, "training")
            self._artifact_store.upload_directory(eval_dir, "evaluation")
            if not train_result.success:
                return self._fail(
                    output_dir,
                    steps,
                    all_artifacts,
                    "Pipeline training step failed.",
                    mlflow_run_id,
                )
            if not eval_result.success:
                return self._fail(
                    output_dir,
                    steps,
                    all_artifacts,
                    "Pipeline evaluation step failed.",
                    mlflow_run_id,
                )
        else:
            # ── Step 2: Training (single CommandJob) ──────────────────────────
            try:
                cfg = TrainingConfig.from_yaml(inp.training_config_path)
                if inp.training_runner == "local-yolo":
                    cfg.mode = TrainingMode.LOCAL_TRAIN
                elif inp.training_runner == "azure-ml":
                    cfg.mode = TrainingMode.AZURE_TRAIN
                elif inp.dry_run:
                    cfg.mode = TrainingMode.LOCAL_DRY_RUN
                train_agent = train_factory(train_dir)
                train_result = self._wire_tracer(train_agent).run_traced(
                    TrainingInput(
                        dataset_path=inp.dataset_path,
                        data_yaml_path=inp.data_yaml_path,
                        training_config=cfg,
                        dataset_validation_status=val_result.status,
                    )
                )
                step = MVPWorkflowStepResult(
                    step="training",
                    status="completed" if train_result.success else "failed",
                    success=train_result.success,
                    errors=train_result.errors,
                    artifacts=train_result.artifacts,
                )
                steps.append(step)
                all_artifacts.extend(train_result.artifacts)
                self._artifact_store.upload_directory(train_dir, "training")
                if not train_result.success:
                    return self._fail(
                        output_dir, steps, all_artifacts, "Training failed.", mlflow_run_id
                    )
            except Exception as exc:
                return self._exception_fail(
                    output_dir, steps, all_artifacts, "training", exc, mlflow_run_id
                )

            # ── Step 3: Evaluation (single CommandJob) ────────────────────────
            try:
                weights_path = train_result.best_weights_path or "dry_run"
                if inp.evaluation_runner == "local-yolo":
                    mode = EvaluationMode.LOCAL_EVAL
                elif inp.evaluation_runner == "azure-ml":
                    mode = EvaluationMode.AZURE_EVAL
                elif inp.dry_run:
                    mode = EvaluationMode.LOCAL_DRY_RUN
                else:
                    mode = EvaluationMode.LOCAL_EVAL
                eval_agent = eval_factory(eval_dir)
                eval_result = self._wire_tracer(eval_agent).run_traced(
                    EvaluationInput(
                        dataset_path=inp.dataset_path,
                        data_yaml_path=inp.data_yaml_path,
                        weights_path=weights_path,
                        mode=mode,
                        training_status=str(train_result.job_status),
                        promotion_policy_path=inp.evaluation_config_path,
                        evaluation_config_path=inp.evaluation_config_path,
                    )
                )
                step = MVPWorkflowStepResult(
                    step="evaluation",
                    status="completed" if eval_result.success else "failed",
                    success=eval_result.success,
                    errors=eval_result.errors,
                    artifacts=eval_result.artifacts,
                )
                steps.append(step)
                all_artifacts.extend(eval_result.artifacts)
                self._artifact_store.upload_directory(eval_dir, "evaluation")
                if not eval_result.success:
                    return self._fail(
                        output_dir, steps, all_artifacts, "Evaluation failed.", mlflow_run_id
                    )
            except Exception as exc:
                return self._exception_fail(
                    output_dir, steps, all_artifacts, "evaluation", exc, mlflow_run_id
                )

        # ── Step 4: Human Approval ────────────────────────────────────────────
        approval_dir = output_dir / "approval"
        try:
            eval_json_path = eval_dir / "evaluation_report.json"
            approval_agent = approval_factory(approval_dir)
            approval_result = self._wire_tracer(approval_agent).run_traced(
                ApprovalInput(
                    evaluation_output_path=str(eval_json_path),
                    approver=inp.approver,
                    output_dir=str(approval_dir),
                    interactive=inp.interactive_approval,
                    action=inp.approval_action,
                    force=inp.force_approve,
                )
            )
            step = MVPWorkflowStepResult(
                step="approval",
                status=str(approval_result.status),
                success=approval_result.success,
                errors=approval_result.errors,
                artifacts=approval_result.generated_artifacts,
            )
            steps.append(step)
            all_artifacts.extend(approval_result.generated_artifacts)
        except Exception as exc:
            return self._exception_fail(
                output_dir, steps, all_artifacts, "approval", exc, mlflow_run_id
            )

        # ── Step 5: Model Registry (conditional) ─────────────────────────────
        registry_output = None
        should_register = (
            inp.register_approved_model
            and approval_result.status == ApprovalStatus.APPROVED
            and approval_result.action == ApprovalAction.APPROVE_MODEL
        )
        if inp.register_approved_model and not should_register:
            # Approval was not approve_model — skip gracefully
            steps.append(
                MVPWorkflowStepResult(
                    step="model_registry",
                    status="skipped",
                    success=True,
                    errors=[],
                    artifacts=[],
                )
            )
        elif should_register:
            registry_dir = output_dir / "model_registry"
            try:
                registry_agent = registry_factory(registry_dir)
                registry_output = self._wire_tracer(registry_agent).run_traced(
                    ModelRegistrationInput(
                        model_name=inp.model_name,
                        training_output_path=str(output_dir / "training" / "training_output.json"),
                        evaluation_output_path=str(
                            output_dir / "evaluation" / "evaluation_output.json"
                        ),
                        approval_decision_path=str(
                            output_dir / "approval" / "approval_decision.json"
                        ),
                        registry_dir=inp.registry_dir,
                        backend=inp.registry_backend,
                        mlflow_run_id=mlflow_run_id,
                        mlflow_experiment_name=(
                            self._mlflow_config.experiment_name if self._mlflow_config else None
                        ),
                        mlflow_tracking_uri=(
                            self._mlflow_config.tracking_uri if self._mlflow_config else None
                        ),
                    )
                )
                step = MVPWorkflowStepResult(
                    step="model_registry",
                    status=str(registry_output.status),
                    success=registry_output.success,
                    errors=registry_output.errors,
                    artifacts=registry_output.artifacts,
                )
                steps.append(step)
                all_artifacts.extend(registry_output.artifacts)
            except Exception as exc:
                return self._exception_fail(
                    output_dir, steps, all_artifacts, "model_registry", exc, mlflow_run_id
                )

        # ── Summary ───────────────────────────────────────────────────────────
        overall_success = all(s.success for s in steps)
        workflow_status = (
            MVPWorkflowStatus.COMPLETED if overall_success else MVPWorkflowStatus.FAILED
        )
        output = MVPWorkflowOutput(
            success=overall_success,
            message=f"MVP workflow {workflow_status}.",
            workflow_status=workflow_status,
            steps=steps,
            artifacts=list(all_artifacts),
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=(
                self._mlflow_config.experiment_name if self._mlflow_config else None
            ),
            mlflow_tracking_uri=(self._mlflow_config.tracking_uri if self._mlflow_config else None),
            registration_status=(str(registry_output.status) if registry_output else None),
            registered_model_name=(
                registry_output.model_name if registry_output and registry_output.success else None
            ),
            registered_model_version=(
                registry_output.version if registry_output and registry_output.success else None
            ),
            registered_model_path=(
                registry_output.registry_path
                if registry_output and registry_output.success
                else None
            ),
        )
        json_path, md_path = self._report_writer.write_workflow_summary(output, output_dir)
        output.workflow_summary_path = str(json_path)
        output.workflow_summary_md_path = str(md_path)
        output.artifacts += [str(json_path), str(md_path)]

        # Log workflow artifacts and end run
        if self._mlflow and mlflow_run_id:
            self._mlflow.log_tags(mlflow_run_id, {"workflow_status": str(workflow_status)})
            if self._mlflow_config and self._mlflow_config.log_reports:
                for p in (str(json_path), str(md_path)):
                    self._mlflow.log_artifact(mlflow_run_id, p)
            final_status = "FINISHED" if overall_success else "FAILED"
            self._mlflow.end_run(mlflow_run_id, status=final_status)

        return output

    # ── Azure ML resolution ─────────────────────────────────────────────────────
    # Training/evaluation/registry each reuse the same azure_config_path; resolved
    # up front in run() so a missing/invalid config fails before any step executes.

    def _maybe_azure_pipeline_runner(self, inp: MVPWorkflowInput) -> AzureMLPipelineRunner | None:
        if inp.training_runner != "azure-ml-pipeline":
            return None
        if not inp.azure_config_path:
            raise ValueError(
                "azure_config_path is required when training_runner='azure-ml-pipeline'."
            )
        return AzureMLPipelineRunner(AzureMLConfig.from_yaml(inp.azure_config_path))

    def _maybe_azure_train_runner(self, inp: MVPWorkflowInput) -> AzureMLTrainingRunner | None:
        if inp.training_runner != "azure-ml":
            return None
        if not inp.azure_config_path:
            raise ValueError("azure_config_path is required when training_runner='azure-ml'.")
        return AzureMLTrainingRunner(AzureMLConfig.from_yaml(inp.azure_config_path))

    def _maybe_azure_eval_runner(self, inp: MVPWorkflowInput) -> AzureMLEvaluationRunner | None:
        if inp.evaluation_runner != "azure-ml":
            return None
        if not inp.azure_config_path:
            raise ValueError("azure_config_path is required when evaluation_runner='azure-ml'.")
        return AzureMLEvaluationRunner(AzureMLConfig.from_yaml(inp.azure_config_path))

    def _maybe_azure_registry_client(self, inp: MVPWorkflowInput) -> ModelRegistryClientBase | None:
        if inp.registry_backend != RegistryBackend.AZURE_ML:
            return None
        if not inp.azure_config_path:
            raise ValueError("azure_config_path is required when registry_backend='azure_ml'.")
        return AzureMLModelRegistryClient(AzureMLConfig.from_yaml(inp.azure_config_path))

    # ── Default factory builders ──────────────────────────────────────────────

    def _default_val_factory(self, mlflow_run_id: str | None) -> Callable[[Path], Any]:
        if mlflow_run_id and self._mlflow:
            client, rid = self._mlflow, mlflow_run_id
            return lambda d: DatasetValidationAgent(
                artifacts_dir=d, mlflow_client=client, mlflow_run_id=rid
            )
        return lambda d: DatasetValidationAgent(artifacts_dir=d)

    def _default_train_factory(
        self,
        mlflow_run_id: str | None,
        azure_runner: AzureMLTrainingRunner | None = None,
    ) -> Callable[[Path], Any]:
        if mlflow_run_id and self._mlflow:
            client, rid = self._mlflow, mlflow_run_id
            return lambda d: TrainingAgent(
                artifacts_dir=d, mlflow_client=client, mlflow_run_id=rid, azure_runner=azure_runner
            )
        return lambda d: TrainingAgent(artifacts_dir=d, azure_runner=azure_runner)

    def _default_eval_factory(
        self,
        mlflow_run_id: str | None,
        azure_runner: AzureMLEvaluationRunner | None = None,
    ) -> Callable[[Path], Any]:
        if mlflow_run_id and self._mlflow:
            client, rid = self._mlflow, mlflow_run_id
            return lambda d: EvaluationAgent(
                artifacts_dir=d, mlflow_client=client, mlflow_run_id=rid, azure_runner=azure_runner
            )
        return lambda d: EvaluationAgent(artifacts_dir=d, azure_runner=azure_runner)

    def _default_approval_factory(self, mlflow_run_id: str | None) -> Callable[[Path], Any]:
        if mlflow_run_id and self._mlflow:
            client, rid = self._mlflow, mlflow_run_id
            return lambda d: HumanApprovalAgent(
                artifacts_dir=d, mlflow_client=client, mlflow_run_id=rid
            )
        return lambda d: HumanApprovalAgent(artifacts_dir=d)

    def _default_registry_factory(
        self,
        mlflow_run_id: str | None,
        registry_client: ModelRegistryClientBase | None = None,
    ) -> Callable[[Path], Any]:
        if mlflow_run_id and self._mlflow:
            client, rid = self._mlflow, mlflow_run_id
            return lambda d: ModelRegistryAgent(
                artifacts_dir=d,
                registry_client=registry_client,
                mlflow_client=client,
                mlflow_run_id=rid,
            )
        return lambda d: ModelRegistryAgent(artifacts_dir=d, registry_client=registry_client)

    # ── Failure helpers ───────────────────────────────────────────────────────

    def _fail(
        self,
        output_dir: Path,
        steps: list[MVPWorkflowStepResult],
        artifacts: list[str],
        msg: str,
        mlflow_run_id: str | None = None,
    ) -> MVPWorkflowOutput:
        output = MVPWorkflowOutput(
            success=False,
            message=msg,
            workflow_status=MVPWorkflowStatus.FAILED,
            steps=steps,
            errors=[msg],
            artifacts=list(artifacts),
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=(
                self._mlflow_config.experiment_name if self._mlflow_config else None
            ),
            mlflow_tracking_uri=(self._mlflow_config.tracking_uri if self._mlflow_config else None),
        )
        json_path, md_path = self._report_writer.write_workflow_summary(output, output_dir)
        output.workflow_summary_path = str(json_path)
        output.workflow_summary_md_path = str(md_path)
        if self._mlflow and mlflow_run_id:
            self._mlflow.log_tags(mlflow_run_id, {"workflow_status": "failed"})
            self._mlflow.end_run(mlflow_run_id, status="FAILED")
        return output

    def _exception_fail(
        self,
        output_dir: Path,
        steps: list[MVPWorkflowStepResult],
        artifacts: list[str],
        step_name: str,
        exc: Exception,
        mlflow_run_id: str | None = None,
    ) -> MVPWorkflowOutput:
        msg = f"{step_name} raised an exception: {exc}"
        self.logger.exception(msg)
        err_step = MVPWorkflowStepResult(
            step=step_name,
            status="failed",
            success=False,
            errors=[msg],
        )
        steps.append(err_step)
        return self._fail(output_dir, steps, artifacts, msg, mlflow_run_id)
