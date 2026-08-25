"""Orchestrator — routes between all standalone agents through the full pipeline.

Per agentic_mlops_workflow_docs/agents/00_orchestrator_agent.md: "не выполняет
ML-логику сам, а управляет состояниями, вызывает нужных агентов, проверяет
policies и создает human approval gates." Concretely, this class:

  - runs an ordered, configurable subset of PIPELINE_STEPS (contracts/orchestrator.py)
  - persists state.json + audit_log.jsonl per workflow_id via WorkflowStateStore
  - checks that state transitions are legal (the spec's "Policy Engine", scoped
    to "is this transition allowed" rather than a general rule DSL)
  - stops (without raising) at the two genuinely blocking human gates that
    aren't already handled inside an individual agent: H4 Training Approval
    and H5 Model Approval. H6 Production Release Approval is already
    enforced inside DeploymentAgent itself; the Orchestrator just surfaces a
    `blocked` step result for it.
  - propagates a rejection at either gate forward as a cascade of graceful
    `SKIPPED` steps (training/evaluation/model_decision/approval/
    model_registry/deployment, as applicable) rather than a failure — a
    human saying "no" is a valid business outcome, not a system error.
  - supports --resume: a workflow_id with a saved state.json skips steps
    already recorded as completed and picks up where it left off
  - optionally tracks the whole run as one parent MLflow run (mlflow_client=
    + mlflow_config=, same pattern as MVPWorkflow): every agent invoked logs
    under it (each agent already supports mlflow_client=/mlflow_run_id=
    injection — nothing new needed there), the run_id is persisted in
    state.json so a --resume after a pause/failure logs into the SAME run
    rather than starting a new one, and the run ends only on a terminal
    status (COMPLETED/FAILED/BLOCKED) — a PENDING_APPROVAL pause leaves it
    open. Disabled by default; enable via --mlflow-config/--enable-mlflow.
  - optionally sends Teams/Slack webhook notifications (notification_client=
    + OrchestratorInput.notifications: NotificationConfig) on key pipeline
    events (step_failed, workflow_completed, workflow_failed, etc.). Same
    injection pattern as mlflow_client. Disabled by default (NoOpNotificationClient).

Deliberately NOT a BaseAgent subclass, matching MVPWorkflow's own placement in
workflows/ rather than agents/ — this chains other agents, it doesn't wrap one
tool.

Human-in-the-loop gate mapping (spec's H1-H6):
    H1 Data Source Approval      -> DataIntakeAgent's `needs_human_source_approval`
                                     status (soft — surfaced, does not block)
    H2 Dataset Cleanup Approval  -> DatasetValidationAgent `failed` status (hard stop)
    H3 Label Review              -> LabelQAAgent `review_required` (standalone,
                                     not part of this forward chain — see module docstring)
    H4 Training Approval         -> TrainingApprovalAgent (`training_approval` step, real,
                                     blocking gate — reads dataset_quality_report.json,
                                     sits between dataset_versioning and training)
    H5 Model Approval            -> HumanApprovalAgent (real, blocking gate)
    H6 Production Release        -> DeploymentAgent's production gate (real, blocking gate)
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.agents.data_intake import DataIntakeAgent
from agentic_mlops.agents.dataset_structuring import DatasetStructuringAgent
from agentic_mlops.agents.dataset_validation import DatasetValidationAgent
from agentic_mlops.agents.dataset_versioning import DatasetVersioningAgent
from agentic_mlops.agents.deployment import DeploymentAgent
from agentic_mlops.agents.evaluation import EvaluationAgent
from agentic_mlops.agents.human_approval import HumanApprovalAgent
from agentic_mlops.agents.model_decision import ModelDecisionAgent
from agentic_mlops.agents.model_registry import ModelRegistryAgent
from agentic_mlops.agents.training import TrainingAgent
from agentic_mlops.agents.training_approval import TrainingApprovalAgent
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalInput
from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.contracts.data_intake import DataIntakeInput
from agentic_mlops.contracts.dataset_structuring import DatasetStructuringInput
from agentic_mlops.contracts.dataset_versioning import (
    DatasetRegistryBackend,
    DatasetVersioningInput,
)
from agentic_mlops.contracts.datasets import DatasetValidationInput
from agentic_mlops.contracts.deployment import DeploymentBackend, DeploymentInput
from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMode
from agentic_mlops.contracts.mlflow_config import MLflowConfig
from agentic_mlops.contracts.model_decision import ModelDecisionInput
from agentic_mlops.contracts.model_registry import ModelRegistrationInput, RegistryBackend
from agentic_mlops.contracts.orchestrator import (
    DEFAULT_STEPS,
    PIPELINE_STEPS,
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorStatus,
    OrchestratorStepResult,
)
from agentic_mlops.contracts.training import TrainingConfig, TrainingInput, TrainingMode
from agentic_mlops.contracts.training_approval import (
    TrainingApprovalAction,
    TrainingApprovalInput,
)
from agentic_mlops.integrations.artifact_store import ArtifactStore, NoOpArtifactStore
from agentic_mlops.integrations.azure_ml_online_endpoint import AzureMLOnlineEndpointDeployer
from agentic_mlops.integrations.dataset_registry import AzureMLDatasetRegistryClient
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.integrations.model_registry import (
    AzureMLModelRegistryClient,
    ModelRegistryClientBase,
)
from agentic_mlops.integrations.notification_client import NotificationClientBase
from agentic_mlops.integrations.workflow_state_store import WorkflowStateStore
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.tools.baseline_comparator import BaselineComparator
from agentic_mlops.tools.baseline_resolver import BaselineResolver
from agentic_mlops.tools.cost_tracker import CostTracker
from agentic_mlops.tools.deployer import ModelDeployer
from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner
from agentic_mlops.tools.pipeline_runner import AzureMLPipelineRunner
from agentic_mlops.tools.report_writer import ReportWriter
from agentic_mlops.tools.risk_scorer import ModelRiskScorer
from agentic_mlops.tools.training_runner import AzureMLTrainingRunner

# Steps that can pause the workflow (pending_approval=True) instead of failing,
# and the fine-grained current_state label / pending_approval_id prefix each uses.
_PENDING_LABELS: dict[str, str] = {
    "training_approval": "TRAINING_APPROVAL_REQUIRED",
    "approval": "MODEL_APPROVAL_REQUIRED",
}
_PENDING_ID_PREFIX: dict[str, str] = {
    "training_approval": "appr_train",
    "approval": "appr",
}


@dataclass
class _StepOutcome:
    success: bool
    status_label: str
    errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    key_outputs: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    pending_approval: bool = False
    # Overrides the coarse OrchestratorStatus mapping on failure, e.g. "blocked"
    # for a gate-blocked step (dataset_versioning / deployment) vs the default "failed".
    coarse: str | None = None


def _skip(reason: str) -> _StepOutcome:
    """A step that didn't run because an upstream gate rejected/skipped —
    a graceful, cascading no-op, not a failure."""
    return _StepOutcome(
        success=True,
        status_label="SKIPPED",
        skipped=True,
        key_outputs={"skipped": True, "skip_reason": reason},
    )


class OrchestratorWorkflow:
    """Chains a configurable subset of PIPELINE_STEPS with state + audit persistence."""

    def __init__(
        self,
        *,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_config: MLflowConfig | None = None,
        artifact_store: ArtifactStore | None = None,
        notification_client: NotificationClientBase | None = None,
        github_actions_client: Any | None = None,
        cost_tracker: CostTracker | None = None,
        baseline_comparator: BaselineComparator | None = None,
        risk_scorer: ModelRiskScorer | None = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_config = mlflow_config
        self._artifact_store: ArtifactStore = artifact_store or NoOpArtifactStore()
        self._notification_client = notification_client
        self._github_actions_client = github_actions_client
        self._cost_tracker = cost_tracker
        self._baseline_comparator = baseline_comparator
        self._risk_scorer = risk_scorer
        # Set fresh at the top of every run() call; read by _mlflow_kwargs()
        # so every _step_* method can inject tracking without threading an
        # extra parameter through all 11 of them.
        self._current_mlflow_run_id: str | None = None
        # Resolved at the top of run() from injected client or inp.notifications.
        self._notifier: NotificationClientBase | None = None
        # Resolved at the top of run() from injected client or inp.github_actions.
        self._gh_client: Any | None = None

    def _mlflow_kwargs(self) -> dict[str, Any]:
        if not self._current_mlflow_run_id:
            return {}
        return {"mlflow_client": self._mlflow, "mlflow_run_id": self._current_mlflow_run_id}

    def _notify(self, event_name: str, payload: dict) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.send(event_name, payload)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Notification send failed: %s", exc)

    def _record_step_cost(
        self,
        step: str,
        inp: OrchestratorInput,
        azure_config: AzureMLConfig | None,
        outcome: _StepOutcome,
    ) -> None:
        if self._cost_tracker is None:
            return
        runner = (
            inp.training_runner
            if step == "training"
            else (inp.evaluation_runner or "fake")
            if step == "evaluation"
            else "fake"
        )
        # Use cluster name for Azure ML so users can map it in their pricing config.
        if runner in ("azure-ml", "azure-ml-pipeline") and azure_config is not None:
            compute_type = azure_config.compute_name
        elif runner == "local-yolo":
            compute_type = "local"
        else:
            compute_type = runner or "fake"
        self._cost_tracker.record(
            step=step,
            runner=runner or "fake",
            compute_type=compute_type,
            started_at=outcome.key_outputs.get("started_at"),
            completed_at=outcome.key_outputs.get("completed_at"),
        )

    def _run_baseline_comparison_if_configured(
        self,
        inp: OrchestratorInput,
        output_root: Path,
        step_outputs: dict,
        all_artifacts: list[str],
    ) -> None:
        """After a successful evaluation step, compare against baseline if configured.

        Priority:
        1. ``inp.baseline_report_path`` — explicit path, always takes precedence.
        2. ``inp.auto_baseline=True`` — auto-resolve the best registered model from
           ``inp.registry_dir``, write a synthetic eval report, and use that.
        3. Neither set — skip comparison.
        """
        baseline_path: str | None = inp.baseline_report_path

        if not baseline_path and inp.auto_baseline:
            resolver = BaselineResolver()
            resolved = resolver.resolve(
                inp.registry_dir,
                exclude_model_name=inp.model_name or None,
            )
            if resolved is not None:
                synthetic_path = resolver.write_synthetic_report(
                    resolved,
                    output_dir=output_root / "evaluation",
                )
                baseline_path = str(synthetic_path)
                self.logger.info(
                    "Auto-resolved baseline from registry",
                    extra={
                        "baseline_model": resolved.model_name,
                        "baseline_version": resolved.version,
                        "baseline_map50": resolved.map50,
                    },
                )
            else:
                self.logger.info(
                    "auto_baseline=True but no registered model found; skipping comparison"
                )

        if not baseline_path:
            return

        comparator = self._baseline_comparator or BaselineComparator()
        eval_report_path = (step_outputs.get("evaluation") or {}).get("report_path")
        if not eval_report_path:
            self.logger.warning(
                "baseline configured but no evaluation report_path found; skipping comparison"
            )
            return
        try:
            result = comparator.compare(
                eval_report_path=eval_report_path,
                baseline_path=baseline_path,
                output_dir=output_root / "evaluation",
            )
            comparison_path = str(output_root / "evaluation" / "baseline_comparison.json")
            all_artifacts.append(comparison_path)
            eval_out = step_outputs.get("evaluation") or {}
            eval_out["baseline_comparison_path"] = comparison_path
            eval_out["baseline_improved"] = result.improved
            eval_out["baseline_delta_map50"] = result.delta_map50
            step_outputs["evaluation"] = eval_out
            self.logger.info(
                "Baseline comparison complete",
                extra={
                    "delta_map50": result.delta_map50,
                    "improved": result.improved,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Baseline comparison failed (non-fatal)", extra={"error": str(exc)})

    def _run_risk_scoring_if_configured(
        self,
        output_root: Path,
        step_outputs: dict,
        all_artifacts: list[str],
    ) -> None:
        """After a successful model_registry step, compute a risk report."""
        scorer = self._risk_scorer or ModelRiskScorer()
        eval_out = step_outputs.get("evaluation") or {}
        eval_report_path = eval_out.get("report_path")
        if not eval_report_path:
            self.logger.warning(
                "risk_scorer: no evaluation report_path in step_outputs; skipping risk scoring"
            )
            return
        baseline_comparison_path = eval_out.get("baseline_comparison_path")
        try:
            scorer.score(
                eval_report_path=eval_report_path,
                baseline_comparison_path=baseline_comparison_path,
                output_dir=output_root / "model_registry",
            )
            risk_path = str(output_root / "model_registry" / "risk_report.json")
            all_artifacts.append(risk_path)
            reg_out = step_outputs.get("model_registry") or {}
            reg_out["risk_report_path"] = risk_path
            step_outputs["model_registry"] = reg_out
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Risk scoring failed (non-fatal)", extra={"error": str(exc)})

    def _resolve_mlflow_run(self, state: dict) -> str | None:
        existing = state.get("mlflow_run_id")
        if existing:
            return existing
        if not (self._mlflow and self._mlflow_config and self._mlflow_config.enabled):
            return None
        run_name = f"{self._mlflow_config.run_name_prefix}-{state['workflow_id']}"
        run_id = self._mlflow.start_run(self._mlflow_config.experiment_name, run_name)
        self._mlflow.log_tags(
            run_id,
            {
                "workflow_name": "orchestrator",
                "workflow_id": state["workflow_id"],
                "steps": ",".join(state.get("steps", [])),
            },
        )
        state["mlflow_run_id"] = run_id
        return run_id

    def run(self, inp: OrchestratorInput) -> OrchestratorOutput:
        store = WorkflowStateStore(Path(inp.runs_dir), inp.workflow_id)
        output_root = Path(inp.output_dir) if inp.output_dir else store.workflow_dir / "artifacts"

        try:
            steps = self._resolve_steps(inp.steps)
        except ValueError as exc:
            return self._immediate_fail(store, inp, str(exc))

        existing_state = store.load_state()
        effective_resume = inp.resume or bool(inp.resume_from_step)

        # resume_from_step requires an existing state to rewind into.
        # Plain resume=True with no state is allowed (starts fresh — idempotent).
        if inp.resume_from_step and existing_state is None:
            return self._immediate_fail(
                store,
                inp,
                f"--resume-from-step requires an existing state file for workflow_id "
                f"'{inp.workflow_id}' ({store.state_path}). Run the workflow first.",
            )
        if existing_state is not None and not effective_resume:
            return self._immediate_fail(
                store,
                inp,
                f"A state file already exists for workflow_id '{inp.workflow_id}' "
                f"({store.state_path}). Pass resume=True to continue it, or use a "
                "different workflow_id.",
            )

        if existing_state is not None and effective_resume:
            state = existing_state

            if inp.resume_from_step:
                if inp.resume_from_step not in steps:
                    return self._immediate_fail(
                        store,
                        inp,
                        f"resume_from_step '{inp.resume_from_step}' is not in the configured "
                        f"steps: {list(steps)}",
                    )
                target_idx = steps.index(inp.resume_from_step)
                steps_to_keep = set(steps[:target_idx])

                # Trim completed_steps: only keep those before the target.
                state["completed_steps"] = [
                    s for s in state.get("completed_steps", []) if s in steps_to_keep
                ]
                # Clear outputs/status for the target step and all after it.
                for s in steps[target_idx:]:
                    state.get("step_outputs", {}).pop(s, None)
                    state.get("step_status", {}).pop(s, None)

                # Reset current_state so the first re-run step's _RUNNING transition is legal.
                if target_idx == 0:
                    state["current_state"] = "NEW"
                else:
                    state["current_state"] = f"{steps[target_idx - 1].upper()}_COMPLETED"

                state["status"] = OrchestratorStatus.RUNNING.value
                state["pending_approval_id"] = None
                store.save_state(state)
                store.append_audit(
                    {
                        "event": "workflow_resume_from_step",
                        "step": inp.resume_from_step,
                        "steps_kept": list(steps_to_keep),
                        "workflow_id": inp.workflow_id,
                    }
                )

            step_results = [
                OrchestratorStepResult(
                    step=s,
                    status=state.get("step_status", {}).get(s, "completed"),
                    success=True,
                    artifacts=state.get("step_outputs", {}).get(s, {}).get("artifacts", []),
                )
                for s in state.get("completed_steps", [])
            ]
            if state.get("status") == OrchestratorStatus.COMPLETED.value:
                store.append_audit({"event": "workflow_already_completed"})
                return self._finish(
                    store,
                    state,
                    step_results,
                    state.get("artifacts", []),
                    OrchestratorStatus.COMPLETED,
                    "Workflow already completed.",
                )
            store.append_audit({"event": "workflow_resumed", "workflow_id": inp.workflow_id})
        else:
            state = {
                "workflow_id": inp.workflow_id,
                "trigger": inp.trigger,
                "steps": list(steps),
                "completed_steps": [],
                "step_status": {},
                "step_outputs": {},
                "artifacts": [],
                "current_state": "NEW",
                "status": OrchestratorStatus.RUNNING.value,
                "last_agent": None,
                "pending_approval_id": None,
                "started_at": datetime.now(tz=UTC).isoformat(),
            }
            store.save_state(state)
            # Persist original input so the web UI can reconstruct it for --resume.
            (store.workflow_dir / "input.json").write_text(
                inp.model_dump_json(indent=2), encoding="utf-8"
            )
            store.append_audit(
                {"event": "workflow_started", "workflow_id": inp.workflow_id, "steps": list(steps)}
            )
            step_results = []

        completed: list[str] = list(state.get("completed_steps", []))
        step_outputs: dict[str, dict] = state.get("step_outputs", {})
        step_status: dict[str, str] = state.get("step_status", {})
        all_artifacts: list[str] = list(state.get("artifacts", []))
        self._current_mlflow_run_id = self._resolve_mlflow_run(state)

        if self._notification_client is not None:
            self._notifier = self._notification_client
        elif inp.notifications and inp.notifications.enabled:
            from agentic_mlops.integrations.notification_client import (  # noqa: PLC0415
                CompositeNotificationClient,
                SmtpEmailNotificationClient,
                WebhookNotificationClient,
            )

            n = inp.notifications
            _clients = []
            if n.teams_webhook_url or n.slack_webhook_url:
                _clients.append(WebhookNotificationClient(n))
            if n.smtp_host and n.email_to:
                _clients.append(SmtpEmailNotificationClient(n))
            self._notifier = (
                _clients[0]
                if len(_clients) == 1
                else CompositeNotificationClient(_clients)
                if _clients
                else None
            )
        else:
            self._notifier = None

        if self._github_actions_client is not None:
            self._gh_client = self._github_actions_client
        elif inp.github_actions and inp.github_actions.enabled:
            from agentic_mlops.integrations.github_actions_client import (  # noqa: PLC0415
                GithubActionsClient,
            )

            self._gh_client = GithubActionsClient(inp.github_actions)
        else:
            self._gh_client = None

        azure_config: AzureMLConfig | None = None
        if inp.azure_config_path:
            try:
                azure_config = AzureMLConfig.from_yaml(inp.azure_config_path)
            except Exception as exc:
                return self._finish(
                    store,
                    state,
                    step_results,
                    all_artifacts,
                    OrchestratorStatus.FAILED,
                    f"Cannot load azure_config_path: {exc}",
                    errors=[str(exc)],
                )

        for step in steps:
            if step in completed:
                continue

            self._transition(state, f"{step.upper()}_RUNNING", steps)
            state["last_agent"] = step
            state["status"] = OrchestratorStatus.RUNNING.value
            store.save_state(state)
            store.append_audit({"event": "step_started", "step": step})
            self._notify("step_started", {"workflow_id": inp.workflow_id, "step": step})

            try:
                outcome = self._dispatch(step, inp, output_root, step_outputs, azure_config)
            except Exception as exc:
                tb = traceback.format_exc(limit=20)
                store.append_audit(
                    {"event": "step_exception", "step": step, "error": str(exc), "traceback": tb}
                )
                self._notify(
                    "step_failed",
                    {"workflow_id": inp.workflow_id, "step": step, "error": str(exc)},
                )
                self._transition(state, f"{step.upper()}_FAILED", steps)
                return self._finish(
                    store,
                    state,
                    step_results,
                    all_artifacts,
                    OrchestratorStatus.FAILED,
                    f"{step} raised an exception: {exc}",
                    errors=[str(exc)],
                )

            step_results.append(
                OrchestratorStepResult(
                    step=step,
                    status=outcome.status_label,
                    success=outcome.success,
                    errors=outcome.errors,
                    artifacts=outcome.artifacts,
                )
            )
            all_artifacts.extend(outcome.artifacts)
            step_outputs[step] = outcome.key_outputs
            step_status[step] = outcome.status_label
            state["step_outputs"] = step_outputs
            state["step_status"] = step_status
            state["artifacts"] = all_artifacts
            store.append_audit(
                {
                    "event": "step_finished",
                    "step": step,
                    "success": outcome.success,
                    "status": outcome.status_label,
                }
            )

            # Best-effort blob upload — happens regardless of outcome so partial
            # artifacts are available for debugging even on failure.
            step_dir = output_root / step
            self._artifact_store.upload_directory(step_dir, f"{inp.workflow_id}/{step}")
            self._artifact_store.upload_file(
                store.workflow_dir / "state.json", f"{inp.workflow_id}/state.json"
            )
            self._artifact_store.upload_file(
                store.workflow_dir / "audit_log.jsonl", f"{inp.workflow_id}/audit_log.jsonl"
            )

            if not outcome.success:
                terminal = (
                    OrchestratorStatus.BLOCKED
                    if outcome.coarse == "blocked"
                    else OrchestratorStatus.FAILED
                )
                suffix = "BLOCKED" if terminal == OrchestratorStatus.BLOCKED else "FAILED"
                self._notify(
                    "step_failed",
                    {"workflow_id": inp.workflow_id, "step": step, "errors": outcome.errors},
                )
                self._transition(state, f"{step.upper()}_{suffix}", steps)
                return self._finish(
                    store,
                    state,
                    step_results,
                    all_artifacts,
                    terminal,
                    f"{step} did not succeed.",
                    errors=outcome.errors,
                )

            if outcome.pending_approval:
                label = _PENDING_LABELS[step]
                self._transition(state, label, steps)
                pending_id = f"{_PENDING_ID_PREFIX[step]}_{inp.workflow_id}"
                state["pending_approval_id"] = pending_id
                return self._finish(
                    store,
                    state,
                    step_results,
                    all_artifacts,
                    OrchestratorStatus.PENDING_APPROVAL,
                    "Workflow paused: awaiting a human approval decision. Re-run with "
                    "resume=True and an action set (or interactive mode) to continue.",
                    pending_approval_id=pending_id,
                )

            completed.append(step)
            state["completed_steps"] = completed
            self._transition(state, f"{step.upper()}_COMPLETED", steps)
            if step == "evaluation":
                self._run_baseline_comparison_if_configured(
                    inp, output_root, step_outputs, all_artifacts
                )
            if step == "model_registry":
                self._run_risk_scoring_if_configured(output_root, step_outputs, all_artifacts)
            self._record_step_cost(step, inp, azure_config, outcome)
            store.save_state(state)
            self._notify(
                "step_completed",
                {
                    "workflow_id": inp.workflow_id,
                    "step": step,
                    "status": outcome.status_label,
                },
            )

        self._transition(state, "COMPLETED", steps)
        return self._finish(
            store,
            state,
            step_results,
            all_artifacts,
            OrchestratorStatus.COMPLETED,
            "Workflow completed.",
        )

    # ── Step resolution ──────────────────────────────────────────────────────

    def _resolve_steps(self, steps: list[str] | None) -> list[str]:
        chosen = list(steps) if steps else list(DEFAULT_STEPS)
        unknown = [s for s in chosen if s not in PIPELINE_STEPS]
        if unknown:
            raise ValueError(f"Unknown step(s): {unknown}. Valid steps: {list(PIPELINE_STEPS)}")
        # Enforce canonical relative order regardless of caller-supplied order.
        return [s for s in PIPELINE_STEPS if s in chosen]

    def _dispatch(
        self,
        step: str,
        inp: OrchestratorInput,
        output_root: Path,
        step_outputs: dict[str, dict],
        azure_config: AzureMLConfig | None,
    ) -> _StepOutcome:
        handlers: dict[str, Callable[..., _StepOutcome]] = {
            "data_intake": self._step_data_intake,
            "dataset_structuring": self._step_dataset_structuring,
            "dataset_validation": self._step_dataset_validation,
            "dataset_versioning": self._step_dataset_versioning,
            "training_approval": self._step_training_approval,
            "training": self._step_training,
            "evaluation": self._step_evaluation,
            "model_decision": self._step_model_decision,
            "approval": self._step_approval,
            "model_registry": self._step_model_registry,
            "deployment": self._step_deployment,
        }
        return handlers[step](inp, output_root, step_outputs, azure_config)

    def _effective_dataset(
        self, step_outputs: dict[str, dict], inp: OrchestratorInput
    ) -> tuple[str, str]:
        structuring = step_outputs.get("dataset_structuring")
        if structuring and structuring.get("dataset_path"):
            return structuring["dataset_path"], structuring["data_yaml_path"]
        if not inp.dataset_path or not inp.data_yaml_path:
            raise ValueError(
                "dataset_path and data_yaml_path are required when dataset_structuring "
                "is not included in `steps`."
            )
        return inp.dataset_path, inp.data_yaml_path

    # ── Step implementations ─────────────────────────────────────────────────

    def _step_data_intake(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        if not inp.raw_data_path:
            return _StepOutcome(
                False, "FAILED", errors=["raw_data_path is required for the data_intake step."]
            )
        agent = DataIntakeAgent(artifacts_dir=output_root / "data_intake", **self._mlflow_kwargs())
        result = agent.run(
            DataIntakeInput(
                raw_data_path=inp.raw_data_path,
                dataset_name=inp.dataset_name,
                source=inp.data_source,
            )
        )
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper(),
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={"status": str(result.status), "manifest_path": result.manifest_path},
        )

    def _step_dataset_structuring(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        if not inp.raw_data_path:
            return _StepOutcome(
                False,
                "FAILED",
                errors=["raw_data_path is required for the dataset_structuring step."],
            )
        if not inp.classes:
            return _StepOutcome(
                False, "FAILED", errors=["classes is required for the dataset_structuring step."]
            )
        step_dir = output_root / "dataset_structuring"
        agent = DatasetStructuringAgent(artifacts_dir=step_dir, **self._mlflow_kwargs())
        result = agent.run(
            DatasetStructuringInput(
                raw_data_path=inp.raw_data_path,
                output_dataset_path=str(step_dir / "structured"),
                classes=inp.classes,
                label_format=inp.label_format,
                coco_annotations_path=inp.coco_annotations_path,
                split_strategy=inp.split_strategy,
                train_ratio=inp.train_ratio,
                val_ratio=inp.val_ratio,
                test_ratio=inp.test_ratio,
                group_by_regex=inp.group_by_regex,
            )
        )
        return _StepOutcome(
            success=result.success,
            status_label="COMPLETED" if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "dataset_path": result.structured_dataset_path,
                "data_yaml_path": result.data_yaml_path,
                "split_report_path": result.split_report_path,
            },
        )

    def _step_dataset_validation(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        dataset_path, data_yaml_path = self._effective_dataset(step_outputs, inp)
        agent = DatasetValidationAgent(
            artifacts_dir=output_root / "dataset_validation", **self._mlflow_kwargs()
        )
        result = agent.run(
            DatasetValidationInput(
                dataset_path=dataset_path,
                data_yaml_path=data_yaml_path,
                fail_on_warnings=inp.fail_on_warnings,
            )
        )
        return _StepOutcome(
            success=result.success,
            status_label=result.status.upper(),
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={"status": result.status, "report_path": result.report_path},
        )

    def _step_dataset_versioning(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        dataset_path, _ = self._effective_dataset(step_outputs, inp)
        validation = step_outputs.get("dataset_validation")

        registry_client = None
        if inp.dataset_registry_backend == DatasetRegistryBackend.AZURE_ML:
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=[
                        "azure_config_path is required when " "dataset_registry_backend='azure_ml'."
                    ],
                )
            registry_client = AzureMLDatasetRegistryClient(azure_config)

        agent = DatasetVersioningAgent(
            artifacts_dir=output_root / "dataset_versioning",
            registry_client=registry_client,
            **self._mlflow_kwargs(),
        )
        result = agent.run(
            DatasetVersioningInput(
                dataset_path=dataset_path,
                dataset_name=inp.dataset_name,
                registry_dir=inp.dataset_registry_dir,
                parent_version=inp.parent_version,
                workflow_id=inp.workflow_id,
                validation_report_path=validation.get("report_path") if validation else None,
                approved_by=inp.approved_by,
                source_batches=inp.source_batches,
                backend=inp.dataset_registry_backend,
            )
        )
        coarse = "blocked" if str(result.status) == "blocked" else None
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper(),
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "status": str(result.status),
                "version": result.version,
                "dataset_version_path": result.dataset_version_path,
            },
            coarse=coarse,
        )

    def _step_training_approval(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        validation = step_outputs.get("dataset_validation")
        if not validation:
            return _StepOutcome(
                False,
                "FAILED",
                errors=[
                    "training_approval requires the dataset_validation step to have run first."
                ],
            )
        if not inp.interactive_training_approval and inp.training_approval_action is None:
            # H4 Training Approval gate: a legitimate pause, not an error — see module
            # docstring. Don't even call TrainingApprovalAgent; it would just fail with
            # "Non-interactive mode requires --action".
            return _StepOutcome(
                success=True,
                status_label="PENDING",
                pending_approval=True,
                key_outputs={"approved": False},
            )

        step_dir = output_root / "training_approval"
        agent = TrainingApprovalAgent(artifacts_dir=step_dir, **self._mlflow_kwargs())
        result = agent.run(
            TrainingApprovalInput(
                dataset_report_path=validation["report_path"],
                approver=inp.training_approver,
                output_dir=str(step_dir),
                interactive=inp.interactive_training_approval,
                action=inp.training_approval_action,
                force=inp.force_training_approval,
            )
        )
        approved = result.success and result.action == TrainingApprovalAction.APPROVE_TRAINING
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper() if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.generated_artifacts,
            key_outputs={
                "approved": approved,
                "status": str(result.status),
                "action": str(result.action) if result.action else None,
            },
        )

    def _step_training(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        training_approval = step_outputs.get("training_approval")
        if training_approval is not None and not training_approval.get("approved"):
            return _skip("training_approval (H4) was not approve_training")
        if not inp.training_config_path:
            return _StepOutcome(
                False, "FAILED", errors=["training_config_path is required for the training step."]
            )
        dataset_path, data_yaml_path = self._effective_dataset(step_outputs, inp)
        cfg = TrainingConfig.from_yaml(inp.training_config_path)
        if inp.training_runner == "local-yolo":
            cfg.mode = TrainingMode.LOCAL_TRAIN
        elif inp.training_runner == "azure-ml":
            cfg.mode = TrainingMode.AZURE_TRAIN
        elif inp.dry_run:
            cfg.mode = TrainingMode.LOCAL_DRY_RUN

        azure_runner = None
        if inp.training_runner == "azure-ml":
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=["azure_config_path is required when training_runner='azure-ml'."],
                )
            azure_runner = AzureMLTrainingRunner(azure_config)

        if inp.training_runner == "azure-ml-pipeline":
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=[
                        "azure_config_path is required when training_runner='azure-ml-pipeline'."
                    ],
                )
            return self._step_training_pipeline(inp, output_root, step_outputs, azure_config, cfg)

        validation = step_outputs.get("dataset_validation")
        step_dir = output_root / "training"
        agent = TrainingAgent(
            artifacts_dir=step_dir, azure_runner=azure_runner, **self._mlflow_kwargs()
        )
        result = agent.run(
            TrainingInput(
                dataset_path=dataset_path,
                data_yaml_path=data_yaml_path,
                training_config=cfg,
                dataset_validation_status=validation.get("status") if validation else None,
            )
        )
        # LOCAL_DRY_RUN doesn't write training_output.json; write it here so
        # downstream steps (model_registry Gate 4) can read it regardless of mode.
        output_json_path = step_dir / "training_output.json"
        if not output_json_path.exists():
            import json as _json  # noqa: PLC0415

            output_json_path.write_text(
                _json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
        return _StepOutcome(
            success=result.success,
            status_label="COMPLETED" if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "best_weights_path": result.best_weights_path,
                "job_status": str(result.job_status),
                "training_output_path": str(output_json_path),
                "started_at": result.remote_started_at,
                "completed_at": result.remote_completed_at,
            },
        )

    def _step_training_pipeline(
        self,
        inp: OrchestratorInput,
        output_root: Path,
        step_outputs: dict,
        azure_config: AzureMLConfig,
        cfg: TrainingConfig,
    ) -> _StepOutcome:
        """Submit a 2-step train+eval PipelineJob and store both outputs."""
        from agentic_mlops.contracts.evaluation import (
            EvaluationInput as _EvalInput,  # noqa: PLC0415
        )

        dataset_path, data_yaml_path = self._effective_dataset(step_outputs, inp)
        validation = step_outputs.get("dataset_validation")
        cfg.mode = TrainingMode.AZURE_PIPELINE

        train_step_dir = output_root / "training"
        eval_step_dir = output_root / "evaluation"
        train_step_dir.mkdir(parents=True, exist_ok=True)
        eval_step_dir.mkdir(parents=True, exist_ok=True)

        training_inp = TrainingInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml_path,
            training_config=cfg,
            dataset_validation_status=validation.get("status") if validation else None,
        )
        evaluation_inp = _EvalInput(
            dataset_path=dataset_path,
            data_yaml_path=data_yaml_path,
            weights_path="pipeline",  # resolved internally by the pipeline DSL
            mode=EvaluationMode.AZURE_PIPELINE_EVAL,
            promotion_policy_path=inp.promotion_policy_path,
            evaluation_config_path=inp.evaluation_config_path,
        )

        pipeline_runner = AzureMLPipelineRunner(azure_config)
        train_out, eval_out = pipeline_runner.run(
            training_inp, evaluation_inp, train_step_dir, eval_step_dir
        )

        # Store eval output path so _step_evaluation can pick it up without submitting a new job
        eval_output_path = str(eval_step_dir / "evaluation_output.json")
        train_out.pipeline_eval_output_path = eval_output_path

        return _StepOutcome(
            success=train_out.success,
            status_label="COMPLETED" if train_out.success else "FAILED",
            errors=train_out.errors,
            artifacts=train_out.artifacts + eval_out.artifacts,
            key_outputs={
                "best_weights_path": train_out.best_weights_path,
                "job_status": str(train_out.job_status),
                "training_output_path": str(train_step_dir / "training_output.json"),
                "pipeline_eval_output_path": eval_output_path,
            },
        )

    def _step_evaluation(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        import json as _json  # noqa: PLC0415

        from agentic_mlops.contracts.evaluation import EvaluationOutput as _EvalOut  # noqa: PLC0415

        training = step_outputs.get("training")
        if not training:
            return _StepOutcome(
                False,
                "FAILED",
                errors=["evaluation requires the training step to have run first."],
            )
        if training.get("skipped"):
            return _skip("training was skipped")

        # If training ran as an Azure ML pipeline, the eval output was already produced.
        pipeline_eval_path = training.get("pipeline_eval_output_path")
        if pipeline_eval_path:
            p = Path(pipeline_eval_path)
            if not p.exists():
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=[f"pipeline_eval_output_path not found on disk: {pipeline_eval_path}"],
                )
            eval_out = _EvalOut.model_validate(_json.loads(p.read_text(encoding="utf-8")))
            return _StepOutcome(
                success=eval_out.success,
                status_label="COMPLETED" if eval_out.success else "FAILED",
                errors=eval_out.errors,
                artifacts=eval_out.artifacts,
                key_outputs={
                    "report_path": str(Path(pipeline_eval_path).parent / "evaluation_report.json"),
                    "output_json_path": pipeline_eval_path,
                    "recommendation": (
                        str(eval_out.recommendation) if eval_out.recommendation else None
                    ),
                },
            )

        dataset_path, data_yaml_path = self._effective_dataset(step_outputs, inp)
        weights_path = training.get("best_weights_path") or "dry_run"

        if inp.evaluation_runner == "local-yolo":
            mode = EvaluationMode.LOCAL_EVAL
        elif inp.evaluation_runner == "azure-ml":
            mode = EvaluationMode.AZURE_EVAL
        elif inp.dry_run:
            mode = EvaluationMode.LOCAL_DRY_RUN
        else:
            mode = EvaluationMode.LOCAL_EVAL

        azure_runner = None
        if inp.evaluation_runner == "azure-ml":
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=["azure_config_path is required when evaluation_runner='azure-ml'."],
                )
            azure_runner = AzureMLEvaluationRunner(azure_config)

        step_dir = output_root / "evaluation"
        agent = EvaluationAgent(
            artifacts_dir=step_dir, azure_runner=azure_runner, **self._mlflow_kwargs()
        )
        result = agent.run(
            EvaluationInput(
                dataset_path=dataset_path,
                data_yaml_path=data_yaml_path,
                weights_path=weights_path,
                mode=mode,
                training_status=training.get("job_status", "completed"),
                promotion_policy_path=inp.promotion_policy_path,
                evaluation_config_path=inp.evaluation_config_path,
            )
        )
        # Write the full EvaluationOutput so ModelRegistryAgent Gate 3 can read
        # success=True.  The EvaluationAgent only writes evaluation_report.json
        # (human-readable subset); this is the machine-readable full record.
        output_json_path = step_dir / "evaluation_output.json"
        output_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return _StepOutcome(
            success=result.success,
            status_label="COMPLETED" if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "report_path": str(step_dir / "evaluation_report.json"),
                "output_json_path": str(output_json_path),
                "recommendation": str(result.recommendation) if result.recommendation else None,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
            },
        )

    def _step_model_decision(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        evaluation = step_outputs.get("evaluation")
        if not evaluation:
            return _StepOutcome(
                False,
                "FAILED",
                errors=["model_decision requires the evaluation step to have run first."],
            )
        if evaluation.get("skipped"):
            return _skip("evaluation was skipped")
        agent = ModelDecisionAgent(
            artifacts_dir=output_root / "model_decision", **self._mlflow_kwargs()
        )
        result = agent.run(
            ModelDecisionInput(
                evaluation_report_path=evaluation["report_path"],
                promotion_policy_path=inp.promotion_policy_path,
                evaluation_config_path=inp.evaluation_config_path,
                baseline_report_path=inp.baseline_report_path,
                measured_latency_ms=inp.measured_latency_ms,
            )
        )
        return _StepOutcome(
            success=result.success,
            status_label="COMPLETED" if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={"decision": str(result.decision) if result.decision else None},
        )

    def _step_approval(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        evaluation = step_outputs.get("evaluation")
        if not evaluation:
            return _StepOutcome(
                False, "FAILED", errors=["approval requires the evaluation step to have run first."]
            )
        if evaluation.get("skipped"):
            return _skip("evaluation was skipped")
        if not inp.interactive_approval and inp.approval_action is None:
            # H5 Model Approval gate: a legitimate pause, not an error — see module
            # docstring. Don't even call HumanApprovalAgent; it would just fail with
            # "Non-interactive mode requires --action".
            return _StepOutcome(
                success=True,
                status_label="PENDING",
                pending_approval=True,
                key_outputs={"approved": False},
            )

        step_dir = output_root / "approval"
        agent = HumanApprovalAgent(artifacts_dir=step_dir, **self._mlflow_kwargs())
        result = agent.run(
            ApprovalInput(
                evaluation_output_path=evaluation["report_path"],
                approver=inp.approver,
                output_dir=str(step_dir),
                interactive=inp.interactive_approval,
                action=inp.approval_action,
                force=inp.force_approve,
            )
        )
        approved = result.success and result.action == ApprovalAction.APPROVE_MODEL
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper() if result.success else "FAILED",
            errors=result.errors,
            artifacts=result.generated_artifacts,
            key_outputs={
                "approved": approved,
                "status": str(result.status),
                "action": str(result.action) if result.action else None,
            },
        )

    def _step_model_registry(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        approval = step_outputs.get("approval")
        if approval is not None and not approval.get("approved"):
            return _skip("approval (H5) was not approve_model")

        training = step_outputs.get("training", {})
        evaluation = step_outputs.get("evaluation", {})
        if training.get("skipped") or evaluation.get("skipped"):
            return _skip("an upstream step was skipped")
        approval_dir = output_root / "approval"

        registry_client: ModelRegistryClientBase | None = None
        if inp.registry_backend == RegistryBackend.AZURE_ML:
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=["azure_config_path is required when registry_backend='azure_ml'."],
                )
            registry_client = AzureMLModelRegistryClient(azure_config)

        agent = ModelRegistryAgent(
            artifacts_dir=output_root / "model_registry",
            registry_client=registry_client,
            **self._mlflow_kwargs(),
        )
        result = agent.run(
            ModelRegistrationInput(
                model_name=inp.model_name,
                training_output_path=training.get(
                    "training_output_path", str(output_root / "training" / "training_output.json")
                ),
                evaluation_output_path=evaluation.get(
                    "output_json_path", str(output_root / "evaluation" / "evaluation_output.json")
                ),
                approval_decision_path=str(approval_dir / "approval_decision.json"),
                registry_dir=inp.registry_dir,
                backend=inp.registry_backend,
            )
        )
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper(),
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "registry_path": result.registry_path,
                "version": result.version,
            },
        )

    def _step_deployment(
        self, inp: OrchestratorInput, output_root: Path, step_outputs: dict, azure_config
    ) -> _StepOutcome:
        approval = step_outputs.get("approval")
        if approval is not None and not approval.get("approved"):
            return _skip("approval (H5) was not approve_model")

        registry = step_outputs.get("model_registry")
        training = step_outputs.get("training", {})
        if training.get("skipped") or (registry is not None and registry.get("skipped")):
            return _skip("an upstream step was skipped")

        deployer = None
        if inp.deployment_backend == DeploymentBackend.AZURE_ML:
            if azure_config is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=["azure_config_path is required when deployment_backend='azure_ml'."],
                )
            azure_model_name = inp.azure_model_name
            azure_model_version = inp.azure_model_version
            if (
                not azure_model_name
                and registry
                and registry.get("registry_path")
                and inp.registry_backend == RegistryBackend.AZURE_ML
            ):
                # Chain straight from a prior azure_ml model_registry step.
                azure_model_name = inp.model_name
                azure_model_version = registry.get("version")
            if not azure_model_name or azure_model_version is None:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=[
                        "azure_model_name and azure_model_version are required for "
                        "deployment_backend='azure_ml' (set them directly, or include "
                        "model_registry with registry_backend='azure_ml' first)."
                    ],
                )
            deployer = ModelDeployer(azure_deployer=AzureMLOnlineEndpointDeployer(azure_config))
            model_path, model_version = None, None
        else:
            azure_model_name, azure_model_version = None, None
            if registry and registry.get("registry_path"):
                model_path = registry["registry_path"]
                model_version = registry.get("version")
            elif training.get("best_weights_path"):
                model_path = training["best_weights_path"]
                model_version = None
            else:
                return _StepOutcome(
                    False,
                    "FAILED",
                    errors=[
                        "deployment requires either model_registry or training to have "
                        "produced model weights."
                    ],
                )

        agent = DeploymentAgent(
            artifacts_dir=output_root / "deployment", deployer=deployer, **self._mlflow_kwargs()
        )
        result = agent.run(
            DeploymentInput(
                model_path=model_path,
                model_name=inp.model_name,
                model_version=model_version,
                target=inp.deployment_target,
                backend=inp.deployment_backend,
                export_format=inp.export_format,
                deployment_dir=inp.deployment_dir,
                endpoint_name=inp.endpoint_name,
                production_approval_path=inp.production_approval_path,
                rollback_plan=inp.rollback_plan,
                azure_model_name=azure_model_name,
                azure_model_version=azure_model_version,
            )
        )
        coarse = "blocked" if str(result.status) == "blocked" else None
        return _StepOutcome(
            success=result.success,
            status_label=str(result.status).upper(),
            errors=result.errors,
            artifacts=result.artifacts,
            key_outputs={
                "endpoint_name": result.endpoint_name,
                "release": result.release,
                "scoring_uri": result.scoring_uri,
            },
            coarse=coarse,
        )

    # ── State transition legality ────────────────────────────────────────────

    def _transition(self, state: dict, new_state: str, steps: list[str]) -> None:
        prev = state.get("current_state", "NEW")
        if not self._is_legal(prev, new_state, steps):
            raise RuntimeError(f"Illegal state transition: {prev} -> {new_state}")
        state["current_state"] = new_state

    def _is_legal(self, prev: str, nxt: str, steps: list[str]) -> bool:
        if prev in ("COMPLETED", "FAILED") or prev.endswith("_FAILED") or prev.endswith("_BLOCKED"):
            return False
        if prev == "NEW":
            return bool(steps) and nxt == f"{steps[0].upper()}_RUNNING"
        if prev in _PENDING_LABELS.values():
            # Resuming from a pause: retry the step that paused us.
            step = next(s for s, label in _PENDING_LABELS.items() if label == prev)
            return nxt == f"{step.upper()}_RUNNING"
        if prev.endswith("_RUNNING"):
            step = prev[: -len("_RUNNING")].lower()
            pending_label = _PENDING_LABELS.get(step)
            if pending_label and nxt == pending_label:
                return True
            return nxt in (
                f"{step.upper()}_COMPLETED",
                f"{step.upper()}_FAILED",
                f"{step.upper()}_BLOCKED",
            )
        if prev.endswith("_COMPLETED"):
            step = prev[: -len("_COMPLETED")].lower()
            if step not in steps:
                return False
            idx = steps.index(step)
            if idx + 1 < len(steps):
                return nxt == f"{steps[idx + 1].upper()}_RUNNING"
            return nxt == "COMPLETED"
        return False

    # ── Finalization ──────────────────────────────────────────────────────────

    def _immediate_fail(
        self, store: WorkflowStateStore, inp: OrchestratorInput, message: str
    ) -> OrchestratorOutput:
        state = {
            "workflow_id": inp.workflow_id,
            "trigger": inp.trigger,
            "current_state": "FAILED",
            "status": OrchestratorStatus.FAILED.value,
            "completed_steps": [],
            "step_outputs": {},
            "artifacts": [],
        }
        store.append_audit({"event": "workflow_failed_precheck", "message": message})
        return self._finish(
            store, state, [], [], OrchestratorStatus.FAILED, message, errors=[message]
        )

    def _finish(
        self,
        store: WorkflowStateStore,
        state: dict,
        step_results: list[OrchestratorStepResult],
        artifacts: list[str],
        status: OrchestratorStatus,
        message: str,
        errors: list[str] | None = None,
        pending_approval_id: str | None = None,
    ) -> OrchestratorOutput:
        state["status"] = status.value
        store.save_state(state)
        store.append_audit(
            {
                "event": "workflow_finished",
                "status": status.value,
                "current_state": state.get("current_state"),
            }
        )
        mlflow_run_id = state.get("mlflow_run_id")
        output = OrchestratorOutput(
            success=status in (OrchestratorStatus.COMPLETED, OrchestratorStatus.PENDING_APPROVAL),
            message=message,
            workflow_id=state.get("workflow_id", ""),
            status=status,
            current_state=state.get("current_state", "NEW"),
            last_agent=state.get("last_agent"),
            steps=step_results,
            pending_approval_id=pending_approval_id,
            state_path=str(store.state_path),
            audit_log_path=str(store.audit_log_path),
            artifacts=list(artifacts),
            errors=errors or [],
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=(
                self._mlflow_config.experiment_name if self._mlflow_config else None
            ),
            mlflow_tracking_uri=self._mlflow_config.tracking_uri if self._mlflow_config else None,
        )
        json_path, md_path = self._report_writer.write_orchestrator_report(
            output, store.workflow_dir / "artifacts"
        )
        output.artifacts += [str(json_path), str(md_path)]

        if self._cost_tracker is not None:
            cost_path = self._cost_tracker.write_report(
                state.get("workflow_id", ""), store.workflow_dir / "artifacts"
            )
            output.artifacts.append(str(cost_path))

        if self._mlflow and mlflow_run_id:
            if self._mlflow_config and self._mlflow_config.log_reports:
                for p in (str(json_path), str(md_path)):
                    self._mlflow.log_artifact(mlflow_run_id, p)
            # A pause is not terminal — leave the run open so --resume logs into it.
            if status != OrchestratorStatus.PENDING_APPROVAL:
                self._mlflow.log_tags(mlflow_run_id, {"workflow_status": status.value})
                final_status = "FINISHED" if status == OrchestratorStatus.COMPLETED else "FAILED"
                self._mlflow.end_run(mlflow_run_id, status=final_status)

        _WORKFLOW_EVENT: dict[OrchestratorStatus, str] = {
            OrchestratorStatus.COMPLETED: "workflow_completed",
            OrchestratorStatus.FAILED: "workflow_failed",
            OrchestratorStatus.BLOCKED: "workflow_blocked",
            OrchestratorStatus.PENDING_APPROVAL: "workflow_pending_approval",
        }
        wf_event = _WORKFLOW_EVENT.get(status)
        if wf_event:
            notify_payload: dict[str, Any] = {
                "workflow_id": state.get("workflow_id", ""),
                "status": status.value,
                "message": message,
            }
            if status == OrchestratorStatus.PENDING_APPROVAL:
                notify_payload["pending_approval_id"] = pending_approval_id
                notify_payload["current_state"] = state.get("current_state", "")
            self._notify(wf_event, notify_payload)

        if self._gh_client is not None and status == OrchestratorStatus.COMPLETED:
            gh_payload: dict[str, Any] = {
                "workflow_id": state.get("workflow_id", ""),
                "status": status.value,
                "step_outputs": state.get("step_outputs", {}),
            }
            try:
                self._gh_client.trigger("workflow_completed", gh_payload)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("GitHub Actions trigger failed: %s", exc)

        return output
