"""Pydantic contracts for the Orchestrator Agent (`OrchestratorWorkflow`).

Per agentic_mlops_workflow_docs/agents/00_orchestrator_agent.md's own "MVP
implementation" note: "Orchestrator may be a simple Python class + JSON state
file" (`runs/<workflow_id>/state.json`, `runs/<workflow_id>/audit_log.jsonl`).
No State Store / Policy Engine / Approval Store / Notification service exists
as a separate networked component — those are all local-disk concerns here,
same "no real infra beyond what's needed" pattern as every other standalone
agent in this codebase.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .approvals import ApprovalAction
from .common import ToolResult
from .dataset_structuring import LabelFormat, SplitStrategy
from .deployment import DeploymentBackend, DeploymentTarget, ExportFormat
from .model_registry import RegistryBackend
from .training_approval import TrainingApprovalAction

# Canonical pipeline order. Every value in OrchestratorInput.steps must be one
# of these, and (if several are given) must appear in this relative order —
# this is the full "New Data -> ... -> Deployment" chain from the architecture
# docs, minus Annotation/Label QA (typically manual, interactive labeling-assist
# steps run *before* a dataset is finalized for structuring — not something you
# blindly auto-chain into a training run) and Monitoring (a separate, recurring,
# post-deploy concern, not a forward pipeline step).
PIPELINE_STEPS: tuple[str, ...] = (
    "data_intake",
    "dataset_structuring",
    "dataset_validation",
    "dataset_versioning",
    "training_approval",
    "training",
    "evaluation",
    "model_decision",
    "approval",
    "model_registry",
    "deployment",
)

# Default when OrchestratorInput.steps is omitted — the original 5 MVP agents,
# same as `run-mvp`.
DEFAULT_STEPS: tuple[str, ...] = (
    "dataset_validation",
    "training",
    "evaluation",
    "approval",
    "model_registry",
)


class OrchestratorStatus(StrEnum):
    RUNNING = "running"
    PENDING_APPROVAL = "pending_approval"  # paused at a human gate — not an error
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class OrchestratorInput(BaseModel):
    workflow_id: str
    trigger: str = "manual"
    # Which pipeline steps to run, in PIPELINE_STEPS order. None -> DEFAULT_STEPS.
    steps: list[str] | None = None
    resume: bool = False
    runs_dir: str = "runs"
    # Root for this run's per-step artifact subdirectories. Defaults to
    # <runs_dir>/<workflow_id>/artifacts.
    output_dir: str | None = None

    dry_run: bool = True
    azure_config_path: str | None = None

    # ── data_intake ──────────────────────────────────────────────────────────
    raw_data_path: str | None = None
    dataset_name: str = "dataset"
    data_source: str | None = None

    # ── dataset_structuring ──────────────────────────────────────────────────
    classes: list[str] = Field(default_factory=list)
    label_format: LabelFormat = LabelFormat.YOLO
    coco_annotations_path: str | None = None
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    train_ratio: float = Field(default=0.8, ge=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    group_by_regex: str | None = None

    # ── dataset_validation ───────────────────────────────────────────────────
    # Fallback dataset location when dataset_structuring is not in `steps`
    # (i.e. the caller already has a structured dataset). Overridden by
    # dataset_structuring's own output when that step ran.
    dataset_path: str | None = None
    data_yaml_path: str | None = None
    fail_on_warnings: bool = False

    # ── dataset_versioning ───────────────────────────────────────────────────
    dataset_registry_dir: str = "outputs/dataset_registry"
    parent_version: int | None = None
    approved_by: str | None = None
    source_batches: list[str] = Field(default_factory=list)

    # ── training_approval (H4 gate) ──────────────────────────────────────────
    training_approver: str | None = None
    training_approval_action: TrainingApprovalAction | None = None
    interactive_training_approval: bool = True
    force_training_approval: bool = False

    # ── training ─────────────────────────────────────────────────────────────
    training_config_path: str | None = None
    training_runner: str | None = None

    # ── evaluation ───────────────────────────────────────────────────────────
    evaluation_config_path: str | None = None
    evaluation_runner: str | None = None

    # ── model_decision ───────────────────────────────────────────────────────
    promotion_policy_path: str | None = None
    baseline_report_path: str | None = None
    measured_latency_ms: float | None = None

    # ── approval ─────────────────────────────────────────────────────────────
    approver: str | None = None
    approval_action: ApprovalAction | None = None
    interactive_approval: bool = True
    force_approve: bool = False

    # ── model_registry ───────────────────────────────────────────────────────
    model_name: str = "yolo-model"
    registry_backend: RegistryBackend = RegistryBackend.LOCAL
    registry_dir: str = "outputs/model_registry"

    # ── deployment ───────────────────────────────────────────────────────────
    deployment_target: DeploymentTarget = DeploymentTarget.STAGING
    deployment_backend: DeploymentBackend = DeploymentBackend.LOCAL
    export_format: ExportFormat = ExportFormat.ONNX
    deployment_dir: str = "outputs/deployments"
    endpoint_name: str | None = None
    production_approval_path: str | None = None
    rollback_plan: str | None = None
    # Only needed for deployment_backend="azure_ml" when NOT chaining from a prior
    # model_registry step with registry_backend="azure_ml" (which already produces
    # these — see OrchestratorWorkflow._step_deployment).
    azure_model_name: str | None = None
    azure_model_version: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: object) -> OrchestratorInput:
        """Load step-config fields from YAML; workflow_id/resume/runs_dir usually
        come from CLI flags instead, so they're accepted as override kwargs."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)


class OrchestratorStepResult(BaseModel):
    step: str
    status: str
    success: bool
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class OrchestratorOutput(ToolResult):
    workflow_id: str = ""
    status: OrchestratorStatus = OrchestratorStatus.RUNNING
    # Fine-grained, human-facing state label, e.g. "TRAINING_RUNNING",
    # "MODEL_APPROVAL_REQUIRED", "COMPLETED" — matches the spec's Output example.
    current_state: str = "NEW"
    last_agent: str | None = None
    steps: list[OrchestratorStepResult] = Field(default_factory=list)
    # Set only while status == PENDING_APPROVAL. A lightweight local identifier,
    # not a ticketing-system ID — there is no Approval Store service here, only
    # the same on-disk approval_decision.json pattern every other gate uses.
    pending_approval_id: str | None = None
    state_path: str | None = None
    audit_log_path: str | None = None
    mlflow_run_id: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
