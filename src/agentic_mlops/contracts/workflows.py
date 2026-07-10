"""Pydantic contracts for the end-to-end MVP workflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from agentic_mlops.contracts.approvals import ApprovalAction
from agentic_mlops.contracts.model_registry import RegistryBackend

from .common import ToolResult


class MVPWorkflowStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class MVPWorkflowStepResult(BaseModel):
    step: str
    status: str
    success: bool
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class MVPWorkflowInput(BaseModel):
    dataset_path: str
    data_yaml_path: str
    training_config_path: str
    evaluation_config_path: str | None = None
    output_dir: str
    approver: str | None = None
    approval_action: ApprovalAction | None = None
    dry_run: bool = True
    interactive_approval: bool = True
    fail_on_warnings: bool = False
    force_approve: bool = False
    training_runner: str | None = None
    evaluation_runner: str | None = None
    # Path to azure_ml.yaml — required when training_runner/evaluation_runner is
    # "azure-ml" or registry_backend is "azure_ml"; the same workspace config is
    # reused across all three steps.
    azure_config_path: str | None = None
    # Model registry (Step 5 — conditional)
    register_approved_model: bool = False
    model_name: str = "yolo-model"
    registry_backend: RegistryBackend = RegistryBackend.LOCAL
    registry_dir: str = "outputs/model_registry"


class MVPWorkflowOutput(ToolResult):
    workflow_status: MVPWorkflowStatus = MVPWorkflowStatus.RUNNING
    steps: list[MVPWorkflowStepResult] = Field(default_factory=list)
    workflow_summary_path: str | None = None
    workflow_summary_md_path: str | None = None
    mlflow_run_id: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    # Registry results (populated when register_approved_model=True)
    registration_status: str | None = None
    registered_model_name: str | None = None
    registered_model_version: int | None = None
    registered_model_path: str | None = None
