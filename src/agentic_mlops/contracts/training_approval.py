"""Pydantic contracts for the Training Approval gate (H4).

Per agentic_mlops_workflow_docs/agents/00_orchestrator_agent.md's H1-H6 list:
"H4 Training Approval" — a human sign-off after the dataset is validated
(and, optionally, versioned) but before a training run is started. Mirrors
HumanApprovalAgent's (H5 Model Approval) interactive/non-interactive shape,
but reads a dataset_quality_report.json instead of an evaluation_report.json,
and the decision space is intentionally smaller: there is no "request more
data" / "request label review" here — those are dataset fixes that would
need a brand new DatasetValidationAgent run anyway, not a decision this gate
can act on directly.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class TrainingApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class TrainingApprovalAction(StrEnum):
    APPROVE_TRAINING = "approve_training"
    REJECT_TRAINING = "reject_training"
    CANCEL = "cancel"


ACTION_TO_STATUS: dict[TrainingApprovalAction, TrainingApprovalStatus] = {
    TrainingApprovalAction.APPROVE_TRAINING: TrainingApprovalStatus.APPROVED,
    TrainingApprovalAction.REJECT_TRAINING: TrainingApprovalStatus.REJECTED,
    TrainingApprovalAction.CANCEL: TrainingApprovalStatus.CANCELLED,
}


class TrainingApprovalInput(BaseModel):
    dataset_report_path: str  # dataset_quality_report.json, written by DatasetValidationAgent
    approver: str | None = None
    output_dir: str
    interactive: bool = True
    action: TrainingApprovalAction | None = None
    comment: str | None = None
    # Required to approve_training in non-interactive mode when the dataset
    # status is "warning" (has non-blocking issues) rather than a clean "passed".
    force: bool = False
    dataset_version_or_path: str = "unknown"


class TrainingApprovalOutput(ToolResult):
    status: TrainingApprovalStatus = TrainingApprovalStatus.PENDING
    action: TrainingApprovalAction | None = None
    approver: str | None = None
    timestamp: str = ""
    dataset_report_path: str = ""
    comment: str | None = None
    generated_artifacts: list[str] = Field(default_factory=list)
