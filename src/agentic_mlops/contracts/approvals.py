"""Pydantic contracts for Human Approval Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_RETRAINING = "needs_retraining"
    NEEDS_MORE_DATA = "needs_more_data"
    NEEDS_LABEL_REVIEW = "needs_label_review"
    CANCELLED = "cancelled"


class ApprovalAction(StrEnum):
    APPROVE_MODEL = "approve_model"
    REJECT_MODEL = "reject_model"
    REQUEST_RETRAINING = "request_retraining"
    REQUEST_MORE_DATA = "request_more_data"
    REQUEST_LABEL_REVIEW = "request_label_review"
    CANCEL = "cancel"


ACTION_TO_STATUS: dict[ApprovalAction, ApprovalStatus] = {
    ApprovalAction.APPROVE_MODEL: ApprovalStatus.APPROVED,
    ApprovalAction.REJECT_MODEL: ApprovalStatus.REJECTED,
    ApprovalAction.REQUEST_RETRAINING: ApprovalStatus.NEEDS_RETRAINING,
    ApprovalAction.REQUEST_MORE_DATA: ApprovalStatus.NEEDS_MORE_DATA,
    ApprovalAction.REQUEST_LABEL_REVIEW: ApprovalStatus.NEEDS_LABEL_REVIEW,
    ApprovalAction.CANCEL: ApprovalStatus.CANCELLED,
}


class ApprovalRequest(BaseModel):
    candidate_model: str = "unknown"
    dataset_version_or_path: str = "unknown"
    evaluation_status: str
    recommendation: str | None
    key_metrics: dict[str, float] = Field(default_factory=dict)
    threshold_summary: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class ApprovalInput(BaseModel):
    evaluation_output_path: str
    approver: str | None = None
    output_dir: str
    interactive: bool = True
    action: ApprovalAction | None = None
    comment: str | None = None
    force: bool = False
    candidate_model: str = "unknown"
    dataset_version_or_path: str = "unknown"


class ApprovalOutput(ToolResult):
    status: ApprovalStatus = ApprovalStatus.PENDING
    action: ApprovalAction | None = None
    approver: str | None = None
    timestamp: str = ""
    evaluation_output_path: str = ""
    approval_request: ApprovalRequest | None = None
    comment: str | None = None
    generated_artifacts: list[str] = Field(default_factory=list)
