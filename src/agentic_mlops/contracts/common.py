"""Shared Pydantic contracts used across all agents and tools."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Base response returned by every tool."""

    success: bool
    message: str
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelFramework(StrEnum):
    """ML framework used for training, evaluation, and inference."""

    YOLO = "yolo"
    TORCHVISION = "torchvision"
    ONNX_ONLY = "onnx_only"


class WorkflowState(StrEnum):
    """MVP subset of the full state machine."""

    NEW_DATASET = "NEW_DATASET"
    DATASET_VALIDATION_RUNNING = "DATASET_VALIDATION_RUNNING"
    DATASET_VALIDATION_PASSED = "DATASET_VALIDATION_PASSED"
    DATASET_VALIDATION_FAILED = "DATASET_VALIDATION_FAILED"
    DATASET_NEEDS_CLEANUP = "DATASET_NEEDS_CLEANUP"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    TRAINING_FAILED = "TRAINING_FAILED"
    EVALUATION_RUNNING = "EVALUATION_RUNNING"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    MODEL_DECISION_RUNNING = "MODEL_DECISION_RUNNING"
    MODEL_APPROVAL_REQUIRED = "MODEL_APPROVAL_REQUIRED"
    MODEL_APPROVED = "MODEL_APPROVED"
    MODEL_REJECTED = "MODEL_REJECTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
