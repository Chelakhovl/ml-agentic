"""Pydantic contracts for the Model Registry Agent."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .common import ToolResult

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class RegistrationStatus(StrEnum):
    REGISTERED = "registered"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class RegistryBackend(StrEnum):
    LOCAL = "local"
    MLFLOW = "mlflow"
    AZURE_ML = "azure_ml"


class RegistrationArtifact(BaseModel):
    name: str
    path: str
    artifact_type: str  # "weights" | "lineage" | "model_card" | "output"


class ModelLineage(BaseModel):
    """Auditable provenance record stored alongside every registered version."""

    # Training
    training_job_id: str | None = None
    training_runner: str | None = None
    training_model: str | None = None
    training_config: dict[str, Any] | None = None
    data_yaml: str | None = None
    dataset_path: str | None = None

    # Evaluation
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    evaluation_recommendation: str | None = None

    # Approval
    approved_by: str | None = None
    approval_action: str | None = None
    approval_timestamp: str | None = None
    approval_comment: str | None = None

    # MLflow
    mlflow_run_id: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None

    # Source files
    training_output_path: str | None = None
    evaluation_output_path: str | None = None
    approval_decision_path: str | None = None


class ModelRegistrationInput(BaseModel):
    model_name: str
    training_output_path: str
    evaluation_output_path: str
    approval_decision_path: str
    registry_dir: str
    backend: RegistryBackend = RegistryBackend.LOCAL
    mlflow_run_id: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not _MODEL_NAME_RE.match(v):
            raise ValueError(
                f"model_name '{v}' is invalid. "
                "Must match ^[a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}$"
            )
        return v


class ModelRegistrationOutput(ToolResult):
    status: RegistrationStatus = RegistrationStatus.BLOCKED
    model_name: str = ""
    version: int | None = None
    registry_path: str | None = None
    lineage: ModelLineage | None = None
    registration_artifacts: list[RegistrationArtifact] = Field(default_factory=list)
    skip_reason: str | None = None
    block_reason: str | None = None
