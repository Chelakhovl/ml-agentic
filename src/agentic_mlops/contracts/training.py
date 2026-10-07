"""Pydantic contracts for the Training Agent."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, computed_field, field_validator

from .common import ModelFramework, ToolResult


class TrainingMode(StrEnum):
    LOCAL_DRY_RUN = "local_dry_run"
    LOCAL_TRAIN = "local_train"
    AZURE_TRAIN = "azure_train"
    AZURE_PIPELINE = "azure_pipeline"


def training_mode_to_runner(mode: TrainingMode) -> str:
    """Canonical external runner name for a TrainingMode.

    Used consistently in MLflow params/tags, training_report.md, and training_output.json.
    """
    _MAP = {
        TrainingMode.LOCAL_DRY_RUN: "fake",
        TrainingMode.LOCAL_TRAIN: "local-yolo",
        TrainingMode.AZURE_TRAIN: "azure-ml",
        TrainingMode.AZURE_PIPELINE: "azure-ml-pipeline",
    }
    return _MAP.get(mode, str(mode))


class TrainingJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """YOLO training configuration, loadable from a YAML file."""

    model: str = "yolo11m.pt"
    epochs: int = Field(default=100, gt=0)
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=16, gt=0)
    optimizer: str = "auto"
    patience: int = Field(default=20, ge=0)
    project: str = "agentic_mlops"
    name: str = "run_001"
    mode: TrainingMode = TrainingMode.LOCAL_DRY_RUN

    # Azure ML — required only for azure_train mode
    azure_compute_cluster: str | None = None
    azure_environment: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v: object) -> str:
        return str(v)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a TrainingConfig from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)


class TrainingArtifact(BaseModel):
    """A single artifact produced by a training job."""

    name: str
    path: str
    artifact_type: Literal["weights", "config", "report", "log", "plan"]


class TrainingInput(BaseModel):
    """Input contract for TrainingAgent.run()."""

    dataset_path: str
    data_yaml_path: str
    training_config: TrainingConfig
    workflow_id: str = "wf_local"
    # Populated from the output of DatasetValidationAgent
    dataset_validation_status: Literal["passed", "failed", "warning"] | None = None
    framework: ModelFramework = ModelFramework.YOLO


class TrainingOutput(ToolResult):
    """Output contract for TrainingAgent.run()."""

    job_id: str | None = None
    mlflow_run_id: str | None = None
    best_weights_path: str | None = None
    training_plan_path: str | None = None
    report_path: str | None = None
    job_status: TrainingJobStatus = TrainingJobStatus.PENDING
    training_artifacts: list[TrainingArtifact] = Field(default_factory=list)
    mode: TrainingMode = TrainingMode.LOCAL_DRY_RUN

    # Azure ML — populated only for AZURE_TRAIN / AZURE_PIPELINE modes
    azure_job_name: str | None = None
    azure_job_status: str | None = None
    azure_studio_url: str | None = None
    azure_compute_name: str | None = None
    azure_experiment_name: str | None = None
    azure_output_name: str | None = None
    azure_output_uri: str | None = None
    remote_started_at: str | None = None
    remote_completed_at: str | None = None
    # Populated by AzureMLPipelineRunner — path to the eval output written during pipeline run
    pipeline_eval_output_path: str | None = None

    @computed_field
    @property
    def runner(self) -> str:
        return training_mode_to_runner(self.mode)
