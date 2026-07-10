"""Contracts for the Evaluation Agent."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .common import ToolResult


class EvaluationMode(StrEnum):
    LOCAL_DRY_RUN = "local_dry_run"
    LOCAL_EVAL = "local_eval"
    AZURE_EVAL = "azure_eval"


class EvaluationRecommendation(StrEnum):
    PROMOTE_CANDIDATE = "promote_candidate"
    RETRAIN = "retrain"
    NEED_LABEL_REVIEW = "need_label_review"
    COLLECT_MORE_DATA = "collect_more_data"
    REVIEW_LABELS = "review_labels"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    REJECT_CANDIDATE = "reject_candidate"


class PerClassMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0


class EvaluationMetrics(BaseModel):
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    per_class_metrics: dict[str, PerClassMetrics] = Field(default_factory=dict)


class CriticalClassConfig(BaseModel):
    min_recall: float = 0.65


class EvaluationMetricsThresholds(BaseModel):
    min_map50: float = 0.75
    min_map50_95: float = 0.50
    min_precision: float = 0.70
    min_recall: float = 0.70


class EvaluationRuntimeConfig(BaseModel):
    max_latency_ms: float | None = None
    max_model_size_mb: float | None = None


class EvaluationConfig(BaseModel):
    task: str = "detect"
    mode: str = "val"
    imgsz: int = 640
    batch: int = 8
    device: str = "cpu"
    project: str = "outputs/yolo_val_runs"
    name: str = "val_run"
    metrics: EvaluationMetricsThresholds = Field(default_factory=EvaluationMetricsThresholds)
    critical_classes: dict[str, CriticalClassConfig] = Field(default_factory=dict)
    runtime: EvaluationRuntimeConfig = Field(default_factory=EvaluationRuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)


class EvaluationInput(BaseModel):
    dataset_path: str
    data_yaml_path: str
    weights_path: str
    workflow_id: str = "wf_local"
    mode: EvaluationMode = EvaluationMode.LOCAL_DRY_RUN
    training_status: Literal["completed", "failed", "cancelled"] | None = None
    promotion_policy_path: str | None = None
    evaluation_config_path: str | None = None


class EvaluationOutput(ToolResult):
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    recommendation: EvaluationRecommendation | None = None
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    evaluation_report_path: str | None = None
    mode: EvaluationMode = EvaluationMode.LOCAL_DRY_RUN
    runner: str | None = None
    model_path: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    reasons: list[str] = Field(default_factory=list)
