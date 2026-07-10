"""Pydantic contracts for the Label QA Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class LabelQAStatus(StrEnum):
    PASSED = "passed"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class QAIssueType(StrEnum):
    BBOX_TOO_SMALL = "bbox_too_small"
    BBOX_TOO_LARGE = "bbox_too_large"
    BBOX_NEAR_BOUNDARY = "bbox_near_boundary"
    SUSPICIOUS_ASPECT_RATIO = "suspicious_aspect_ratio"
    MISSING_LABEL_FILE = "missing_label_file"
    CLASS_IMBALANCE = "class_imbalance"
    REFERENCE_MODEL_DISAGREEMENT = "reference_model_disagreement"


class LabelQAInput(BaseModel):
    dataset_path: str
    data_yaml_path: str | None = None

    # Optional reference model — enables the disagreement check. When omitted,
    # only the deterministic geometric/statistical checks run.
    reference_model_path: str | None = None
    reference_model_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    reference_model_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Thresholds for the deterministic checks
    too_small_threshold: float = Field(default=0.01, gt=0.0)
    too_large_threshold: float = Field(default=0.90, lt=1.0)
    boundary_margin: float = Field(default=0.01, ge=0.0)
    max_aspect_ratio: float = Field(default=10.0, gt=1.0)
    class_imbalance_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # Status thresholds — suspicious sample *count* that flips passed -> review_required
    review_required_threshold: int = Field(default=5, ge=0)


class SuspiciousSample(BaseModel):
    image: str
    split: str
    issue_type: QAIssueType
    message: str
    line: int | None = None
    class_id: int | None = None
    class_name: str | None = None
    bbox: list[float] | None = None  # [x_center, y_center, width, height], normalised


class LabelQAOutput(ToolResult):
    status: LabelQAStatus = LabelQAStatus.FAILED
    label_quality_score: float = 0.0
    suspicious_samples: list[SuspiciousSample] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    num_images_checked: int = 0
    num_labels_checked: int = 0
    reference_model_used: bool = False
    report_path: str | None = None
