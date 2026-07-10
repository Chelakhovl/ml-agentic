"""Pydantic contracts for the Annotation / Pseudo-label Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .common import ToolResult


class ConfidenceBucket(StrEnum):
    HIGH = "high"      # candidate label — still sample-audited by a human, never auto-final
    MEDIUM = "medium"  # routed to human review
    LOW = "low"        # hard sample / expert review


class ConfidenceThresholds(BaseModel):
    auto_candidate: float = Field(default=0.90, ge=0.0, le=1.0)
    human_review: float = Field(default=0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> ConfidenceThresholds:
        if self.human_review > self.auto_candidate:
            raise ValueError(
                f"human_review ({self.human_review}) must be <= auto_candidate "
                f"({self.auto_candidate})"
            )
        return self


class AnnotationInput(BaseModel):
    images_path: str
    model_path: str
    confidence_thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    imgsz: int = 640
    device: str = "cpu"

    # Safety rule: pseudo-labels never overwrite existing human labels. When set,
    # any image with a label file already present under existing_labels_path is
    # skipped entirely (recorded, not re-predicted).
    existing_labels_path: str | None = None
    skip_existing_labels: bool = True


class PseudoLabelRecord(BaseModel):
    image: str
    bucket: ConfidenceBucket | None = None  # None only when skipped_existing_label
    num_detections: int = 0
    mean_confidence: float | None = None
    min_confidence: float | None = None
    label_path: str | None = None
    skipped_existing_label: bool = False


class AnnotationOutput(ToolResult):
    pseudo_labels_path: str | None = None
    review_queue_path: str | None = None
    report_path: str | None = None
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    num_images_processed: int = 0
    num_images_skipped_existing: int = 0
    records: list[PseudoLabelRecord] = Field(default_factory=list)
