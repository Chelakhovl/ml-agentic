"""Pydantic contracts for the Dataset Structuring Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .common import ToolResult


class LabelFormat(StrEnum):
    YOLO = "yolo"
    COCO = "coco"


class SplitStrategy(StrEnum):
    RANDOM = "random"
    GROUPED_BY_SOURCE = "grouped_by_source"


class DatasetStructuringInput(BaseModel):
    raw_data_path: str
    output_dataset_path: str
    classes: list[str] = Field(min_length=1)

    label_format: LabelFormat = LabelFormat.YOLO
    # Required when label_format == "coco"; a single COCO-style annotations JSON file.
    coco_annotations_path: str | None = None

    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    train_ratio: float = Field(default=0.8, ge=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)

    # Only used when split_strategy == "grouped_by_source". Must have exactly one
    # capturing group, e.g. r"^(video\d+)_frame\d+" to keep video frames together.
    # Without it, grouped_by_source degrades to one-file-per-group (no leakage
    # protection) and a warning is recorded instead of failing outright.
    group_by_regex: str | None = None

    seed: int = 42

    @model_validator(mode="after")
    def _check_ratios_sum_to_one(self) -> DatasetStructuringInput:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )
        return self


class SplitAssignment(BaseModel):
    filename: str
    split: str
    group_key: str | None = None


class DatasetStructuringOutput(ToolResult):
    structured_dataset_path: str | None = None
    data_yaml_path: str | None = None
    split_report_path: str | None = None
    num_images: int = 0
    num_labels: int = 0
    split_counts: dict[str, int] = Field(default_factory=dict)
    classes: list[str] = Field(default_factory=list)
    assignments: list[SplitAssignment] = Field(default_factory=list)
