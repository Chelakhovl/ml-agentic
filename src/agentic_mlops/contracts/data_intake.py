"""Pydantic contracts for the Data Intake Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class DataIntakeStatus(StrEnum):
    PASSED = "passed"
    NEEDS_HUMAN_SOURCE_APPROVAL = "needs_human_source_approval"
    FAILED = "failed"


class DataIntakeInput(BaseModel):
    raw_data_path: str
    dataset_name: str
    source: str | None = None
    expected_formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])

    # Thresholds — ratios are computed against total files scanned
    min_files: int = Field(default=1, ge=0)
    corrupted_ratio_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    duplicate_ratio_threshold: float = Field(default=0.20, ge=0.0, le=1.0)


class ImageFileInfo(BaseModel):
    filename: str
    path: str
    size_bytes: int
    format: str | None = None
    width: int | None = None
    height: int | None = None
    sha256: str


class DataIntakeOutput(ToolResult):
    status: DataIntakeStatus = DataIntakeStatus.FAILED
    manifest_path: str | None = None
    dataset_name: str = ""
    source: str | None = None
    num_files: int = 0
    valid_images: int = 0
    corrupted_images: list[str] = Field(default_factory=list)
    duplicate_groups: list[list[str]] = Field(default_factory=list)
    unexpected_format_files: list[str] = Field(default_factory=list)
    image_records: list[ImageFileInfo] = Field(default_factory=list)
    pillow_available: bool = False
