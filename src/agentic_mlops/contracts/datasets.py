"""Pydantic contracts for Dataset Validation Agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .common import ToolResult


class IssueSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class LabelIssue(BaseModel):
    """A single issue found in a label file."""

    severity: IssueSeverity
    file: str
    line: int | None = None
    message: str


class DatasetValidationInput(BaseModel):
    """Input contract for the Dataset Validation tool."""

    dataset_path: str
    data_yaml_path: str | None = None
    fail_on_warnings: bool = False


class DatasetValidationOutput(ToolResult):
    """Output contract for the Dataset Validation tool."""

    status: Literal["passed", "failed", "warning"]
    num_images: int = 0
    num_labels: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    blocking_issues: list[str] = Field(default_factory=list)
    label_issues: list[LabelIssue] = Field(default_factory=list)
    report_path: str | None = None
