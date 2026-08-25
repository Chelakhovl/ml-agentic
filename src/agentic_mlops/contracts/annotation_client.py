"""Pydantic v2 contracts for the CVAT / Label Studio annotation client connector."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .common import ToolResult


class AnnotationBackend(StrEnum):
    CVAT = "cvat"
    LABEL_STUDIO = "label_studio"
    FAKE = "fake"


class AnnotationClientConfig(BaseModel):
    backend: AnnotationBackend = AnnotationBackend.FAKE
    base_url: str = ""
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    project_name: str = "agentic-mlops"
    timeout_seconds: int = 30


class AnnotationTask(BaseModel):
    task_id: str
    name: str
    status: str  # "new" | "in_progress" | "completed" | "validation"
    image_count: int = 0
    annotation_count: int = 0
    url: str | None = None


class AnnotationExportFormat(StrEnum):
    YOLO_DETECTION = "yolo_detection"
    COCO = "coco"
    VOC = "voc"


class UploadTaskInput(BaseModel):
    task_name: str
    image_paths: list[Path]
    labels: list[str]
    format: AnnotationExportFormat = AnnotationExportFormat.YOLO_DETECTION


class UploadTaskOutput(ToolResult):
    task: AnnotationTask | None = None


class ExportAnnotationsInput(BaseModel):
    task_id: str
    output_dir: Path
    format: AnnotationExportFormat = AnnotationExportFormat.YOLO_DETECTION


class ExportAnnotationsOutput(ToolResult):
    exported_files: list[Path] = Field(default_factory=list)
    annotation_count: int = 0


class ListTasksOutput(ToolResult):
    tasks: list[AnnotationTask] = Field(default_factory=list)
