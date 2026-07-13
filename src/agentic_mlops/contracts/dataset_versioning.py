"""Pydantic contracts for the Dataset Versioning Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class DatasetVersionStatus(StrEnum):
    REGISTERED = "registered"
    DEDUPLICATED = "deduplicated"  # content hash matched an existing version — no-op
    BLOCKED = "blocked"
    FAILED = "failed"


class DatasetRegistryBackend(StrEnum):
    LOCAL = "local"        # filesystem registry with hash-based dedup
    AZURE_ML = "azure_ml"  # real Azure ML Data asset (no local dedup — Azure owns versioning)


class DatasetVersioningInput(BaseModel):
    dataset_path: str
    dataset_name: str
    registry_dir: str = "outputs/dataset_registry"
    parent_version: int | None = None
    workflow_id: str | None = None
    # Optional upstream reports — gates registration when either says "failed".
    validation_report_path: str | None = None
    label_quality_report_path: str | None = None
    approved_by: str | None = None
    source_batches: list[str] = Field(default_factory=list)
    backend: DatasetRegistryBackend = DatasetRegistryBackend.LOCAL


class DatasetLineage(BaseModel):
    dataset_name: str
    version: int
    parent_version: int | None = None
    workflow_id: str | None = None
    source_batches: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    hash: str
    approved_by: str | None = None
    validation_status: str | None = None
    label_qa_status: str | None = None
    registered_at: str
    source_dataset_path: str | None = None


class DatasetVersioningOutput(ToolResult):
    status: DatasetVersionStatus = DatasetVersionStatus.FAILED
    dataset_name: str = ""
    version: int | None = None
    dataset_version_path: str | None = None
    dataset_version_artifact: str | None = None
    hash: str | None = None
    lineage: DatasetLineage | None = None
    block_reason: str | None = None
