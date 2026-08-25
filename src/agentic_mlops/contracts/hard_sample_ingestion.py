"""Pydantic contracts for HardSampleIngestionAgent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ToolResult


class HardSampleIngestionInput(BaseModel):
    """Input for HardSampleIngestionAgent.

    Reads a hard_samples_manifest.json produced by MonitoringAgent, locates each
    image in *images_source_dir*, copies found images to a staging directory, then
    runs DataIntakeAgent over that staging directory.
    """

    hard_samples_manifest_path: str
    images_source_dir: str
    dataset_name: str
    # Optional provenance label forwarded to DataIntakeAgent.  Defaults to
    # "hard_sample_mining" so the intake manifest is self-describing.
    source: str | None = None

    # DataIntakeAgent thresholds (forwarded verbatim)
    expected_formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])
    min_files: int = Field(default=1, ge=0)
    corrupted_ratio_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    duplicate_ratio_threshold: float = Field(default=0.20, ge=0.0, le=1.0)


class HardSampleIngestionOutput(ToolResult):
    num_hard_samples_in_manifest: int = 0
    num_images_found: int = 0
    num_images_not_found: int = 0
    staging_dir: str | None = None
    # Status string from the inner DataIntakeAgent run (e.g. "passed",
    # "needs_human_source_approval", "failed").
    intake_status: str | None = None
    intake_manifest_path: str | None = None
