"""Pydantic contracts for the Deployment Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult


class DeploymentTarget(StrEnum):
    STAGING = "staging"
    PRODUCTION = "production"


class ExportFormat(StrEnum):
    PT = "pt"      # passthrough copy, no conversion — useful for local-only serving
    ONNX = "onnx"  # real Ultralytics export, requires the `onnx` package


class DeploymentStatus(StrEnum):
    DEPLOYED_TO_STAGING = "deployed_to_staging"
    DEPLOYED_TO_PRODUCTION = "deployed_to_production"
    BLOCKED = "blocked"
    FAILED = "failed"


class DeploymentInput(BaseModel):
    model_path: str
    model_name: str
    model_version: int | None = None
    target: DeploymentTarget = DeploymentTarget.STAGING
    export_format: ExportFormat = ExportFormat.ONNX
    deployment_dir: str = "outputs/deployments"
    endpoint_name: str | None = None

    # H6 safety gate (spec: "production только после approval") — both required
    # when target == production; staging can proceed without them ("semi-automatic").
    production_approval_path: str | None = None
    rollback_plan: str | None = None


class DeploymentOutput(ToolResult):
    status: DeploymentStatus = DeploymentStatus.FAILED
    endpoint_name: str | None = None
    exported_model_path: str | None = None
    release: int | None = None
    smoke_test_results: list[str] = Field(default_factory=list)
    deployment_report_path: str | None = None
    block_reason: str | None = None
