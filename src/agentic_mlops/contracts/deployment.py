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


class DeploymentBackend(StrEnum):
    LOCAL = "local"        # export + smoke test + versioned local release directory
    AZURE_ML = "azure_ml"  # real Managed Online Endpoint via the Azure ML SDK v2


class DeploymentInput(BaseModel):
    # Required for backend="local"; unused for backend="azure_ml" (which deploys
    # an already-registered Azure ML Model asset instead — see azure_model_name).
    model_path: str | None = None
    model_name: str
    model_version: int | None = None
    target: DeploymentTarget = DeploymentTarget.STAGING
    export_format: ExportFormat = ExportFormat.ONNX
    deployment_dir: str = "outputs/deployments"
    endpoint_name: str | None = None
    backend: DeploymentBackend = DeploymentBackend.LOCAL

    # H6 safety gate (spec: "production только после approval") — both required
    # when target == production; staging can proceed without them ("semi-automatic").
    production_approval_path: str | None = None
    rollback_plan: str | None = None

    # Required for backend="azure_ml" — references an Azure ML Model asset that
    # must already be registered (typically via `register-model --backend
    # azure_ml`/model_registry step). Connection info (subscription, resource
    # group, workspace, instance type/count) comes from an externally-injected
    # AzureMLConfig, same pattern as AzureMLModelRegistryClient.
    azure_model_name: str | None = None
    azure_model_version: int | None = None


class DeploymentOutput(ToolResult):
    status: DeploymentStatus = DeploymentStatus.FAILED
    endpoint_name: str | None = None
    exported_model_path: str | None = None
    release: int | None = None
    smoke_test_results: list[str] = Field(default_factory=list)
    deployment_report_path: str | None = None
    block_reason: str | None = None

    # backend="azure_ml" only
    scoring_uri: str | None = None
    azure_deployment_name: str | None = None
