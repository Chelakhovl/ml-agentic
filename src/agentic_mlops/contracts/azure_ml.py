"""Pydantic v2 config for Azure ML training jobs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class AzureMLEnvironmentMode(StrEnum):
    REGISTERED = "registered"
    INLINE = "inline"


class AzureMLAuthConfig(BaseModel):
    credential: str = "default"


class AzureMLEnvironmentConfig(BaseModel):
    mode: AzureMLEnvironmentMode = AzureMLEnvironmentMode.REGISTERED
    registered_environment: str | None = None
    base_image: str | None = None
    conda_file: str | None = None

    @model_validator(mode="after")
    def _check_env_fields(self) -> AzureMLEnvironmentConfig:
        if self.mode == AzureMLEnvironmentMode.REGISTERED and not self.registered_environment:
            raise ValueError("registered_environment is required when mode='registered'")
        if self.mode == AzureMLEnvironmentMode.INLINE and not self.base_image:
            raise ValueError("base_image is required when mode='inline'")
        return self


class AzureMLJobConfig(BaseModel):
    timeout_minutes: int = Field(default=120, gt=0)
    stream_logs: bool = True
    download_outputs: bool = True
    output_name: str = "model_output"
    tags: dict[str, str] = Field(
        default_factory=lambda: {"system": "agentic-mlops", "task": "object-detection"}
    )


class AzureMLDataConfig(BaseModel):
    input_mode: Literal["ro_mount", "download"] = "ro_mount"
    asset_uri: str | None = None


class AzureMLServingConfig(BaseModel):
    """Managed Online Endpoint provisioning settings, shared across deployments.

    Separate from AzureMLEnvironmentConfig (training/eval jobs) because a serving
    environment typically needs different dependencies (an inference server, e.g.
    azureml-inference-server-http) than a training environment — reuse the same
    `registered_environment` only if it actually has both.
    """

    instance_type: str = "Standard_DS2_v2"
    instance_count: int = Field(default=1, gt=0)
    auth_mode: Literal["key", "aml_token"] = "key"
    timeout_minutes: int = Field(default=20, gt=0)
    environment: AzureMLEnvironmentConfig | None = None


class AzureMLConfig(BaseModel):
    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_name: str
    experiment_name: str = "agentic-mlops-yolo-training"
    authentication: AzureMLAuthConfig = Field(default_factory=AzureMLAuthConfig)
    environment: AzureMLEnvironmentConfig = Field(default_factory=AzureMLEnvironmentConfig)
    job: AzureMLJobConfig = Field(default_factory=AzureMLJobConfig)
    data: AzureMLDataConfig = Field(default_factory=AzureMLDataConfig)
    serving: AzureMLServingConfig = Field(default_factory=AzureMLServingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AzureMLConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
