"""Pydantic config contracts for Docker image build and AKS deployment."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DockerConfig(BaseModel):
    """Configuration for Docker image build + push."""

    # e.g. "myacr.azurecr.io" or "docker.io/myorg"
    registry: str
    # Overrides model_name when building the image name
    image_name: str | None = None
    # e.g. "mlops/" — prepended to image_name inside the registry
    image_prefix: str = ""
    base_image: str = "python:3.11-slim"
    # Extra pip packages added with `RUN pip install ...`
    extra_requirements: list[str] = Field(default_factory=list)
    # Port the serving process listens on inside the container
    port: int = 8080

    @classmethod
    def from_yaml(cls, path: str | Path) -> DockerConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def image_tag(self, model_name: str, version_tag: str) -> str:
        """Return the fully-qualified image tag for a given model + version."""
        name = self.image_prefix + (self.image_name or model_name.replace("_", "-"))
        return f"{self.registry}/{name}:{version_tag}"


class AksConfig(BaseModel):
    """Configuration for Azure Kubernetes Service deployment via kubectl."""

    namespace: str = "mlops"
    replicas: int = 1
    port: int = 8080
    # ClusterIP (internal only) | LoadBalancer (external IP) | NodePort
    service_type: str = "ClusterIP"
    cpu_request: str = "500m"
    memory_request: str = "1Gi"
    cpu_limit: str = "2000m"
    memory_limit: str = "2Gi"
    # Path to a kubeconfig file; None = kubectl default (~/.kube/config)
    kubeconfig_path: str | None = None
    # kubectl context to use; None = current context
    context: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AksConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
