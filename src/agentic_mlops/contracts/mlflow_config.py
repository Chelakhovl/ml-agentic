"""MLflow tracking configuration contract."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""

    enabled: bool = False
    tracking_uri: str = "sqlite:///outputs/mlruns.db"
    experiment_name: str = "agentic-mlops-local"
    run_name_prefix: str = "mvp-local-yolo"
    log_artifacts: bool = True
    log_reports: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> MLflowConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    @classmethod
    def disabled(cls) -> MLflowConfig:
        """Return a config that disables all tracking."""
        return cls(enabled=False)
