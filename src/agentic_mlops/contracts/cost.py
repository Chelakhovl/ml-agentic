"""Pydantic contracts for cost tracking."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ComputePricing(BaseModel):
    """Hourly USD rates keyed by Azure ML instance type or compute cluster name.

    Add custom entries to map a cluster name (e.g. "gpu-cluster") to an hourly
    rate if your cluster's VM size is not listed here.  Keys are matched exactly
    against the compute_type field of each CostEntry.
    """

    rates: dict[str, float] = Field(
        default_factory=lambda: {
            # General purpose (DS series)
            "Standard_DS2_v2": 0.14,
            "Standard_DS3_v2": 0.27,
            "Standard_DS4_v2": 0.54,
            "Standard_DS12_v2": 0.44,
            # GPU — NC series (K80)
            "Standard_NC6": 0.90,
            "Standard_NC12": 1.80,
            "Standard_NC24": 3.60,
            # GPU — NCsv3 series (V100)
            "Standard_NC6s_v3": 3.06,
            "Standard_NC12s_v3": 6.12,
            "Standard_NC24s_v3": 12.24,
            # GPU — ND series (A100/H100)
            "Standard_ND40rs_v2": 22.03,
            # GPU — NV series (inferencing / rendering)
            "Standard_NV6": 1.14,
            "Standard_NV12": 2.28,
            # Local / dry-run — no billable cloud cost
            "local": 0.0,
            "fake": 0.0,
        }
    )
    currency: str = "USD"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ComputePricing:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def rate_for(self, compute_type: str) -> float:
        """Return the hourly rate; 0.0 for unrecognised compute types."""
        return self.rates.get(compute_type, 0.0)


class CostEntry(BaseModel):
    """Cost record for one pipeline step."""

    step: str
    # Runner mode used: fake | local-yolo | azure-ml | azure-ml-pipeline | docker | aks
    runner: str
    # Azure ML instance type or cluster name, or "local" / "fake" for non-cloud.
    compute_type: str
    instance_count: int = 1
    duration_seconds: float | None = None
    estimated_cost: float = 0.0
    currency: str = "USD"
    notes: str = ""


class CostSummary(BaseModel):
    """Aggregated cost summary for a workflow run."""

    workflow_id: str
    total_cost: float
    currency: str
    entries: list[CostEntry] = Field(default_factory=list)
    generated_at: str = ""
