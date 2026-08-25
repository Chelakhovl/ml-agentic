"""Contracts for model risk scoring."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskFactor(BaseModel):
    name: str
    description: str
    contribution: float  # additive share of the final score in [0, 1]


class ModelRiskReport(BaseModel):
    """Aggregated risk score for a registered model.

    ``risk_score`` is in [0, 1] — higher means more risk.
    ``risk_level`` is derived from the score:
        < 0.35  → low
        0.35–0.65 → medium
        > 0.65  → high
    """

    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    factors: list[RiskFactor] = Field(default_factory=list)

    eval_report_path: str | None = None
    baseline_comparison_path: str | None = None
    generated_at: str = ""
