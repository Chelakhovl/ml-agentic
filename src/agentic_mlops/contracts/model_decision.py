"""Pydantic contracts for the Model Decision Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .common import ToolResult
from .evaluation import EvaluationMetrics


class ModelDecision(StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"
    RETRAIN = "retrain"
    NEED_MORE_DATA = "need_more_data"
    NEED_LABEL_REVIEW = "need_label_review"


class ModelDecisionInput(BaseModel):
    evaluation_report_path: str
    # Reuses workflows.policies.PromotionPolicy — same file EvaluationAgent already
    # loads. Only its require_improvement_over_baseline / baseline_improvement_min_map50
    # fields matter here; the threshold checks themselves were already applied by
    # EvaluationAgent and are read back from evaluation_report_path, not recomputed.
    promotion_policy_path: str | None = None
    # Reuses contracts.evaluation.EvaluationConfig.runtime for max_latency_ms /
    # max_model_size_mb — fields that existed on the contract but were never read
    # by any evaluation code path until this agent.
    evaluation_config_path: str | None = None
    baseline_report_path: str | None = None
    # This agent does not benchmark inference itself — pass an externally-measured
    # value (e.g. from a separate benchmarking step) to enable the latency check.
    measured_latency_ms: float | None = None


class ModelDecisionOutput(ToolResult):
    decision: ModelDecision | None = None
    reasons: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    metrics: EvaluationMetrics | None = None
    baseline_metrics: EvaluationMetrics | None = None
    map50_improvement: float | None = None
    model_size_mb: float | None = None
    decision_report_path: str | None = None
    required_approval_gate: str = "MODEL_APPROVAL"
