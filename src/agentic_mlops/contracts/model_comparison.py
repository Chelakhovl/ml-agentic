"""Pydantic contracts for the Model Comparison Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ToolResult
from .evaluation import EvaluationRecommendation


class ComparisonWeights(BaseModel):
    """Relative importance of each metric in the composite ranking score.

    Weights are normalised internally so they don't need to sum to 1.0.
    Set a weight to 0.0 to exclude that metric from the composite score.
    Latency and model_size are "lower is better" — the comparator inverts them.
    """

    map50: float = 0.40
    precision: float = 0.20
    recall: float = 0.20
    map50_95: float = 0.10
    latency_ms: float = 0.05  # lower is better
    model_size_mb: float = 0.05  # lower is better


class ModelComparisonInput(BaseModel):
    # Paths to evaluation_report.json files produced by EvaluationAgent.
    report_paths: list[str]
    # Human-readable labels for each model; defaults to path stem if omitted.
    model_names: list[str] | None = None
    # Optional PromotionPolicy YAML to determine per-model pass/fail.
    promotion_policy_path: str | None = None
    # Custom scoring weights; uses ComparisonWeights defaults if omitted.
    weights: ComparisonWeights | None = None
    # Per-model measured latency in ms (index-aligned with report_paths).
    # Overrides any latency value found in the report's metadata.
    measured_latency_ms: list[float | None] | None = None
    # Per-model measured model size in MB (index-aligned with report_paths).
    measured_model_size_mb: list[float | None] | None = None
    # Output directory for comparison_report.json/md
    output_dir: str = "outputs/comparison"


class ModelRankEntry(BaseModel):
    rank: int
    model_name: str
    report_path: str
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    latency_ms: float | None = None
    model_size_mb: float | None = None
    composite_score: float = 0.0
    # True/False when a policy was given; None when no policy was provided.
    passes_policy: bool | None = None
    recommendation: EvaluationRecommendation | None = None
    notes: list[str] = Field(default_factory=list)


class ModelComparisonOutput(ToolResult):
    rankings: list[ModelRankEntry] = Field(default_factory=list)
    # model_name of the top-ranked model that passes policy (or rank-1 if no policy).
    winner: str | None = None
    winner_report_path: str | None = None
    comparison_report_path: str | None = None
    md_report_path: str | None = None
    total_models: int = 0
    policy_passing: int | None = None  # None when no policy was given
