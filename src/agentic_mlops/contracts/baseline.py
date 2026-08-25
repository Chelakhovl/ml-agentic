"""Contract for automatic baseline comparison results."""

from __future__ import annotations

from pydantic import BaseModel


class BaselineComparisonResult(BaseModel):
    """Delta metrics between a new evaluation and a fixed baseline.

    Positive deltas mean the new model improved over the baseline.
    ``improved`` is True when ``delta_map50 >= min_improvement_map50``.
    """

    new_map50: float = 0.0
    new_map50_95: float = 0.0
    new_precision: float = 0.0
    new_recall: float = 0.0

    baseline_map50: float = 0.0
    baseline_map50_95: float = 0.0
    baseline_precision: float = 0.0
    baseline_recall: float = 0.0

    delta_map50: float = 0.0
    delta_map50_95: float = 0.0
    delta_precision: float = 0.0
    delta_recall: float = 0.0

    improved: bool = False
    min_improvement_map50: float = 0.0

    baseline_path: str | None = None
    eval_report_path: str | None = None
    generated_at: str = ""
