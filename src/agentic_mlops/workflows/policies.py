"""Promotion policy: load from YAML and evaluate metrics against thresholds."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agentic_mlops.contracts.evaluation import (
    EvaluationMetrics,
    EvaluationRecommendation,
    PerClassMetrics,
)


class PolicyThresholds(BaseModel):
    map50_min: float = 0.75
    map50_95_min: float = 0.50
    precision_min: float = 0.70
    recall_min: float = 0.70


class PromotionPolicy(BaseModel):
    thresholds: PolicyThresholds = Field(default_factory=PolicyThresholds)
    critical_class_recall_min: float = 0.65
    critical_classes: list[str] = Field(default_factory=list)
    per_class_recall_min: dict[str, float] = Field(default_factory=dict)
    require_improvement_over_baseline: bool = False
    baseline_improvement_min_map50: float = 0.01

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromotionPolicy:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> PromotionPolicy:
        return cls()


def evaluate_metrics_against_policy(
    metrics: EvaluationMetrics,
    policy: PromotionPolicy,
) -> tuple[EvaluationRecommendation, list[str], list[str]]:
    """Return (recommendation, passed_checks, failed_checks).

    Decision logic:
    - Any critical class below critical_class_recall_min → NEED_LABEL_REVIEW
    - Only recall fails (map50/map50_95/precision pass) → COLLECT_MORE_DATA
    - Only precision fails (map50/map50_95/recall pass) → REVIEW_LABELS
    - Any other global threshold failure → RETRAIN
    - All pass → PROMOTE_CANDIDATE
    """
    passed: list[str] = []
    failed: list[str] = []
    critical_fail = False

    t = policy.thresholds

    map50_pass = metrics.map50 >= t.map50_min
    map50_95_pass = metrics.map50_95 >= t.map50_95_min
    precision_pass = metrics.precision >= t.precision_min
    recall_pass = metrics.recall >= t.recall_min

    def _record(name: str, value: float, minimum: float, ok: bool) -> None:
        if ok:
            passed.append(f"{name} {value:.4f} >= {minimum:.4f}")
        else:
            failed.append(f"{name} {value:.4f} < {minimum:.4f}")

    _record("map50", metrics.map50, t.map50_min, map50_pass)
    _record("map50_95", metrics.map50_95, t.map50_95_min, map50_95_pass)
    _record("precision", metrics.precision, t.precision_min, precision_pass)
    _record("recall", metrics.recall, t.recall_min, recall_pass)

    for cls_name in policy.critical_classes:
        pcm: PerClassMetrics | None = metrics.per_class_metrics.get(cls_name)
        if pcm is None:
            failed.append(f"critical class '{cls_name}' missing from per-class metrics")
            critical_fail = True
        elif pcm.recall < policy.critical_class_recall_min:
            failed.append(
                f"critical class '{cls_name}' recall {pcm.recall:.4f}"
                f" < {policy.critical_class_recall_min:.4f}"
            )
            critical_fail = True
        else:
            passed.append(
                f"critical class '{cls_name}' recall {pcm.recall:.4f}"
                f" >= {policy.critical_class_recall_min:.4f}"
            )

    for cls_name, min_recall in policy.per_class_recall_min.items():
        pcm = metrics.per_class_metrics.get(cls_name)
        if pcm is None:
            failed.append(f"class '{cls_name}' missing from per-class metrics")
        elif pcm.recall < min_recall:
            failed.append(
                f"class '{cls_name}' recall {pcm.recall:.4f} < {min_recall:.4f}"
            )
        else:
            passed.append(
                f"class '{cls_name}' recall {pcm.recall:.4f} >= {min_recall:.4f}"
            )

    if critical_fail:
        recommendation = EvaluationRecommendation.NEED_LABEL_REVIEW
    elif not failed:
        recommendation = EvaluationRecommendation.PROMOTE_CANDIDATE
    elif not recall_pass and map50_pass and map50_95_pass and precision_pass:
        recommendation = EvaluationRecommendation.COLLECT_MORE_DATA
    elif not precision_pass and map50_pass and map50_95_pass and recall_pass:
        recommendation = EvaluationRecommendation.REVIEW_LABELS
    else:
        recommendation = EvaluationRecommendation.RETRAIN

    return recommendation, passed, failed
