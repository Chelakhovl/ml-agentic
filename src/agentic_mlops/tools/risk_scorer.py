"""Automatic model risk scorer.

Reads an evaluation_report.json (and optionally a baseline_comparison.json)
and produces a ModelRiskReport with a score in [0, 1] and a risk level.

Scoring factors (each contributes up to its weight):
  failed_checks     0.35  — failed policy checks → proportional contribution
  metric_proximity  0.30  — how close each passing metric is to its threshold
  recommendation    0.20  — non-PROMOTE_CANDIDATE recommendation adds full weight
  baseline_regression 0.15 — negative delta_map50 adds full weight
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.contracts.risk import ModelRiskReport, RiskFactor, RiskLevel
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_WEIGHT_FAILED_CHECKS = 0.35
_WEIGHT_METRIC_PROXIMITY = 0.30
_WEIGHT_RECOMMENDATION = 0.20
_WEIGHT_BASELINE_REGRESSION = 0.15

_PROXIMITY_DANGER_BAND = 0.05  # within 5 pp of threshold → full proximity risk

# Recommendation values that indicate the model is NOT ready to promote.
_NON_PROMOTE_RECOMMENDATIONS = {
    "retrain",
    "need_label_review",
    "collect_more_data",
    "review_labels",
    "reject_candidate",
    "needs_human_review",
}


def _level(score: float) -> RiskLevel:
    if score < 0.35:
        return RiskLevel.LOW
    if score <= 0.65:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


class ModelRiskScorer:
    """Compute a risk score for a registered model.

    Args:
        proximity_danger_band: Metrics within this many points of their
            threshold trigger full proximity risk. Default 0.05 (5 pp).
    """

    def __init__(self, proximity_danger_band: float = _PROXIMITY_DANGER_BAND) -> None:
        self._band = proximity_danger_band

    def score(
        self,
        eval_report_path: str | Path,
        *,
        baseline_comparison_path: str | Path | None = None,
        output_dir: Path | None = None,
    ) -> ModelRiskReport:
        """Compute risk score from *eval_report_path*.

        If *baseline_comparison_path* is given, adds a baseline-regression
        factor when delta_map50 < 0. If *output_dir* is given, writes
        ``risk_report.json`` there.
        """
        eval_path = Path(eval_report_path)
        try:
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("ModelRiskScorer: cannot read eval report", extra={"error": str(exc)})
            raise

        factors: list[RiskFactor] = []
        total_score = 0.0

        # ── Factor 1: failed checks ───────────────────────────────────────────
        failed_checks: list[str] = eval_data.get("failed_checks") or []
        passed_checks: list[str] = eval_data.get("passed_checks") or []
        total_checks = len(failed_checks) + len(passed_checks)
        if total_checks > 0:
            fail_ratio = len(failed_checks) / total_checks
            contribution = round(_WEIGHT_FAILED_CHECKS * fail_ratio, 4)
        else:
            fail_ratio = 0.0
            contribution = 0.0
        factors.append(
            RiskFactor(
                name="failed_checks",
                description=(
                    f"{len(failed_checks)}/{total_checks} policy checks failed"
                    if total_checks
                    else "no policy checks recorded"
                ),
                contribution=contribution,
            )
        )
        total_score += contribution

        # ── Factor 2: metric proximity to thresholds ──────────────────────────
        metrics = eval_data.get("metrics") or {}
        thresholds = {
            "map50": 0.75,
            "map50_95": 0.50,
            "precision": 0.70,
            "recall": 0.70,
        }
        proximity_sum = 0.0
        num_metrics = len(thresholds)
        for metric_name, threshold in thresholds.items():
            value = float(metrics.get(metric_name, 0.0))
            if value < threshold:
                proximity_sum += 1.0
            else:
                gap = value - threshold
                if gap < self._band:
                    proximity_sum += max(0.0, 1.0 - gap / self._band)
        proximity_ratio = proximity_sum / num_metrics if num_metrics else 0.0
        prox_contribution = round(_WEIGHT_METRIC_PROXIMITY * proximity_ratio, 4)
        factors.append(
            RiskFactor(
                name="metric_proximity",
                description=(
                    f"{proximity_ratio:.0%} of metrics are failing"
                    " or dangerously close to thresholds"
                ),
                contribution=prox_contribution,
            )
        )
        total_score += prox_contribution

        # ── Factor 3: recommendation ──────────────────────────────────────────
        recommendation = str(eval_data.get("recommendation") or "").lower()
        non_promote = recommendation in _NON_PROMOTE_RECOMMENDATIONS
        rec_contribution = _WEIGHT_RECOMMENDATION if non_promote else 0.0
        factors.append(
            RiskFactor(
                name="recommendation",
                description=(
                    f"evaluation recommendation is '{recommendation}' (not promote)"
                    if non_promote
                    else f"evaluation recommendation is '{recommendation or 'unknown'}'"
                ),
                contribution=rec_contribution,
            )
        )
        total_score += rec_contribution

        # ── Factor 4: baseline regression ─────────────────────────────────────
        bc_path_str: str | None = None
        if baseline_comparison_path is not None:
            bc_path = Path(baseline_comparison_path)
            bc_path_str = str(bc_path)
            try:
                bc_data = json.loads(bc_path.read_text(encoding="utf-8"))
                delta_map50 = float(bc_data.get("delta_map50", 0.0))
                regression = delta_map50 < 0
                bl_contribution = _WEIGHT_BASELINE_REGRESSION if regression else 0.0
                factors.append(
                    RiskFactor(
                        name="baseline_regression",
                        description=(
                            f"mAP50 regressed by {abs(delta_map50)*100:.2f}pp vs baseline"
                            if regression
                            else f"mAP50 improved by {delta_map50*100:.2f}pp vs baseline"
                        ),
                        contribution=bl_contribution,
                    )
                )
                total_score += bl_contribution
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "ModelRiskScorer: baseline comparison unreadable (skipped)",
                    extra={"error": str(exc)},
                )
        else:
            factors.append(
                RiskFactor(
                    name="baseline_regression",
                    description="no baseline comparison available",
                    contribution=0.0,
                )
            )

        risk_score = min(round(total_score, 4), 1.0)
        report = ModelRiskReport(
            risk_score=risk_score,
            risk_level=_level(risk_score),
            factors=factors,
            eval_report_path=str(eval_path),
            baseline_comparison_path=bc_path_str,
            generated_at=datetime.now(tz=UTC).isoformat(),
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "risk_report.json"
            out_path.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            logger.info(
                "Risk report written",
                extra={"path": str(out_path), "risk_score": risk_score, "level": report.risk_level},
            )

        return report


class FakeModelRiskScorer(ModelRiskScorer):
    """Test double — records calls; returns a configurable result."""

    def __init__(
        self,
        result: ModelRiskReport | None = None,
    ) -> None:
        super().__init__()
        self._result = result or ModelRiskReport(
            risk_score=0.2,
            risk_level=RiskLevel.LOW,
            factors=[RiskFactor(name="fake", description="fake factor", contribution=0.2)],
        )
        self.calls: list[str] = []

    def score(
        self,
        eval_report_path: str | Path,
        *,
        baseline_comparison_path: str | Path | None = None,
        output_dir: Path | None = None,
    ) -> ModelRiskReport:
        self.calls.append(str(eval_report_path))
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "risk_report.json"
            out_path.write_text(
                json.dumps(self._result.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
        return self._result
