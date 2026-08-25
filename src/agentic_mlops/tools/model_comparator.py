"""ModelComparator — loads evaluation reports and produces a ranked comparison.

Ranking algorithm
-----------------
1.  Load each evaluation_report.json; extract EvaluationMetrics + optional
    latency / model_size values.
2.  For each metric dimension, compute min-max normalisation across the models
    that have a non-None value for that dimension.
        normalised = (val - min) / (max - min)   — 0.5 when all values equal
3.  "Lower is better" metrics (latency_ms, model_size_mb) are inverted after
    normalisation: normalised = 1 - normalised.
4.  Composite score = weighted average of normalised values.  Weights whose
    corresponding metric is None for a given model are excluded from that
    model's denominator (weight is redistributed proportionally).
5.  Sort by composite score descending; assign ranks.
6.  Winner = highest-ranked model that passes the promotion policy (if
    provided); falls back to rank-1 when no policy is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.evaluation import EvaluationRecommendation
from agentic_mlops.contracts.model_comparison import (
    ComparisonWeights,
    ModelComparisonInput,
    ModelComparisonOutput,
    ModelRankEntry,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

# Metrics where a *lower* value is better; these are inverted during normalisation.
_LOWER_IS_BETTER = {"latency_ms", "model_size_mb"}

# Keys extracted from an evaluation report dict.
_METRIC_KEYS: list[str] = [
    "map50",
    "map50_95",
    "precision",
    "recall",
    "latency_ms",
    "model_size_mb",
]


def _load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    """Pull metric values out of a serialised EvaluationOutput."""
    metrics_block = report.get("metrics", {})
    result: dict[str, float | None] = {
        "map50": metrics_block.get("map50"),
        "map50_95": metrics_block.get("map50_95"),
        "precision": metrics_block.get("precision"),
        "recall": metrics_block.get("recall"),
        # latency/model_size live in metadata when written by ModelDecisionAgent
        "latency_ms": report.get("metadata", {}).get("measured_latency_ms"),
        "model_size_mb": report.get("metadata", {}).get("model_size_mb"),
    }
    # Treat 0.0 map50/precision/recall as present (may be a genuine zero).
    # Treat None as "not measured".
    return result


def _normalise(values: list[float | None]) -> list[float]:
    """Min-max normalise a list; entries that are None stay None-coded as 0.5."""
    present = [v for v in values if v is not None]
    if not present or len(present) == 1:
        return [0.5 if v is not None else 0.0 for v in values]
    lo, hi = min(present), max(present)
    if hi == lo:
        return [0.5 if v is not None else 0.0 for v in values]
    return [(v - lo) / (hi - lo) if v is not None else 0.0 for v in values]


def _composite(
    normalised: dict[str, list[float]],
    weights: ComparisonWeights,
    idx: int,
    available: dict[str, list[bool]],
) -> float:
    """Weighted average for model at index `idx`, redistributing weight for missing metrics."""
    total_weight = 0.0
    score = 0.0
    raw_weights: dict[str, float] = {
        "map50": weights.map50,
        "map50_95": weights.map50_95,
        "precision": weights.precision,
        "recall": weights.recall,
        "latency_ms": weights.latency_ms,
        "model_size_mb": weights.model_size_mb,
    }
    for key, w in raw_weights.items():
        if w <= 0.0:
            continue
        if not available[key][idx]:
            continue
        norm = normalised[key][idx]
        score += norm * w
        total_weight += w
    return score / total_weight if total_weight > 0 else 0.0


class ModelComparator:
    def compare(self, inp: ModelComparisonInput) -> ModelComparisonOutput:
        if not inp.report_paths:
            return ModelComparisonOutput(
                success=False,
                message="No report_paths provided.",
                errors=["report_paths must not be empty."],
            )

        weights = inp.weights or ComparisonWeights()
        n = len(inp.report_paths)

        # ── resolve model names ───────────────────────────────────────────────
        if inp.model_names and len(inp.model_names) == n:
            names = list(inp.model_names)
        else:
            names = [Path(p).stem for p in inp.report_paths]

        # ── load reports ──────────────────────────────────────────────────────
        reports: list[dict[str, Any]] = []
        load_errors: list[str] = []
        for path in inp.report_paths:
            try:
                reports.append(_load_report(path))
            except (OSError, json.JSONDecodeError) as exc:
                load_errors.append(f"Cannot read {path}: {exc}")
                reports.append({})

        if load_errors:
            return ModelComparisonOutput(
                success=False,
                message="Failed to load one or more evaluation reports.",
                errors=load_errors,
            )

        # ── extract metrics per model ─────────────────────────────────────────
        metrics_per_model: list[dict[str, float | None]] = []
        for i, report in enumerate(reports):
            extracted = _extract_metrics(report)
            # Override with explicit caller-provided values
            if inp.measured_latency_ms and i < len(inp.measured_latency_ms):
                if inp.measured_latency_ms[i] is not None:
                    extracted["latency_ms"] = inp.measured_latency_ms[i]
            if inp.measured_model_size_mb and i < len(inp.measured_model_size_mb):
                if inp.measured_model_size_mb[i] is not None:
                    extracted["model_size_mb"] = inp.measured_model_size_mb[i]
            metrics_per_model.append(extracted)

        # ── normalise per metric ──────────────────────────────────────────────
        normalised: dict[str, list[float]] = {}
        available: dict[str, list[bool]] = {}
        for key in _METRIC_KEYS:
            raw = [m[key] for m in metrics_per_model]
            avail = [v is not None for v in raw]
            norm = _normalise(raw)
            if key in _LOWER_IS_BETTER:
                norm = [1.0 - v if avail[i] else 0.0 for i, v in enumerate(norm)]
            normalised[key] = norm
            available[key] = avail

        # ── check promotion policy ────────────────────────────────────────────
        policy_results: list[bool | None] = [None] * n
        if inp.promotion_policy_path:
            for i, report in enumerate(reports):
                rec = report.get("recommendation")
                policy_results[i] = rec == EvaluationRecommendation.PROMOTE_CANDIDATE

        # ── compute composite scores and build entries ────────────────────────
        entries: list[ModelRankEntry] = []
        for i in range(n):
            m = metrics_per_model[i]
            rec_raw = reports[i].get("recommendation")
            try:
                rec = EvaluationRecommendation(rec_raw) if rec_raw else None
            except ValueError:
                rec = None
            score = _composite(normalised, weights, i, available)
            notes = _build_notes(reports[i], m, names[i])
            entries.append(
                ModelRankEntry(
                    rank=0,  # filled after sort
                    model_name=names[i],
                    report_path=inp.report_paths[i],
                    map50=m["map50"],
                    map50_95=m["map50_95"],
                    precision=m["precision"],
                    recall=m["recall"],
                    latency_ms=m["latency_ms"],
                    model_size_mb=m["model_size_mb"],
                    composite_score=score,
                    passes_policy=policy_results[i],
                    recommendation=rec,
                    notes=notes,
                )
            )

        # ── sort + assign ranks ───────────────────────────────────────────────
        entries.sort(key=lambda e: e.composite_score, reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank

        # ── determine winner ──────────────────────────────────────────────────
        winner_entry: ModelRankEntry | None = None
        if inp.promotion_policy_path:
            winner_entry = next((e for e in entries if e.passes_policy), None)
        else:
            winner_entry = entries[0] if entries else None

        policy_passing = (
            sum(1 for e in entries if e.passes_policy) if inp.promotion_policy_path else None
        )

        logger.info(
            "Model comparison complete",
            extra={"total": n, "winner": winner_entry.model_name if winner_entry else None},
        )

        return ModelComparisonOutput(
            success=True,
            message=(
                f"Compared {n} models. "
                + (
                    f"Winner: {winner_entry.model_name} "
                    f"(score={winner_entry.composite_score:.3f})."
                    if winner_entry
                    else "No model passed the promotion policy."
                )
            ),
            rankings=entries,
            winner=winner_entry.model_name if winner_entry else None,
            winner_report_path=winner_entry.report_path if winner_entry else None,
            total_models=n,
            policy_passing=policy_passing,
        )


def _build_notes(
    report: dict[str, Any], metrics: dict[str, float | None], model_name: str
) -> list[str]:
    notes: list[str] = []
    failed = report.get("failed_checks", [])
    if failed:
        notes.append(f"Failed checks: {', '.join(failed[:3])}" + (" …" if len(failed) > 3 else ""))
    rec = report.get("recommendation")
    if rec and rec != EvaluationRecommendation.PROMOTE_CANDIDATE:
        notes.append(f"Recommendation: {rec}")
    if metrics.get("latency_ms") is None:
        notes.append("Latency not measured — excluded from ranking.")
    if metrics.get("model_size_mb") is None:
        notes.append("Model size not measured — excluded from ranking.")
    return notes
