"""Model decider — the core tool used by the Model Decision Agent.

Reads an already-computed evaluation_report.json (EvaluationAgent already ran the
threshold checks via workflows.policies.evaluate_metrics_against_policy — this tool
does not recompute them) and maps its 7-way EvaluationRecommendation onto the spec's
5-way ModelDecision. It then adds two checks that exist nowhere else in this
codebase: baseline comparison and runtime (model size / latency) budgets — both
promote a candidate only if it also beats its baseline and fits its deployment
budget, downgrading PROMOTE to RETRAIN otherwise. Never approves anything itself —
that stays HumanApprovalAgent's job.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_mlops.contracts.evaluation import EvaluationConfig, EvaluationMetrics
from agentic_mlops.contracts.model_decision import (
    ModelDecision,
    ModelDecisionInput,
    ModelDecisionOutput,
)
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.workflows.policies import PromotionPolicy

logger = get_logger(__name__)

_RECOMMENDATION_TO_DECISION: dict[str, ModelDecision] = {
    "promote_candidate": ModelDecision.PROMOTE,
    "retrain": ModelDecision.RETRAIN,
    "need_label_review": ModelDecision.NEED_LABEL_REVIEW,
    "collect_more_data": ModelDecision.NEED_MORE_DATA,
    "review_labels": ModelDecision.NEED_LABEL_REVIEW,
    "needs_human_review": ModelDecision.RETRAIN,
    "reject_candidate": ModelDecision.REJECT,
}


class ModelDecider:
    """Turns an evaluation report into an explainable, 5-way promotion decision."""

    def decide(self, inp: ModelDecisionInput) -> ModelDecisionOutput:
        try:
            eval_data = _load_json(inp.evaluation_report_path)
        except (OSError, json.JSONDecodeError) as exc:
            return _failed(f"Cannot read evaluation_report_path: {exc}")

        eval_success = bool(eval_data.get("success", False))
        eval_recommendation = eval_data.get("recommendation")
        metrics = _parse_metrics(eval_data.get("metrics"))
        passed_checks: list[str] = list(eval_data.get("passed_checks", []))
        failed_checks: list[str] = list(eval_data.get("failed_checks", []))

        if not eval_success or eval_recommendation is None:
            reason = eval_data.get("message") or "Evaluation did not succeed."
            return ModelDecisionOutput(
                success=True,
                message="Decision: REJECT — evaluation did not succeed.",
                decision=ModelDecision.REJECT,
                reasons=[reason],
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                metrics=metrics,
            )

        decision = _RECOMMENDATION_TO_DECISION.get(eval_recommendation, ModelDecision.RETRAIN)
        reasons: list[str] = list(failed_checks)

        policy: PromotionPolicy | None = None
        if inp.promotion_policy_path:
            policy = PromotionPolicy.from_yaml(inp.promotion_policy_path)

        baseline_metrics: EvaluationMetrics | None = None
        map50_improvement: float | None = None
        if inp.baseline_report_path and metrics is not None:
            try:
                baseline_data = _load_json(inp.baseline_report_path)
                baseline_metrics = _parse_metrics(baseline_data.get("metrics"))
            except (OSError, json.JSONDecodeError) as exc:
                reasons.append(f"Cannot read baseline_report_path: {exc}")
                baseline_metrics = None

            if baseline_metrics is not None:
                map50_improvement = metrics.map50 - baseline_metrics.map50
                if policy is not None and policy.require_improvement_over_baseline:
                    if map50_improvement < policy.baseline_improvement_min_map50:
                        reasons.append(
                            f"mAP50 improvement over baseline {map50_improvement:+.4f} "
                            f"is below required {policy.baseline_improvement_min_map50:.4f}"
                        )
                        decision = _downgrade_promote(decision)
                    else:
                        passed_checks.append(
                            f"mAP50 improved {map50_improvement:+.4f} over baseline "
                            f"(required >= {policy.baseline_improvement_min_map50:.4f})"
                        )

        runtime_cfg = None
        if inp.evaluation_config_path:
            runtime_cfg = EvaluationConfig.from_yaml(inp.evaluation_config_path).runtime

        model_size_mb: float | None = None
        model_path = eval_data.get("model_path")
        if model_path and runtime_cfg and runtime_cfg.max_model_size_mb is not None:
            p = Path(model_path)
            if p.exists():
                model_size_mb = p.stat().st_size / (1024 * 1024)
                if model_size_mb > runtime_cfg.max_model_size_mb:
                    reasons.append(
                        f"Model size {model_size_mb:.1f}MB exceeds max "
                        f"{runtime_cfg.max_model_size_mb:.1f}MB"
                    )
                    decision = _downgrade_promote(decision)
                else:
                    passed_checks.append(
                        f"Model size {model_size_mb:.1f}MB <= {runtime_cfg.max_model_size_mb:.1f}MB"
                    )
            else:
                reasons.append(f"model_path '{model_path}' not found — cannot check model size")

        if (
            inp.measured_latency_ms is not None
            and runtime_cfg is not None
            and runtime_cfg.max_latency_ms is not None
        ):
            if inp.measured_latency_ms > runtime_cfg.max_latency_ms:
                reasons.append(
                    f"Measured latency {inp.measured_latency_ms:.1f}ms exceeds max "
                    f"{runtime_cfg.max_latency_ms:.1f}ms"
                )
                decision = _downgrade_promote(decision)
            else:
                passed_checks.append(
                    f"Latency {inp.measured_latency_ms:.1f}ms <= {runtime_cfg.max_latency_ms:.1f}ms"
                )

        if not reasons:
            reasons.append("All promotion checks passed.")

        logger.info(
            "Model decision made",
            extra={"decision": decision, "map50_improvement": map50_improvement},
        )

        return ModelDecisionOutput(
            success=True,
            message=f"Decision: {decision.value.upper()}",
            decision=decision,
            reasons=reasons,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            map50_improvement=map50_improvement,
            model_size_mb=model_size_mb,
        )


def _downgrade_promote(decision: ModelDecision) -> ModelDecision:
    """A PROMOTE that fails baseline/runtime budget checks becomes RETRAIN.

    Non-PROMOTE decisions are left as-is — those checks only ever make a candidate
    *less* promotable, never more.
    """
    return ModelDecision.RETRAIN if decision == ModelDecision.PROMOTE else decision


def _parse_metrics(raw: dict | None) -> EvaluationMetrics | None:
    if not raw:
        return None
    return EvaluationMetrics.model_validate(raw)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _failed(message: str) -> ModelDecisionOutput:
    return ModelDecisionOutput(success=False, message=message, errors=[message])
