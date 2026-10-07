"""Comprehensive Pydantic v2 contract validation tests.

Covers: boundary values, cross-field validators, enum coercion, serialization
round-trips, and the gaps we hardened in this audit pass.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentic_mlops.contracts.annotation import AnnotationInput, ConfidenceThresholds
from agentic_mlops.contracts.common import ModelFramework, WorkflowState
from agentic_mlops.contracts.deployment import DeploymentInput, DeploymentTarget
from agentic_mlops.contracts.evaluation import (
    EvaluationInput,
    EvaluationMetrics,
    EvaluationMode,
)
from agentic_mlops.contracts.monitoring import MonitoringInput, MonitoringThresholds
from agentic_mlops.contracts.orchestrator import OrchestratorInput, PIPELINE_STEPS
from agentic_mlops.contracts.training import TrainingConfig, TrainingInput, TrainingMode
from agentic_mlops.workflows.policies import PolicyThresholds, PromotionPolicy


# ── TrainingConfig boundary values ────────────────────────────────────────────


class TestTrainingConfigBoundaries:
    def test_epochs_zero_raises(self):
        with pytest.raises(ValidationError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_epochs_negative_raises(self):
        with pytest.raises(ValidationError, match="epochs"):
            TrainingConfig(epochs=-1)

    def test_epochs_one_ok(self):
        cfg = TrainingConfig(epochs=1)
        assert cfg.epochs == 1

    def test_imgsz_zero_raises(self):
        with pytest.raises(ValidationError, match="imgsz"):
            TrainingConfig(imgsz=0)

    def test_imgsz_negative_raises(self):
        with pytest.raises(ValidationError, match="imgsz"):
            TrainingConfig(imgsz=-640)

    def test_imgsz_one_ok(self):
        cfg = TrainingConfig(imgsz=1)
        assert cfg.imgsz == 1

    def test_batch_zero_raises(self):
        with pytest.raises(ValidationError, match="batch"):
            TrainingConfig(batch=0)

    def test_batch_one_ok(self):
        cfg = TrainingConfig(batch=1)
        assert cfg.batch == 1

    def test_patience_zero_ok(self):
        cfg = TrainingConfig(patience=0)
        assert cfg.patience == 0

    def test_patience_negative_raises(self):
        with pytest.raises(ValidationError, match="patience"):
            TrainingConfig(patience=-1)

    def test_mode_coercion_from_string(self):
        cfg = TrainingConfig(mode="azure_train")
        assert cfg.mode == TrainingMode.AZURE_TRAIN

    def test_mode_invalid_string_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(mode="invalid_mode")


# ── AnnotationInput boundary values ──────────────────────────────────────────


class TestAnnotationInputBoundaries:
    def test_imgsz_zero_raises(self):
        with pytest.raises(ValidationError, match="imgsz"):
            AnnotationInput(images_path="/img", model_path="/m.pt", imgsz=0)

    def test_imgsz_negative_raises(self):
        with pytest.raises(ValidationError, match="imgsz"):
            AnnotationInput(images_path="/img", model_path="/m.pt", imgsz=-1)

    def test_imgsz_one_ok(self):
        inp = AnnotationInput(images_path="/img", model_path="/m.pt", imgsz=1)
        assert inp.imgsz == 1

    def test_default_imgsz(self):
        inp = AnnotationInput(images_path="/img", model_path="/m.pt")
        assert inp.imgsz == 640


# ── ConfidenceThresholds cross-field validator ────────────────────────────────


class TestConfidenceThresholds:
    def test_human_review_gt_auto_candidate_raises(self):
        with pytest.raises(ValidationError, match="human_review"):
            ConfidenceThresholds(auto_candidate=0.5, human_review=0.9)

    def test_equal_values_ok(self):
        ct = ConfidenceThresholds(auto_candidate=0.7, human_review=0.7)
        assert ct.auto_candidate == ct.human_review

    def test_both_zero_ok(self):
        ct = ConfidenceThresholds(auto_candidate=0.0, human_review=0.0)
        assert ct.auto_candidate == 0.0

    def test_both_one_ok(self):
        ct = ConfidenceThresholds(auto_candidate=1.0, human_review=1.0)
        assert ct.auto_candidate == 1.0

    def test_auto_candidate_gt_one_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceThresholds(auto_candidate=1.1, human_review=0.5)

    def test_human_review_negative_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceThresholds(auto_candidate=0.9, human_review=-0.1)


# ── MonitoringThresholds boundary values ─────────────────────────────────────


class TestMonitoringThresholds:
    def test_p95_latency_zero_raises(self):
        with pytest.raises(ValidationError, match="p95_latency_ms"):
            MonitoringThresholds(p95_latency_ms=0.0)

    def test_p95_latency_negative_raises(self):
        with pytest.raises(ValidationError, match="p95_latency_ms"):
            MonitoringThresholds(p95_latency_ms=-1.0)

    def test_error_rate_gt_one_raises(self):
        with pytest.raises(ValidationError, match="error_rate"):
            MonitoringThresholds(error_rate=1.1)

    def test_drift_score_negative_raises(self):
        with pytest.raises(ValidationError, match="drift_score"):
            MonitoringThresholds(drift_score=-0.01)

    def test_valid_thresholds_ok(self):
        t = MonitoringThresholds(low_confidence_ratio=0.0, error_rate=1.0, p95_latency_ms=0.001)
        assert t.low_confidence_ratio == 0.0
        assert t.error_rate == 1.0


# ── DeploymentInput canary_percentage boundary ────────────────────────────────


class TestDeploymentInputBoundaries:
    def test_canary_zero_raises(self):
        with pytest.raises(ValidationError, match="canary_percentage"):
            DeploymentInput(model_name="m", canary_percentage=0)

    def test_canary_101_raises(self):
        with pytest.raises(ValidationError, match="canary_percentage"):
            DeploymentInput(model_name="m", canary_percentage=101)

    def test_canary_one_ok(self):
        inp = DeploymentInput(model_name="m", canary_percentage=1)
        assert inp.canary_percentage == 1

    def test_canary_100_ok(self):
        inp = DeploymentInput(model_name="m", canary_percentage=100)
        assert inp.canary_percentage == 100

    def test_default_canary(self):
        inp = DeploymentInput(model_name="m")
        assert inp.canary_percentage == 100


# ── PolicyThresholds range validation (hardened in this audit) ────────────────


class TestPolicyThresholds:
    def test_map50_min_above_one_raises(self):
        with pytest.raises(ValidationError, match="map50_min"):
            PolicyThresholds(map50_min=1.5)

    def test_map50_min_negative_raises(self):
        with pytest.raises(ValidationError, match="map50_min"):
            PolicyThresholds(map50_min=-0.1)

    def test_precision_min_above_one_raises(self):
        with pytest.raises(ValidationError, match="precision_min"):
            PolicyThresholds(precision_min=2.0)

    def test_recall_min_above_one_raises(self):
        with pytest.raises(ValidationError, match="recall_min"):
            PolicyThresholds(recall_min=1.1)

    def test_all_zero_ok(self):
        t = PolicyThresholds(map50_min=0.0, map50_95_min=0.0, precision_min=0.0, recall_min=0.0)
        assert t.map50_min == 0.0

    def test_all_one_ok(self):
        t = PolicyThresholds(map50_min=1.0, map50_95_min=1.0, precision_min=1.0, recall_min=1.0)
        assert t.map50_min == 1.0


# ── PromotionPolicy range validation ─────────────────────────────────────────


class TestPromotionPolicyValidation:
    def test_critical_class_recall_min_above_one_raises(self):
        with pytest.raises(ValidationError, match="critical_class_recall_min"):
            PromotionPolicy(critical_class_recall_min=1.1)

    def test_critical_class_recall_min_negative_raises(self):
        with pytest.raises(ValidationError, match="critical_class_recall_min"):
            PromotionPolicy(critical_class_recall_min=-0.01)

    def test_critical_class_recall_min_boundary_values_ok(self):
        p0 = PromotionPolicy(critical_class_recall_min=0.0)
        p1 = PromotionPolicy(critical_class_recall_min=1.0)
        assert p0.critical_class_recall_min == 0.0
        assert p1.critical_class_recall_min == 1.0

    def test_baseline_improvement_negative_raises(self):
        with pytest.raises(ValidationError, match="baseline_improvement_min_map50"):
            PromotionPolicy(baseline_improvement_min_map50=-0.01)

    def test_baseline_improvement_zero_ok(self):
        p = PromotionPolicy(baseline_improvement_min_map50=0.0)
        assert p.baseline_improvement_min_map50 == 0.0


# ── OrchestratorInput step validation (hardened in this audit) ────────────────


class TestOrchestratorInputSteps:
    def _base_kwargs(self):
        return {"workflow_id": "wf_test"}

    def test_invalid_step_raises(self):
        with pytest.raises(ValidationError, match="nonexistent_step"):
            OrchestratorInput(steps=["nonexistent_step"], **self._base_kwargs())

    def test_mixed_valid_invalid_raises(self):
        with pytest.raises(ValidationError, match="bad_step"):
            OrchestratorInput(steps=["training", "bad_step"], **self._base_kwargs())

    def test_valid_steps_ok(self):
        inp = OrchestratorInput(steps=["training", "evaluation"], **self._base_kwargs())
        assert inp.steps == ["training", "evaluation"]

    def test_none_steps_ok(self):
        inp = OrchestratorInput(steps=None, **self._base_kwargs())
        assert inp.steps is None

    def test_all_pipeline_steps_ok(self):
        inp = OrchestratorInput(steps=list(PIPELINE_STEPS), **self._base_kwargs())
        assert set(inp.steps) == set(PIPELINE_STEPS)  # type: ignore[arg-type]


# ── OrchestratorInput split ratio validation ──────────────────────────────────


class TestOrchestratorInputSplitRatios:
    def test_all_zero_ratios_raise(self):
        with pytest.raises(ValidationError, match="all be 0.0"):
            OrchestratorInput(
                workflow_id="wf_test",
                train_ratio=0.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

    def test_only_train_nonzero_ok(self):
        inp = OrchestratorInput(
            workflow_id="wf_test",
            train_ratio=0.9,
            val_ratio=0.0,
            test_ratio=0.0,
        )
        assert inp.train_ratio == 0.9

    def test_default_ratios_ok(self):
        inp = OrchestratorInput(workflow_id="wf_test")
        assert inp.train_ratio + inp.val_ratio + inp.test_ratio > 0


# ── Enum coercion edge cases ──────────────────────────────────────────────────


class TestEnumCoercion:
    def test_model_framework_from_string(self):
        inp = AnnotationInput(images_path="/i", model_path="/m", framework="onnx_only")
        assert inp.framework == ModelFramework.ONNX_ONLY

    def test_model_framework_invalid_raises(self):
        with pytest.raises(ValidationError):
            AnnotationInput(images_path="/i", model_path="/m", framework="pytorch")

    def test_evaluation_mode_from_string(self):
        inp = EvaluationInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            weights_path="/m.pt",
            mode="azure_eval",
        )
        assert inp.mode == EvaluationMode.AZURE_EVAL

    def test_deployment_target_from_string(self):
        inp = DeploymentInput(model_name="m", target="production")
        assert inp.target == DeploymentTarget.PRODUCTION

    def test_workflow_state_all_members(self):
        for state in WorkflowState:
            assert isinstance(state, str)

    def test_monitoring_input_source_invalid_raises(self):
        with pytest.raises(ValidationError):
            MonitoringInput(endpoint_name="ep", source="kafka")  # type: ignore[arg-type]


# ── Serialization round-trips ─────────────────────────────────────────────────


class TestSerializationRoundTrips:
    def _training_config(self):
        return TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN, epochs=50, imgsz=320)

    def test_training_input_round_trip(self):
        inp = TrainingInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            training_config=self._training_config(),
            framework=ModelFramework.YOLO,
        )
        data = json.loads(inp.model_dump_json())
        restored = TrainingInput.model_validate(data)
        assert restored.framework == ModelFramework.YOLO
        assert restored.training_config.epochs == 50
        assert restored.training_config.imgsz == 320

    def test_evaluation_input_round_trip(self):
        inp = EvaluationInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            weights_path="/m.pt",
            framework=ModelFramework.ONNX_ONLY,
        )
        data = json.loads(inp.model_dump_json())
        restored = EvaluationInput.model_validate(data)
        assert restored.framework == ModelFramework.ONNX_ONLY

    def test_annotation_input_round_trip(self):
        inp = AnnotationInput(
            images_path="/img",
            model_path="/m.pt",
            imgsz=320,
            confidence_thresholds=ConfidenceThresholds(auto_candidate=0.85, human_review=0.45),
        )
        data = json.loads(inp.model_dump_json())
        restored = AnnotationInput.model_validate(data)
        assert restored.imgsz == 320
        assert restored.confidence_thresholds.auto_candidate == 0.85

    def test_deployment_input_round_trip(self):
        inp = DeploymentInput(
            model_name="my-model",
            target=DeploymentTarget.PRODUCTION,
            canary_percentage=50,
        )
        data = json.loads(inp.model_dump_json())
        restored = DeploymentInput.model_validate(data)
        assert restored.canary_percentage == 50
        assert restored.target == DeploymentTarget.PRODUCTION

    def test_evaluation_metrics_round_trip(self):
        m = EvaluationMetrics(map50=0.82, map50_95=0.61, precision=0.77, recall=0.73)
        data = json.loads(m.model_dump_json())
        restored = EvaluationMetrics.model_validate(data)
        assert restored.map50 == pytest.approx(0.82)
        assert restored.recall == pytest.approx(0.73)

    def test_promotion_policy_round_trip(self):
        p = PromotionPolicy(
            thresholds=PolicyThresholds(map50_min=0.80, recall_min=0.65),
            critical_class_recall_min=0.70,
            require_improvement_over_baseline=True,
            baseline_improvement_min_map50=0.02,
        )
        data = json.loads(p.model_dump_json())
        restored = PromotionPolicy.model_validate(data)
        assert restored.thresholds.map50_min == pytest.approx(0.80)
        assert restored.critical_class_recall_min == pytest.approx(0.70)
        assert restored.require_improvement_over_baseline is True


# ── evaluate_metrics_against_policy edge cases ────────────────────────────────


class TestEvaluatePolicyEdgeCases:
    def test_all_zeros_fails_all(self):
        from agentic_mlops.contracts.evaluation import EvaluationRecommendation  # noqa: PLC0415
        from agentic_mlops.workflows.policies import evaluate_metrics_against_policy  # noqa: PLC0415

        policy = PromotionPolicy()
        metrics = EvaluationMetrics(map50=0.0, map50_95=0.0, precision=0.0, recall=0.0)
        rec, passed, failed = evaluate_metrics_against_policy(metrics, policy)
        assert rec == EvaluationRecommendation.RETRAIN
        assert len(failed) >= 4

    def test_all_ones_passes_all(self):
        from agentic_mlops.contracts.evaluation import EvaluationRecommendation  # noqa: PLC0415
        from agentic_mlops.workflows.policies import evaluate_metrics_against_policy  # noqa: PLC0415

        policy = PromotionPolicy()
        metrics = EvaluationMetrics(map50=1.0, map50_95=1.0, precision=1.0, recall=1.0)
        rec, passed, failed = evaluate_metrics_against_policy(metrics, policy)
        assert rec == EvaluationRecommendation.PROMOTE_CANDIDATE
        assert not failed

    def test_missing_critical_class_triggers_label_review(self):
        from agentic_mlops.contracts.evaluation import EvaluationRecommendation  # noqa: PLC0415
        from agentic_mlops.workflows.policies import evaluate_metrics_against_policy  # noqa: PLC0415

        policy = PromotionPolicy(critical_classes=["crack"])
        metrics = EvaluationMetrics(map50=1.0, map50_95=1.0, precision=1.0, recall=1.0)
        rec, _passed, failed = evaluate_metrics_against_policy(metrics, policy)
        assert rec == EvaluationRecommendation.NEED_LABEL_REVIEW
        assert any("crack" in f for f in failed)

    def test_only_recall_fails_collect_more_data(self):
        from agentic_mlops.contracts.evaluation import EvaluationRecommendation  # noqa: PLC0415
        from agentic_mlops.workflows.policies import evaluate_metrics_against_policy  # noqa: PLC0415

        policy = PromotionPolicy(thresholds=PolicyThresholds(recall_min=0.95))
        metrics = EvaluationMetrics(map50=0.90, map50_95=0.80, precision=0.85, recall=0.60)
        rec, _passed, _failed = evaluate_metrics_against_policy(metrics, policy)
        assert rec == EvaluationRecommendation.COLLECT_MORE_DATA

    def test_only_precision_fails_review_labels(self):
        from agentic_mlops.contracts.evaluation import EvaluationRecommendation  # noqa: PLC0415
        from agentic_mlops.workflows.policies import evaluate_metrics_against_policy  # noqa: PLC0415

        policy = PromotionPolicy(thresholds=PolicyThresholds(precision_min=0.95))
        metrics = EvaluationMetrics(map50=0.90, map50_95=0.80, precision=0.60, recall=0.85)
        rec, _passed, _failed = evaluate_metrics_against_policy(metrics, policy)
        assert rec == EvaluationRecommendation.REVIEW_LABELS
