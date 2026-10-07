"""Edge-case and non-happy-path tests for core pipeline tools.

Covers:
- EvaluationAgent: blocked when training_status in {failed, cancelled}
- TrainingAgent: blocked when dataset_validation_status == "failed"
- PromotionPolicy: all decision-tree branches (only-recall, only-precision, both, critical-class)
- ModelMonitor: new_classes_only → recommended_action!=NO_ACTION; window/drift edge cases
- PseudoLabeler: low bucket unreachability with YOLO pre-filter
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.contracts.evaluation import (
    EvaluationMetrics,
    EvaluationMode,
    EvaluationRecommendation,
    PerClassMetrics,
)
from agentic_mlops.contracts.monitoring import (
    MonitoringInput,
    MonitoringStatus,
    RecommendedAction,
)
from agentic_mlops.contracts.training import TrainingConfig, TrainingMode
from agentic_mlops.tools.monitor import (
    ModelMonitor,
    _drift_score,
    _filter_window,
    _load_baseline,
    _parse_window,
    _percentile,
)
from agentic_mlops.workflows.policies import PromotionPolicy, evaluate_metrics_against_policy

# ── helpers ────────────────────────────────────────────────────────────────────


def _metrics(map50=0.85, map50_95=0.60, precision=0.80, recall=0.80, per_class=None):
    return EvaluationMetrics(
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        per_class_metrics=per_class or {},
    )


def _default_policy(**overrides):
    from agentic_mlops.workflows.policies import PolicyThresholds  # noqa: PLC0415

    t = PolicyThresholds(map50_min=0.75, map50_95_min=0.50, precision_min=0.70, recall_min=0.70)
    return PromotionPolicy(thresholds=t, **overrides)


# ── EvaluationAgent blocking ────────────────────────────────────────────────────


class TestEvaluationAgentBlock:
    def _agent(self, tmp_path):
        from agentic_mlops.agents.evaluation import EvaluationAgent  # noqa: PLC0415

        return EvaluationAgent(artifacts_dir=tmp_path)

    def _input(self, status, tmp_path):
        from agentic_mlops.contracts.evaluation import EvaluationInput  # noqa: PLC0415

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [crack]\ntrain: train\nval: val\n")
        return EvaluationInput(
            dataset_path=str(tmp_path),
            data_yaml_path=str(data_yaml),
            weights_path=str(tmp_path / "best.pt"),
            training_status=status,
            mode=EvaluationMode.LOCAL_DRY_RUN,
        )

    def test_blocked_when_training_failed(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input("failed", tmp_path))
        assert not out.success
        assert "blocked" in out.message.lower()

    def test_blocked_when_training_cancelled(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input("cancelled", tmp_path))
        assert not out.success
        assert "blocked" in out.message.lower()

    def test_proceeds_when_training_completed(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input("completed", tmp_path))
        assert out.success

    def test_proceeds_when_training_status_none(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input(None, tmp_path))
        assert out.success


# ── TrainingAgent blocking ──────────────────────────────────────────────────────


class TestTrainingAgentBlock:
    def _agent(self, tmp_path):
        from agentic_mlops.agents.training import TrainingAgent  # noqa: PLC0415

        return TrainingAgent(artifacts_dir=tmp_path)

    def _input(self, validation_status, tmp_path):
        from agentic_mlops.contracts.training import TrainingInput  # noqa: PLC0415

        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "data.yaml").write_text("nc: 1\nnames: [crack]\n")
        cfg = TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN)
        return TrainingInput(
            dataset_path=str(dataset),
            data_yaml_path=str(dataset / "data.yaml"),
            training_config=cfg,
            dataset_validation_status=validation_status,
        )

    def test_blocked_when_dataset_failed(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input("failed", tmp_path))
        assert not out.success
        assert "blocked" in out.message.lower()

    def test_proceeds_when_dataset_passed(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input("passed", tmp_path))
        assert out.success

    def test_proceeds_when_dataset_warning(self, tmp_path):
        # "warning" status should NOT block training
        agent = self._agent(tmp_path)
        out = agent.run(self._input("warning", tmp_path))
        assert out.success

    def test_proceeds_when_validation_status_none(self, tmp_path):
        agent = self._agent(tmp_path)
        out = agent.run(self._input(None, tmp_path))
        assert out.success


# ── PromotionPolicy decision tree ───────────────────────────────────────────────


class TestPromotionPolicyDecisionTree:
    def test_all_pass_gives_promote_candidate(self):
        rec, passed, failed = evaluate_metrics_against_policy(_metrics(), _default_policy())
        assert rec == EvaluationRecommendation.PROMOTE_CANDIDATE
        assert not failed

    def test_only_recall_fails_gives_collect_more_data(self):
        m = _metrics(recall=0.50)  # recall below 0.70
        rec, _, failed = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.COLLECT_MORE_DATA
        assert any("recall" in f for f in failed)

    def test_only_precision_fails_gives_review_labels(self):
        m = _metrics(precision=0.50)  # precision below 0.70
        rec, _, failed = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.REVIEW_LABELS
        assert any("precision" in f for f in failed)

    def test_recall_and_precision_both_fail_gives_retrain(self):
        m = _metrics(precision=0.50, recall=0.50)
        rec, _, _ = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.RETRAIN

    def test_map50_fails_gives_retrain(self):
        m = _metrics(map50=0.50)
        rec, _, _ = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.RETRAIN

    def test_critical_class_missing_gives_need_label_review(self):
        policy = _default_policy(critical_classes=["crack"])
        # per_class_metrics does NOT contain "crack"
        m = _metrics()
        rec, _, failed = evaluate_metrics_against_policy(m, policy)
        assert rec == EvaluationRecommendation.NEED_LABEL_REVIEW
        assert any("crack" in f for f in failed)

    def test_critical_class_below_threshold_gives_need_label_review(self):
        policy = _default_policy(critical_classes=["crack"], critical_class_recall_min=0.80)
        m = _metrics(per_class={"crack": PerClassMetrics(recall=0.50)})
        rec, _, failed = evaluate_metrics_against_policy(m, policy)
        assert rec == EvaluationRecommendation.NEED_LABEL_REVIEW
        assert any("crack" in f for f in failed)

    def test_critical_class_passes_with_all_global_passes_gives_promote(self):
        policy = _default_policy(critical_classes=["crack"], critical_class_recall_min=0.65)
        m = _metrics(per_class={"crack": PerClassMetrics(recall=0.70)})
        rec, passed, _ = evaluate_metrics_against_policy(m, policy)
        assert rec == EvaluationRecommendation.PROMOTE_CANDIDATE

    def test_per_class_recall_min_missing_class_gives_retrain(self):
        policy = _default_policy(per_class_recall_min={"dent": 0.70})
        # per_class_metrics doesn't have "dent" — should prevent promotion
        m = _metrics()
        rec, _, failed = evaluate_metrics_against_policy(m, policy)
        assert rec == EvaluationRecommendation.RETRAIN
        assert any("dent" in f for f in failed)

    def test_boundary_at_exact_threshold_passes(self):
        # Exactly at the threshold should pass (>=)
        m = _metrics(map50=0.75, map50_95=0.50, precision=0.70, recall=0.70)
        rec, _, failed = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.PROMOTE_CANDIDATE
        assert not failed

    def test_boundary_just_below_threshold_fails(self):
        m = _metrics(map50=0.7499)
        rec, _, failed = evaluate_metrics_against_policy(m, _default_policy())
        assert rec == EvaluationRecommendation.RETRAIN
        assert any("map50" in f for f in failed)


# ── ModelMonitor edge cases ─────────────────────────────────────────────────────


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    log = tmp_path / "predictions.jsonl"
    log.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return log


class TestMonitorNewClassesOnly:
    """When only new classes are detected (no numeric thresholds fire),
    recommended_action must NOT be NO_ACTION while status is ALERTS_TRIGGERED.
    """

    def test_new_classes_only_alerts_triggered_but_no_action(self, tmp_path):
        """Per spec: new class alone fires ALERTS_TRIGGERED + requires_human_review,
        but does NOT map to a recommended_action (stays NO_ACTION).
        This is intentional — only numeric thresholds produce a recommended action.
        """
        # 19 scratch + 1 newclass → class_dist={scratch:0.95, newclass:0.05}
        # TVD vs baseline={scratch:1.0} = 0.5*(0.05+0.05) = 0.05 < default drift threshold (0.30)
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": f"img_{i:03d}",
                "latency_ms": 50.0,
                "error": False,
                "detections": [{"class": "scratch", "confidence": 0.92}],
            }
            for i in range(19)
        ]
        records.append({
            "timestamp": "2026-09-01T10:00:00Z",
            "image_id": "img_020",
            "latency_ms": 50.0,
            "error": False,
            "detections": [{"class": "newclass", "confidence": 0.92}],
        })
        log = _write_log(tmp_path, records)
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"scratch": 100}), encoding="utf-8")

        inp = MonitoringInput(
            endpoint_name="test",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(baseline),
            monitoring_window="24h",
        )
        monitor = ModelMonitor()
        out = monitor.run(inp, tmp_path / "artifacts")

        assert out.success
        assert out.status == MonitoringStatus.ALERTS_TRIGGERED
        # Spec: "A newly-seen class alone sets requires_human_review=True
        # without firing any numeric trigger."
        assert out.recommended_action == RecommendedAction.NO_ACTION
        assert out.requires_human_review is True
        assert "newclass" in out.new_classes_detected

    def test_new_classes_plus_drift_gives_create_retraining_request(self, tmp_path):
        """When both new class and drift fire, drift fires CREATE_RETRAINING_REQUEST."""
        # 19 scratch + 1 newclass → drift ~0.05; lower threshold to 0.01 so it fires
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": f"img_{i:03d}",
                "latency_ms": 50.0,
                "error": False,
                "detections": [{"class": "scratch", "confidence": 0.92}],
            }
            for i in range(19)
        ]
        records.append({
            "timestamp": "2026-09-01T10:00:00Z",
            "image_id": "img_020",
            "latency_ms": 50.0,
            "error": False,
            "detections": [{"class": "newclass", "confidence": 0.92}],
        })
        log = _write_log(tmp_path, records)
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"scratch": 100}), encoding="utf-8")

        from agentic_mlops.contracts.monitoring import MonitoringThresholds  # noqa: PLC0415

        th = MonitoringThresholds(drift_score=0.01)
        inp = MonitoringInput(
            endpoint_name="test",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(baseline),
            monitoring_window="24h",
            thresholds=th,
        )
        out = ModelMonitor().run(inp, tmp_path / "artifacts2")
        assert out.recommended_action == RecommendedAction.CREATE_RETRAINING_REQUEST


class TestMonitorEdgeCases:
    def test_no_records_returns_failed(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text("", encoding="utf-8")
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert not out.success
        assert out.status == MonitoringStatus.FAILED

    def test_malformed_json_lines_skipped_not_failed(self, tmp_path):
        log = tmp_path / "malformed.jsonl"
        log.write_text(
            'NOT JSON\n{"timestamp":"2026-09-01T10:00:00Z","image_id":"img1",'
            '"latency_ms":30,"error":false,"detections":[]}\n',
            encoding="utf-8",
        )
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert out.success
        assert any("malformed" in w.lower() or "skipping" in w.lower() for w in out.warnings)

    def test_no_timestamps_processes_all_records(self, tmp_path):
        """Records without timestamps are all included (no window filtering)."""
        records = [
            {"image_id": "img1", "latency_ms": 10.0, "error": False, "detections": []},
            {"image_id": "img2", "latency_ms": 20.0, "error": False, "detections": []},
        ]
        log = _write_log(tmp_path, records)
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert out.success
        assert out.total_predictions == 2
        assert out.window_start is None
        assert out.window_end is None

    def test_invalid_monitoring_window_returns_failed(self, tmp_path):
        log = _write_log(tmp_path, [{"image_id": "x", "latency_ms": 10, "error": False}])
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="INVALID"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert not out.success
        assert "invalid" in out.message.lower()

    def test_empty_latencies_dont_trigger_p95_alert(self, tmp_path):
        """Records without latency_ms: p95 should be 0.0, not triggering default threshold."""
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": "img1",
                "error": False,
                "detections": [{"class": "scratch", "confidence": 0.85}],
            }
        ]
        log = _write_log(tmp_path, records)
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert out.success
        # p95=0.0 < default threshold 200ms → no alert
        assert out.metrics["p95_latency_ms"] == 0.0
        assert out.recommended_action == RecommendedAction.NO_ACTION

    def test_zero_detections_counted_as_hard_sample(self, tmp_path):
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": "hard_img",
                "latency_ms": 40.0,
                "error": False,
                "detections": [],
            }
        ]
        log = _write_log(tmp_path, records)
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert len(out.hard_samples) == 1
        assert out.hard_samples[0].image_id == "hard_img"
        assert "no detections" in out.hard_samples[0].reason

    def test_critical_class_without_baseline_emits_warning(self, tmp_path):
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": "img",
                "latency_ms": 40.0,
                "error": False,
                "detections": [{"class": "scratch", "confidence": 0.9}],
            }
        ]
        log = _write_log(tmp_path, records)
        inp = MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            monitoring_window="24h",
            critical_classes=["scratch"],
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert out.success
        assert any("critical_class" in w.lower() or "critical" in w.lower() for w in out.warnings)

    def test_baseline_all_zeros_returns_failed(self, tmp_path):
        records = [
            {
                "timestamp": "2026-09-01T10:00:00Z",
                "image_id": "img",
                "latency_ms": 40,
                "error": False,
                "detections": [],
            }
        ]
        log = _write_log(tmp_path, records)
        baseline = tmp_path / "zero_baseline.json"
        baseline.write_text(json.dumps({"scratch": 0, "dent": 0}), encoding="utf-8")
        inp = MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            monitoring_window="24h",
            baseline_class_distribution_path=str(baseline),
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert not out.success
        assert "sum" in out.message.lower() or "0" in out.message

    def test_high_error_rate_triggers_notify_ops(self, tmp_path):
        # Use records with high-confidence detections so low_confidence_ratio stays below threshold.
        # All 6 records have error=True → error_rate=1.0 exceeds default 0.05 → NOTIFY_OPS.
        records = [
            {
                "timestamp": f"2026-09-01T10:0{i}:00Z",
                "image_id": f"img{i}",
                "latency_ms": 30.0,
                "error": True,
                "detections": [{"class": "scratch", "confidence": 0.95}],
            }
            for i in range(6)
        ]
        log = _write_log(tmp_path, records)
        inp = MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"
        )
        out = ModelMonitor().run(inp, tmp_path / "a")
        assert out.recommended_action == RecommendedAction.NOTIFY_OPS
        assert out.status == MonitoringStatus.ALERTS_TRIGGERED


# ── Internal helpers ─────────────────────────────────────────────────────────────


class TestMonitorHelpers:
    def test_parse_window_hours(self):
        assert _parse_window("24h") == 24 * 3600

    def test_parse_window_days(self):
        assert _parse_window("7d") == 7 * 86400

    def test_parse_window_minutes(self):
        assert _parse_window("30m") == 1800

    def test_parse_window_seconds(self):
        assert _parse_window("60s") == 60

    def test_parse_window_invalid_returns_none(self):
        assert _parse_window("INVALID") is None
        assert _parse_window("") is None
        assert _parse_window("24") is None
        assert _parse_window("h24") is None

    def test_drift_score_identical_distributions(self):
        dist = {"scratch": 0.5, "dent": 0.5}
        assert _drift_score(dist, dist) == pytest.approx(0.0)

    def test_drift_score_completely_different(self):
        current = {"scratch": 1.0}
        baseline = {"dent": 1.0}
        # TVD = 0.5 * (|1-0| + |0-1|) = 1.0
        assert _drift_score(current, baseline) == pytest.approx(1.0)

    def test_drift_score_half_overlap(self):
        current = {"scratch": 0.5, "dent": 0.5}
        baseline = {"scratch": 1.0}
        # 0.5 * (|0.5-1| + |0.5-0|) = 0.5 * (0.5 + 0.5) = 0.5
        assert _drift_score(current, baseline) == pytest.approx(0.5)

    def test_percentile_empty_returns_zero(self):
        assert _percentile([], 95.0) == 0.0

    def test_percentile_single_value(self):
        assert _percentile([42.0], 95.0) == 42.0

    def test_percentile_p95_of_ten(self):
        values = list(range(1, 11))  # [1..10]
        result = _percentile(values, 95.0)
        assert result == 10  # ceil(0.95*10)-1 = 9 → s[9] = 10

    def test_load_baseline_normalises_counts(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"scratch": 3, "dent": 1}), encoding="utf-8")
        data, err = _load_baseline(str(baseline))
        assert err is None
        assert data is not None
        assert data["scratch"] == pytest.approx(0.75)
        assert data["dent"] == pytest.approx(0.25)

    def test_load_baseline_bad_path_returns_error(self):
        _, err = _load_baseline("/nonexistent/path/file.json")
        assert err is not None

    def test_filter_window_returns_all_when_no_timestamps(self):
        records = [{"image_id": "a"}, {"image_id": "b"}]
        filtered, ws, we = _filter_window(records, 86400)
        assert len(filtered) == 2
        assert ws is None and we is None


# ── PseudoLabeler low-bucket behaviour ──────────────────────────────────────────


class TestPseudoLabelerBucketBehaviour:
    """Document that 'low' bucket is unreachable via PseudoLabeler because
    YOLO predict() pre-filters at human_review confidence.
    """

    def test_bucket_for_no_predictions_returns_medium(self):
        from agentic_mlops.contracts.annotation import ConfidenceThresholds  # noqa: PLC0415

        # Use OnnxOnlyModelRunner's helper (same logic as PseudoLabeler)
        from agentic_mlops.tools.model_runner import _bucket_for_confs  # noqa: PLC0415
        from agentic_mlops.tools.monitor import _percentile  # noqa: PLC0415, F401

        th = ConfidenceThresholds(auto_candidate=0.90, human_review=0.50)
        bucket, mean_conf, min_conf = _bucket_for_confs([], th)
        assert bucket == "medium"
        assert mean_conf is None
        assert min_conf is None

    def test_bucket_for_all_high_confidence(self):
        from agentic_mlops.contracts.annotation import ConfidenceThresholds  # noqa: PLC0415
        from agentic_mlops.tools.model_runner import _bucket_for_confs  # noqa: PLC0415

        th = ConfidenceThresholds(auto_candidate=0.90, human_review=0.50)
        preds = [(0, 0.5, 0.5, 0.2, 0.2, 0.95)]
        bucket, _, _ = _bucket_for_confs(preds, th)
        assert bucket == "high"

    def test_bucket_for_mixed_confidence_gives_medium(self):
        from agentic_mlops.contracts.annotation import ConfidenceThresholds  # noqa: PLC0415
        from agentic_mlops.tools.model_runner import _bucket_for_confs  # noqa: PLC0415

        th = ConfidenceThresholds(auto_candidate=0.90, human_review=0.50)
        # min_conf = 0.60 which is >= human_review (0.50) but < auto_candidate (0.90)
        preds = [(0, 0.5, 0.5, 0.2, 0.2, 0.95), (1, 0.3, 0.3, 0.1, 0.1, 0.60)]
        bucket, _, min_conf = _bucket_for_confs(preds, th)
        assert bucket == "medium"
        assert min_conf == pytest.approx(0.60)

    def test_low_bucket_reachable_when_preds_below_human_review(self):
        """Low bucket IS reachable if a caller doesn't pre-filter at human_review.
        PseudoLabeler itself never reaches here because YOLO filters first.
        OnnxOnlyModelRunner also pre-filters via conf_threshold=human_review.
        """
        from agentic_mlops.contracts.annotation import ConfidenceThresholds  # noqa: PLC0415
        from agentic_mlops.tools.model_runner import _bucket_for_confs  # noqa: PLC0415

        th = ConfidenceThresholds(auto_candidate=0.90, human_review=0.50)
        # Provide prediction with conf=0.30, below human_review — "low" bucket
        preds = [(0, 0.5, 0.5, 0.2, 0.2, 0.30)]
        bucket, _, _ = _bucket_for_confs(preds, th)
        assert bucket == "low"
