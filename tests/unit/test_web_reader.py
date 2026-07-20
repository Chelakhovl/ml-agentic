"""Tests for agentic_mlops.web.reader — pure filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.web.reader import (
    get_audit_log,
    get_model_versions,
    get_monitoring_report,
    get_workflow,
    list_models,
    list_monitoring_reports,
    list_workflows,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_state(runs_dir: Path, wf_id: str, **extra) -> Path:
    wf_dir = runs_dir / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": wf_id,
        "status": "completed",
        "current_state": "COMPLETED",
        "steps": ["dataset_validation"],
        "completed_steps": ["dataset_validation"],
        "step_status": {"dataset_validation": "COMPLETED"},
        "step_outputs": {},
        "artifacts": [],
        "started_at": "2026-01-01T10:00:00+00:00",
        "updated_at": "2026-01-01T10:05:00+00:00",
        **extra,
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return wf_dir


def _write_audit(runs_dir: Path, wf_id: str, events: list[dict]) -> None:
    log_file = runs_dir / wf_id / "audit_log.jsonl"
    lines = [json.dumps(e) for e in events]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_registry_model(
    registry_dir: Path, model_name: str, version: int = 1, map50: float = 0.85
):
    ver_dir = registry_dir / model_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "map50": map50,
        "map50_95": 0.55,
        "precision": 0.9,
        "recall": 0.87,
        "approved_by": "test.user",
        "approval_action": "approve_model",
        "approval_timestamp": "2026-01-01T10:10:00+00:00",
    }
    (ver_dir / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    reg_out = {
        "version": version,
        "sha256": "abc123deadbeef",
        "registered_at": "2026-01-01T10:10:00+00:00",
    }
    (ver_dir / "registration_output.json").write_text(json.dumps(reg_out), encoding="utf-8")
    latest = {
        "model_name": model_name,
        "version": version,
        "registry_path": str(ver_dir),
        "registered_at": "2026-01-01T10:10:00+00:00",
        "sha256": "abc123deadbeef",
        "map50": map50,
    }
    (registry_dir / model_name / "latest.json").write_text(json.dumps(latest), encoding="utf-8")


# ── list_workflows ────────────────────────────────────────────────────────────


class TestListWorkflows:
    def test_empty_dir(self, tmp_path):
        assert list_workflows(tmp_path / "runs") == []

    def test_missing_dir(self, tmp_path):
        assert list_workflows(tmp_path / "nonexistent") == []

    def test_returns_sorted_newest_first(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_001", started_at="2026-01-01T08:00:00+00:00")
        _write_state(runs, "wf_002", started_at="2026-01-02T09:00:00+00:00")
        result = list_workflows(runs)
        assert [w["workflow_id"] for w in result] == ["wf_002", "wf_001"]

    def test_skips_corrupt_json(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        bad_dir = runs / "wf_bad"
        bad_dir.mkdir()
        (bad_dir / "state.json").write_text("not valid json", encoding="utf-8")
        _write_state(runs, "wf_good")
        result = list_workflows(runs)
        assert len(result) == 1
        assert result[0]["workflow_id"] == "wf_good"

    def test_includes_all_state_fields(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_full")
        result = list_workflows(runs)
        assert result[0]["status"] == "completed"
        assert "completed_steps" in result[0]


# ── get_workflow ──────────────────────────────────────────────────────────────


class TestGetWorkflow:
    def test_returns_none_for_missing(self, tmp_path):
        assert get_workflow(tmp_path / "runs", "wf_missing") is None

    def test_returns_state_dict(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_abc")
        state = get_workflow(runs, "wf_abc")
        assert state is not None
        assert state["workflow_id"] == "wf_abc"

    def test_returns_none_for_corrupt(self, tmp_path):
        runs = tmp_path / "runs"
        wf_dir = runs / "wf_bad"
        wf_dir.mkdir(parents=True)
        (wf_dir / "state.json").write_text("{bad}", encoding="utf-8")
        assert get_workflow(runs, "wf_bad") is None


# ── get_audit_log ─────────────────────────────────────────────────────────────


class TestGetAuditLog:
    def test_returns_empty_for_missing_file(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_1")
        assert get_audit_log(runs, "wf_1") == []

    def test_newest_first(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_1")
        events = [
            {"event": "workflow_started", "timestamp": "2026-01-01T10:00:00+00:00"},
            {"event": "step_started", "step": "training", "timestamp": "2026-01-01T10:01:00+00:00"},
            {
                "event": "workflow_finished",
                "status": "completed",
                "timestamp": "2026-01-01T10:02:00+00:00",
            },
        ]
        _write_audit(runs, "wf_1", events)
        log = get_audit_log(runs, "wf_1")
        assert log[0]["event"] == "workflow_finished"
        assert log[-1]["event"] == "workflow_started"

    def test_skips_blank_and_malformed_lines(self, tmp_path):
        runs = tmp_path / "runs"
        _write_state(runs, "wf_2")
        log_file = runs / "wf_2" / "audit_log.jsonl"
        log_file.write_text(
            '{"event": "ok"}\n\nnot-json\n{"event": "also-ok"}',
            encoding="utf-8",
        )
        log = get_audit_log(runs, "wf_2")
        assert len(log) == 2


# ── list_models ───────────────────────────────────────────────────────────────


class TestListModels:
    def test_empty_registry(self, tmp_path):
        assert list_models(tmp_path / "reg") == []

    def test_returns_model_list(self, tmp_path):
        reg = tmp_path / "reg"
        _write_registry_model(reg, "detector-a", map50=0.85)
        _write_registry_model(reg, "detector-b", map50=0.72)
        models = list_models(reg)
        names = {m["model_name"] for m in models}
        assert names == {"detector-a", "detector-b"}

    def test_sorted_newest_first(self, tmp_path):
        reg = tmp_path / "reg"
        _write_registry_model(reg, "old-model")
        # Overwrite latest.json with earlier timestamp
        latest = json.loads((reg / "old-model" / "latest.json").read_text())
        latest["registered_at"] = "2025-01-01T00:00:00+00:00"
        (reg / "old-model" / "latest.json").write_text(json.dumps(latest))

        _write_registry_model(reg, "new-model")
        models = list_models(reg)
        assert models[0]["model_name"] == "new-model"

    def test_skips_corrupt_latest(self, tmp_path):
        reg = tmp_path / "reg"
        _write_registry_model(reg, "good-model")
        bad = reg / "bad-model"
        bad.mkdir(parents=True)
        (bad / "latest.json").write_text("oops", encoding="utf-8")
        models = list_models(reg)
        assert len(models) == 1
        assert models[0]["model_name"] == "good-model"


# ── get_model_versions ────────────────────────────────────────────────────────


class TestGetModelVersions:
    def test_returns_empty_for_unknown_model(self, tmp_path):
        assert get_model_versions(tmp_path / "reg", "ghost") == []

    def test_returns_versions_sorted_desc(self, tmp_path):
        reg = tmp_path / "reg"
        _write_registry_model(reg, "yolo", version=1, map50=0.7)
        _write_registry_model(reg, "yolo", version=2, map50=0.82)
        versions = get_model_versions(reg, "yolo")
        assert [v["version"] for v in versions] == [2, 1]

    def test_lineage_fields_present(self, tmp_path):
        reg = tmp_path / "reg"
        _write_registry_model(reg, "yolo", version=1, map50=0.88)
        versions = get_model_versions(reg, "yolo")
        assert versions[0]["lineage"]["map50"] == pytest.approx(0.88)
        assert versions[0]["registration"]["sha256"] == "abc123deadbeef"

    def test_missing_lineage_yields_empty_dict(self, tmp_path):
        reg = tmp_path / "reg"
        ver_dir = reg / "yolo" / "versions" / "1"
        ver_dir.mkdir(parents=True)
        # No lineage.json written
        versions = get_model_versions(reg, "yolo")
        assert versions[0]["lineage"] == {}


# ── list_monitoring_reports ───────────────────────────────────────────────────


class TestListMonitoringReports:
    def _write_report(self, runs_dir: Path, wf_id: str, step: str, **extra) -> None:
        report_dir = runs_dir / wf_id / "artifacts" / step
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "success": True,
            "status": "completed",
            "endpoint_name": "test-ep",
            "total_predictions": 100,
            "generated_at": "2026-01-01T12:00:00+00:00",
            "drift_detected": False,
            "recommended_action": "no_action",
            **extra,
        }
        (report_dir / "monitoring_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_empty_runs_dir(self, tmp_path):
        assert list_monitoring_reports(tmp_path / "runs") == []

    def test_finds_report(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_mon", "monitoring")
        reports = list_monitoring_reports(runs)
        assert len(reports) == 1
        assert reports[0]["endpoint_name"] == "test-ep"
        assert reports[0]["_workflow_id"] == "wf_mon"

    def test_multiple_reports_sorted_newest_first(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_old", "monitoring", generated_at="2026-01-01T08:00:00+00:00")
        self._write_report(runs, "wf_new", "monitoring", generated_at="2026-01-02T08:00:00+00:00")
        reports = list_monitoring_reports(runs)
        assert reports[0]["_workflow_id"] == "wf_new"

    def test_skips_corrupt_report(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_good", "monitoring")
        bad_dir = runs / "wf_bad" / "artifacts" / "monitoring"
        bad_dir.mkdir(parents=True)
        (bad_dir / "monitoring_report.json").write_text("INVALID", encoding="utf-8")
        reports = list_monitoring_reports(runs)
        assert len(reports) == 1

    def test_num_hard_samples_computed_from_list(self, tmp_path):
        runs = tmp_path / "runs"
        hard = [{"image_id": "a", "reason": "low_conf"}, {"image_id": "b", "reason": "zero_det"}]
        self._write_report(runs, "wf_hs", "monitoring", hard_samples=hard)
        reports = list_monitoring_reports(runs)
        assert reports[0]["num_hard_samples"] == 2

    def test_num_hard_samples_zero_when_empty(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_zero", "monitoring", hard_samples=[])
        reports = list_monitoring_reports(runs)
        assert reports[0]["num_hard_samples"] == 0


# ── get_monitoring_report ─────────────────────────────────────────────────────


class TestGetMonitoringReport:
    def _write_report(
        self, runs_dir: Path, wf_id: str, step: str, hard_samples=None, **extra
    ) -> Path:
        report_dir = runs_dir / wf_id / "artifacts" / step
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "success": True,
            "status": "completed",
            "endpoint_name": "prod-ep",
            "total_predictions": 50,
            "generated_at": "2026-06-01T12:00:00+00:00",
            "metrics": {"avg_confidence": 0.72, "p95_latency_ms": 45.0},
            "class_distribution": {"scratch": 0.6, "dent": 0.4},
            "drift_detected": False,
            "recommended_action": "no_action",
            "hard_samples": hard_samples or [],
            **extra,
        }
        (report_dir / "monitoring_report.json").write_text(
            __import__("json").dumps(report), encoding="utf-8"
        )
        return report_dir

    def test_returns_none_for_missing_report(self, tmp_path):
        assert get_monitoring_report(tmp_path / "runs", "wf_x", "monitoring") is None

    def test_returns_report_with_annotations(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_01", "monitoring")
        report = get_monitoring_report(runs, "wf_01", "monitoring")
        assert report is not None
        assert report["endpoint_name"] == "prod-ep"
        assert report["_workflow_id"] == "wf_01"
        assert report["_step"] == "monitoring"

    def test_hard_samples_embedded_in_report(self, tmp_path):
        runs = tmp_path / "runs"
        hs = [
            {"image_id": "img_1", "reason": "low_conf", "num_detections": 1, "min_confidence": 0.2}
        ]
        self._write_report(runs, "wf_02", "monitoring", hard_samples=hs)
        report = get_monitoring_report(runs, "wf_02", "monitoring")
        assert len(report["hard_samples"]) == 1
        assert report["hard_samples"][0]["image_id"] == "img_1"

    def test_falls_back_to_hard_samples_manifest(self, tmp_path):
        runs = tmp_path / "runs"
        report_dir = self._write_report(runs, "wf_03", "monitoring", hard_samples=[])
        manifest = [{"image_id": "m_01", "reason": "zero_detections", "num_detections": 0}]
        (report_dir / "hard_samples_manifest.json").write_text(
            __import__("json").dumps(manifest), encoding="utf-8"
        )
        report = get_monitoring_report(runs, "wf_03", "monitoring")
        assert len(report["hard_samples"]) == 1
        assert report["hard_samples"][0]["image_id"] == "m_01"

    def test_does_not_overwrite_existing_hard_samples_with_manifest(self, tmp_path):
        runs = tmp_path / "runs"
        hs = [{"image_id": "embedded", "reason": "low_conf", "num_detections": 2}]
        report_dir = self._write_report(runs, "wf_04", "monitoring", hard_samples=hs)
        # manifest present but should be ignored when report already has hard_samples
        manifest = [{"image_id": "from_manifest", "reason": "zero_detections"}]
        (report_dir / "hard_samples_manifest.json").write_text(
            __import__("json").dumps(manifest), encoding="utf-8"
        )
        report = get_monitoring_report(runs, "wf_04", "monitoring")
        assert report["hard_samples"][0]["image_id"] == "embedded"

    def test_returns_none_for_corrupt_json(self, tmp_path):
        runs = tmp_path / "runs"
        report_dir = runs / "wf_bad" / "artifacts" / "monitoring"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "monitoring_report.json").write_text("CORRUPT", encoding="utf-8")
        assert get_monitoring_report(runs, "wf_bad", "monitoring") is None

    def test_custom_step_name(self, tmp_path):
        runs = tmp_path / "runs"
        self._write_report(runs, "wf_05", "custom_monitor")
        report = get_monitoring_report(runs, "wf_05", "custom_monitor")
        assert report is not None
        assert report["_step"] == "custom_monitor"
