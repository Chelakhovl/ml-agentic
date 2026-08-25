"""Tests for agentic_mlops.web.routes — HTTP route handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from agentic_mlops.web.app import create_app  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_state(runs_dir: Path, wf_id: str, status: str = "completed", **extra) -> Path:
    wf_dir = runs_dir / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": wf_id,
        "status": status,
        "current_state": "COMPLETED" if status == "completed" else status.upper(),
        "steps": ["dataset_validation", "training"],
        "completed_steps": ["dataset_validation"] if status == "completed" else [],
        "step_status": {"dataset_validation": "COMPLETED"},
        "step_outputs": {},
        "tags": {},
        "started_at": "2026-01-01T10:00:00+00:00",
        "updated_at": "2026-01-01T10:05:00+00:00",
        **extra,
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return wf_dir


def _write_model(registry_dir: Path, model_name: str, version: int = 1, map50: float = 0.85):
    ver_dir = registry_dir / model_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {"map50": map50, "map50_95": 0.55, "precision": 0.9, "recall": 0.87}
    (ver_dir / "lineage.json").write_text(json.dumps(lineage))
    reg_out = {
        "version": version,
        "sha256": "deadbeef",
        "registered_at": "2026-01-01T10:10:00+00:00",
    }
    (ver_dir / "registration_output.json").write_text(json.dumps(reg_out))
    latest = {
        "model_name": model_name,
        "version": version,
        "registered_at": "2026-01-01T10:10:00+00:00",
        "map50": map50,
    }
    (registry_dir / model_name / "latest.json").write_text(json.dumps(latest))


def _write_dataset(datasets_dir: Path, dataset_name: str, version: int = 1):
    ver_dir = datasets_dir / dataset_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "dataset_name": dataset_name,
        "version": version,
        "hash": "abc123",
        "registered_at": "2026-01-01T10:00:00+00:00",
        "validation_status": "passed",
        "classes": ["scratch", "dent"],
    }
    (ver_dir / "lineage.json").write_text(json.dumps(lineage))
    latest = {
        "dataset_name": dataset_name,
        "version": version,
        "registered_at": "2026-01-01T10:00:00+00:00",
    }
    (datasets_dir / dataset_name / "latest.json").write_text(json.dumps(latest))


def _write_monitoring(runs_dir: Path, wf_id: str, step: str = "monitoring") -> None:
    d = runs_dir / wf_id / "artifacts" / step
    d.mkdir(parents=True, exist_ok=True)
    report = {
        "success": True,
        "status": "completed",
        "endpoint_name": "prod-ep",
        "total_predictions": 100,
        "generated_at": "2026-01-01T12:00:00+00:00",
        "drift_detected": False,
        "recommended_action": "no_action",
        "hard_samples": [],
        "metrics": {"avg_confidence": 0.8, "p95_latency_ms": 40.0},
        "class_distribution": {"scratch": 0.6, "dent": 0.4},
    }
    (d / "monitoring_report.json").write_text(json.dumps(report))


def _write_hard_samples(runs_dir: Path, wf_id: str, step: str = "monitoring", count: int = 3):
    d = runs_dir / wf_id / "artifacts" / step
    d.mkdir(parents=True, exist_ok=True)
    samples = [{"image_id": f"img_{i}", "reason": "low_conf"} for i in range(count)]
    (d / "hard_samples_manifest.json").write_text(json.dumps(samples))


def _write_cost_report(runs_dir: Path, wf_id: str, total: float = 1.50):
    d = runs_dir / wf_id / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    report = {
        "total_cost_usd": total,
        "currency": "USD",
        "generated_at": "2026-01-01T11:00:00+00:00",
        "entries": [{"step": "training", "cost_usd": total}],
    }
    (d / "cost_report.json").write_text(json.dumps(report))


@pytest.fixture()
def setup(tmp_path):
    runs = tmp_path / "runs"
    registry = tmp_path / "models"
    datasets = tmp_path / "datasets"
    for d in (runs, registry, datasets):
        d.mkdir(parents=True, exist_ok=True)
    _write_state(runs, "wf_001", status="completed")
    _write_state(runs, "wf_002", status="running")
    _write_model(registry, "my-model")
    _write_dataset(datasets, "factory-defects")
    app = create_app(
        runs_dir=str(runs),
        registry_dir=str(registry),
        datasets_dir=str(datasets),
    )
    client = TestClient(app, raise_server_exceptions=True)
    return client, runs, registry, datasets


# ── Dashboard ─────────────────────────────────────────────────────────────────


class TestDashboard:
    def test_renders_200(self, setup):
        client, *_ = setup
        resp = client.get("/")
        assert resp.status_code == 200

    def test_shows_workflow_ids(self, setup):
        client, *_ = setup
        resp = client.get("/")
        assert "wf_001" in resp.text
        assert "wf_002" in resp.text

    def test_shows_status_badges(self, setup):
        client, *_ = setup
        resp = client.get("/")
        assert "completed" in resp.text
        assert "running" in resp.text

    def test_empty_runs_shows_empty_state(self, tmp_path):
        runs = tmp_path / "empty_runs"
        runs.mkdir()
        app = create_app(
            runs_dir=str(runs),
            registry_dir=str(tmp_path / "reg"),
            datasets_dir=str(tmp_path / "ds"),
        )
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "No workflow runs" in resp.text

    def test_shows_pagination_controls(self, setup):
        client, *_ = setup
        resp = client.get("/")
        assert "pageSizeSelect" in resp.text

    def test_shows_compare_checkbox_column(self, setup):
        client, *_ = setup
        resp = client.get("/")
        assert "cmp-check" in resp.text


# ── Workflow detail ───────────────────────────────────────────────────────────


class TestWorkflowDetail:
    def test_404_for_unknown(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/no_such_wf")
        assert resp.status_code == 404

    def test_200_for_known(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/wf_001")
        assert resp.status_code == 200

    def test_page_contains_workflow_id(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/wf_001")
        assert "wf_001" in resp.text

    def test_contains_compare_link(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/wf_001")
        assert "Compare with another run" in resp.text

    def test_contains_export_link(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/wf_001")
        assert "Export JSON" in resp.text

    def test_shows_cost_section_when_present(self, setup):
        client, runs, *_ = setup
        _write_cost_report(runs, "wf_001", total=2.75)
        resp = client.get("/workflows/wf_001")
        assert resp.status_code == 200
        assert "2.75" in resp.text or "Cost" in resp.text

    def test_pending_approval_shows_banner(self, setup):
        client, runs, *_ = setup
        _write_state(
            runs, "wf_pend", status="pending_approval", current_state="MODEL_APPROVAL_REQUIRED"
        )
        resp = client.get("/workflows/wf_pend")
        assert "Approval Required" in resp.text


# ── Tags ─────────────────────────────────────────────────────────────────────


class TestWorkflowTags:
    def test_add_tag_redirects(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/wf_001/tags",
            data={"tag_key": "env", "tag_value": "prod"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/workflows/wf_001" in resp.headers["location"]

    def test_remove_tag_redirects(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/wf_001/tags",
            data={"remove_key": "env"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

    def test_invalid_workflow_id_returns_400(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/../evil/tags",
            data={"tag_key": "k", "tag_value": "v"},
            follow_redirects=False,
        )
        assert resp.status_code in (400, 404, 422)

    def test_unknown_workflow_returns_404(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/no_such_wf/tags",
            data={"tag_key": "env", "tag_value": "prod"},
            follow_redirects=False,
        )
        assert resp.status_code in (303, 404)


# ── Compare runs ──────────────────────────────────────────────────────────────


class TestWorkflowCompare:
    def test_compare_page_empty(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/compare")
        assert resp.status_code == 200

    def test_compare_page_with_valid_runs(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/compare?a=wf_001&b=wf_002")
        assert resp.status_code == 200

    def test_compare_with_unknown_shows_error(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/compare?a=wf_001&b=no_such")
        assert resp.status_code == 200
        assert "not found" in resp.text.lower() or "error" in resp.text.lower()


# ── Costs ─────────────────────────────────────────────────────────────────────


class TestCosts:
    def test_costs_page_empty(self, setup):
        client, *_ = setup
        resp = client.get("/costs")
        assert resp.status_code == 200

    def test_costs_page_shows_report(self, setup):
        client, runs, *_ = setup
        _write_cost_report(runs, "wf_001", total=3.14)
        resp = client.get("/costs")
        assert resp.status_code == 200
        assert "3.14" in resp.text or "wf_001" in resp.text


# ── Prune ─────────────────────────────────────────────────────────────────────


class TestPrune:
    def test_prune_preview_form_renders(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/prune")
        assert resp.status_code == 200

    def test_prune_preview_with_keep_last(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/prune?keep_last=5")
        assert resp.status_code == 200

    def test_prune_post_redirects(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/prune",
            data={"keep_last": "10"},
            follow_redirects=False,
        )
        assert resp.status_code == 303


# ── Monitoring ────────────────────────────────────────────────────────────────


class TestMonitoring:
    def test_monitoring_list_empty(self, setup):
        client, *_ = setup
        resp = client.get("/monitoring")
        assert resp.status_code == 200

    def test_monitoring_list_shows_report(self, setup):
        client, runs, *_ = setup
        _write_monitoring(runs, "wf_001")
        resp = client.get("/monitoring")
        assert resp.status_code == 200
        assert "prod-ep" in resp.text

    def test_monitoring_detail_404_for_unknown(self, setup):
        client, *_ = setup
        resp = client.get("/monitoring/no_such_wf/monitoring")
        assert resp.status_code == 404

    def test_monitoring_detail_200_for_known(self, setup):
        client, runs, *_ = setup
        _write_monitoring(runs, "wf_001")
        resp = client.get("/monitoring/wf_001/monitoring")
        assert resp.status_code == 200
        assert "prod-ep" in resp.text


# ── Hard samples ──────────────────────────────────────────────────────────────


class TestHardSamplesPage:
    def test_hard_samples_page_empty(self, setup):
        client, *_ = setup
        resp = client.get("/monitoring/hard-samples")
        assert resp.status_code == 200

    def test_hard_samples_shows_count(self, setup):
        client, runs, *_ = setup
        _write_hard_samples(runs, "wf_001", count=5)
        resp = client.get("/monitoring/hard-samples")
        assert resp.status_code == 200
        assert "5" in resp.text

    def test_hard_samples_shows_workflow_id(self, setup):
        client, runs, *_ = setup
        _write_hard_samples(runs, "wf_001", count=2)
        resp = client.get("/monitoring/hard-samples")
        assert "wf_001" in resp.text


# ── Audit API ─────────────────────────────────────────────────────────────────


class TestAuditApi:
    def test_returns_entries_and_total(self, setup):
        client, runs, *_ = setup
        log_file = runs / "wf_001" / "audit_log.jsonl"
        events = [
            '{"event": "workflow_started", "timestamp": "2026-01-01T10:00:00+00:00"}',
            '{"event": "step_started", "step": "training",'
            ' "timestamp": "2026-01-01T10:01:00+00:00"}',
        ]
        log_file.write_text("\n".join(events))
        resp = client.get("/api/workflows/wf_001/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total" in data
        assert data["total"] == 2

    def test_since_parameter_limits_new_entries(self, setup):
        client, runs, *_ = setup
        log_file = runs / "wf_001" / "audit_log.jsonl"
        events = [
            '{"event": "e1"}',
            '{"event": "e2"}',
            '{"event": "e3"}',
        ]
        log_file.write_text("\n".join(events))
        resp = client.get("/api/workflows/wf_001/audit?since=2")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["entries"]) == 1

    def test_unknown_workflow_returns_empty(self, setup):
        client, *_ = setup
        resp = client.get("/api/workflows/no_such_wf/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["entries"] == []


# ── Status API ────────────────────────────────────────────────────────────────


class TestStatusApi:
    def test_returns_status_json(self, setup):
        client, *_ = setup
        resp = client.get("/api/workflows/wf_001/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_id"] == "wf_001"
        assert data["status"] == "completed"

    def test_404_for_unknown(self, setup):
        client, *_ = setup
        resp = client.get("/api/workflows/no_such_wf/status")
        assert resp.status_code == 404


# ── Export API ────────────────────────────────────────────────────────────────


class TestExportApi:
    def test_returns_json_attachment(self, setup):
        client, *_ = setup
        resp = client.get("/api/workflows/wf_001/export")
        assert resp.status_code == 200
        assert resp.headers.get("content-disposition", "").startswith("attachment")
        data = resp.json()
        assert data["workflow_id"] == "wf_001"

    def test_404_for_unknown(self, setup):
        client, *_ = setup
        resp = client.get("/api/workflows/no_such_wf/export")
        assert resp.status_code == 404


# ── Model registry ────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_list_renders(self, setup):
        client, *_ = setup
        resp = client.get("/registry")
        assert resp.status_code == 200
        assert "my-model" in resp.text

    def test_registry_list_empty(self, tmp_path):
        app = create_app(
            runs_dir=str(tmp_path / "runs"),
            registry_dir=str(tmp_path / "reg"),
            datasets_dir=str(tmp_path / "ds"),
        )
        client = TestClient(app)
        resp = client.get("/registry")
        assert resp.status_code == 200

    def test_model_detail_renders(self, setup):
        client, *_ = setup
        resp = client.get("/registry/my-model")
        assert resp.status_code == 200
        assert "my-model" in resp.text

    def test_model_detail_404_for_unknown(self, setup):
        client, *_ = setup
        resp = client.get("/registry/no-such-model")
        assert resp.status_code == 404

    def test_rank_page_renders(self, setup):
        client, *_ = setup
        resp = client.get("/registry/rank")
        assert resp.status_code == 200

    def test_compare_page_renders(self, setup):
        client, *_ = setup
        resp = client.get("/registry/compare")
        assert resp.status_code == 200


# ── Dataset registry ──────────────────────────────────────────────────────────


class TestDatasets:
    def test_datasets_list_renders(self, setup):
        client, *_ = setup
        resp = client.get("/datasets")
        assert resp.status_code == 200
        assert "factory-defects" in resp.text

    def test_dataset_detail_renders(self, setup):
        client, *_ = setup
        resp = client.get("/datasets/factory-defects")
        assert resp.status_code == 200
        assert "factory-defects" in resp.text

    def test_dataset_detail_404(self, setup):
        client, *_ = setup
        resp = client.get("/datasets/no-such-dataset")
        assert resp.status_code == 404


# ── Resume ────────────────────────────────────────────────────────────────────


class TestResume:
    def test_no_input_json_redirects_with_flash(self, setup):
        client, *_ = setup
        resp = client.post(
            "/workflows/wf_001/resume",
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "wf_001" in resp.headers["location"]

    def test_with_input_json_redirects(self, setup):
        client, runs, *_ = setup
        input_data = {
            "workflow_id": "wf_001",
            "runs_dir": str(runs),
            "steps": ["dataset_validation"],
            "dry_run": True,
            "interactive_approval": False,
            "interactive_training_approval": False,
            "resume": False,
        }
        (runs / "wf_001" / "input.json").write_text(json.dumps(input_data))
        resp = client.post(
            "/workflows/wf_001/resume",
            follow_redirects=False,
        )
        assert resp.status_code == 303


# ── New workflow form ─────────────────────────────────────────────────────────


class TestNewWorkflow:
    def test_get_renders_form(self, setup):
        client, *_ = setup
        resp = client.get("/workflows/new")
        assert resp.status_code == 200
        assert "workflow" in resp.text.lower()


# ── Prometheus metrics ────────────────────────────────────────────────────────


class TestPrometheusMetrics:
    def test_metrics_endpoint_returns_text(self, setup):
        client, *_ = setup
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "mlops_workflows_total" in resp.text

    def test_metrics_reflects_workflow_count(self, setup):
        client, *_ = setup
        resp = client.get("/metrics")
        assert 'status="completed"' in resp.text
        assert 'status="running"' in resp.text
