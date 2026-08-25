"""Unit tests for the Monitoring Agent, monitor tool, and CLI command.

Coverage matrix:
    1.  predictions_log_path not found -> failed
    2.  Invalid monitoring_window -> failed
    3.  Empty log file -> failed
    4.  Malformed JSON line is skipped with a warning, valid lines still processed
    5.  Window filtering excludes records outside the window
    6.  Zero-detection image counts as a hard sample
    7.  Low-confidence detection counts as a hard sample
    8.  low_confidence_ratio trigger -> NEED_MORE_DATA, requires_human_review
    9.  p95_latency_ms trigger -> NOTIFY_OPS
   10.  error_rate trigger -> NOTIFY_OPS, requires_human_review
   11.  Drift score computed from baseline -> CREATE_RETRAINING_REQUEST
   12.  critical_class_drop trigger takes priority -> MODEL_REVIEW
   13.  New class alone (no threshold breach) -> requires_human_review, NO_ACTION
   14.  No baseline given -> drift/critical-class checks skipped, metrics default to 0
   15.  Unreadable baseline_class_distribution_path -> failed
   16.  No triggers fired -> COMPLETED / NO_ACTION
   17.  hard_samples_manifest.json content matches computed hard samples
   18.  MonitoringAgent writes monitoring_report.json / .md
   19.  MonitoringAgent logs to MLflow only on success
   20.  CLI: monitor succeeds (exit 0)
   21.  CLI: monitor exits 1 on missing predictions_log
   22.  CLI: monitor exits 1 on invalid monitoring_window
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_mlops.agents.monitoring import MonitoringAgent
from agentic_mlops.contracts.monitoring import (
    MonitoringInput,
    MonitoringStatus,
    RecommendedAction,
)
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.monitor import ModelMonitor

# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_log(tmp_path: Path, records: list[dict], name: str = "predictions.jsonl") -> Path:
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _rec(
    image_id: str,
    timestamp: str,
    confidences: list[float] | None = None,
    latency_ms: float = 50.0,
    error: bool = False,
    cls: str = "dent",
) -> dict:
    detections = [{"class": cls, "confidence": c} for c in (confidences or [])]
    return {
        "image_id": image_id,
        "timestamp": timestamp,
        "latency_ms": latency_ms,
        "error": error,
        "detections": detections,
    }


# ── 1-3. Structural failures ────────────────────────────────────────────────────


def test_predictions_log_not_found_fails(tmp_path: Path) -> None:
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(tmp_path / "nope.jsonl")),
        tmp_path / "out",
    )
    assert result.success is False
    assert result.status == MonitoringStatus.FAILED


def test_invalid_monitoring_window_fails(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.9])])
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), monitoring_window="bogus"
        ),
        tmp_path / "out",
    )
    assert result.success is False
    assert result.status == MonitoringStatus.FAILED
    assert "Invalid monitoring_window" in result.message


def test_empty_log_file_fails(tmp_path: Path) -> None:
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.success is False
    assert result.status == MonitoringStatus.FAILED


# ── 4. Malformed lines ──────────────────────────────────────────────────────────


def test_malformed_json_line_skipped_with_warning(tmp_path: Path) -> None:
    log = tmp_path / "predictions.jsonl"
    log.write_text(
        "{not valid json\n" + json.dumps(_rec("a", "2026-07-10T12:00:00Z", [0.9])) + "\n",
        encoding="utf-8",
    )
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.success is True
    assert result.total_predictions == 1
    assert any("malformed JSON" in w for w in result.warnings)


# ── 5. Window filtering ─────────────────────────────────────────────────────────


def test_window_filtering_excludes_old_records(tmp_path: Path) -> None:
    records = [
        _rec("old", "2026-07-01T00:00:00Z", [0.9]),
        _rec("recent", "2026-07-10T12:00:00Z", [0.9]),
    ]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log), monitoring_window="24h"),
        tmp_path / "out",
    )
    assert result.success is True
    assert result.total_predictions == 1


# ── 6-7. Hard sample mining ──────────────────────────────────────────────────────


def test_zero_detection_image_is_hard_sample(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [])])
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert len(result.hard_samples) == 1
    assert result.hard_samples[0].reason == "no detections"


def test_low_confidence_detection_is_hard_sample(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.2])])
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep", predictions_log_path=str(log), low_confidence_threshold=0.5
        ),
        tmp_path / "out",
    )
    assert len(result.hard_samples) == 1
    assert result.hard_samples[0].min_confidence == 0.2


# ── 8-10. Threshold triggers ────────────────────────────────────────────────────


def test_low_confidence_ratio_trigger_fires(tmp_path: Path) -> None:
    records = [_rec(f"low{i}", "2026-07-10T12:00:00Z", [0.1]) for i in range(3)]
    records += [_rec(f"high{i}", "2026-07-10T12:00:00Z", [0.9]) for i in range(7)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.status == MonitoringStatus.ALERTS_TRIGGERED
    assert result.recommended_action == RecommendedAction.NEED_MORE_DATA
    assert result.requires_human_review is True


def test_p95_latency_trigger_fires(tmp_path: Path) -> None:
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.9], latency_ms=300.0) for i in range(10)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.recommended_action == RecommendedAction.NOTIFY_OPS
    assert any("p95_latency_ms" in a for a in result.triggered_alerts)


def test_error_rate_trigger_fires(tmp_path: Path) -> None:
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.9], error=(i == 0)) for i in range(10)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert any("error_rate" in a for a in result.triggered_alerts)
    assert result.requires_human_review is True


# ── 11-14. Drift / baseline ──────────────────────────────────────────────────────


def test_drift_score_triggers_retraining_request(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scratch": 0.9, "dent": 0.1}), encoding="utf-8")
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.9], cls="dent") for i in range(10)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(baseline),
        ),
        tmp_path / "out",
    )
    assert result.drift_detected is True
    assert result.recommended_action == RecommendedAction.CREATE_RETRAINING_REQUEST
    assert result.metrics["drift_score"] > 0.3


def test_critical_class_drop_takes_priority_over_other_triggers(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scratch": 1.0}), encoding="utf-8")
    # everything else also fires: low confidence, high latency, errors
    records = [
        _rec(f"i{i}", "2026-07-10T12:00:00Z", [0.1], latency_ms=300.0, error=True, cls="dent")
        for i in range(10)
    ]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(baseline),
            critical_classes=["scratch"],
        ),
        tmp_path / "out",
    )
    assert result.recommended_action == RecommendedAction.MODEL_REVIEW
    assert len(result.triggered_alerts) >= 3  # critical drop, drift, latency, error, low-conf


def test_new_class_alone_requires_review_but_no_action(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"dent": 1.0}), encoding="utf-8")
    # 19 dent + 1 brand-new class -> small TVD, stays under the drift threshold
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.9], cls="dent") for i in range(19)]
    records.append(_rec("new", "2026-07-10T12:00:00Z", [0.9], cls="unknown_defect"))
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(baseline),
        ),
        tmp_path / "out",
    )
    assert result.new_classes_detected == ["unknown_defect"]
    assert result.requires_human_review is True
    assert result.recommended_action == RecommendedAction.NO_ACTION
    assert result.status == MonitoringStatus.ALERTS_TRIGGERED


def test_no_baseline_skips_drift_checks(tmp_path: Path) -> None:
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.9]) for i in range(5)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.metrics["drift_score"] == 0.0
    assert result.drift_detected is False
    assert result.new_classes_detected == []


def test_unreadable_baseline_path_fails(tmp_path: Path) -> None:
    records = [_rec("a", "2026-07-10T12:00:00Z", [0.9])]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(
            endpoint_name="ep",
            predictions_log_path=str(log),
            baseline_class_distribution_path=str(tmp_path / "nope.json"),
        ),
        tmp_path / "out",
    )
    assert result.success is False
    assert result.status == MonitoringStatus.FAILED


# ── 16. No triggers ──────────────────────────────────────────────────────────────


def test_no_triggers_fired_completes_cleanly(tmp_path: Path) -> None:
    records = [_rec(f"i{i}", "2026-07-10T12:00:00Z", [0.95], latency_ms=20.0) for i in range(10)]
    log = _write_log(tmp_path, records)
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), tmp_path / "out"
    )
    assert result.status == MonitoringStatus.COMPLETED
    assert result.recommended_action == RecommendedAction.NO_ACTION
    assert result.requires_human_review is False
    assert result.triggered_alerts == []


# ── 17. Hard samples manifest ────────────────────────────────────────────────────


def test_hard_samples_manifest_written(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [])])
    out_dir = tmp_path / "out"
    result = ModelMonitor().run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)), out_dir
    )
    manifest_path = Path(result.hard_samples_manifest_path)
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["image_id"] == "a"


# ── 18-19. MonitoringAgent ────────────────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.9])])
    artifacts_dir = tmp_path / "artifacts"

    agent = MonitoringAgent(artifacts_dir=artifacts_dir)
    result = agent.run(MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)))

    assert (artifacts_dir / "monitoring_report.json").exists()
    assert (artifacts_dir / "monitoring_report.md").exists()
    assert result.monitoring_report_path == str(artifacts_dir / "monitoring_report.json")

    data = json.loads((artifacts_dir / "monitoring_report.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"


def test_agent_logs_to_mlflow_only_on_success(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.9])])
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = MonitoringAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(MonitoringInput(endpoint_name="ep", predictions_log_path=str(log)))

    assert client.runs[run_id]["tags"].get("workflow_step") == "monitoring"

    client2 = FakeMLflowClient()
    run_id2 = client2.start_run("exp", "run")
    agent2 = MonitoringAgent(
        artifacts_dir=tmp_path / "artifacts2", mlflow_client=client2, mlflow_run_id=run_id2
    )
    agent2.run(
        MonitoringInput(endpoint_name="ep", predictions_log_path=str(tmp_path / "nope.jsonl"))
    )  # fails
    assert client2.runs[run_id2]["tags"] == {}


# ── 20-22. CLI ─────────────────────────────────────────────────────────────────


def test_cli_monitor_succeeds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.9])])

    result = CliRunner().invoke(
        app,
        [
            "monitor",
            str(log),
            "--endpoint-name",
            "ep",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "COMPLETED" in result.output


def test_cli_monitor_exits_1_on_missing_log(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "monitor",
            str(tmp_path / "nope.jsonl"),
            "--endpoint-name",
            "ep",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1


def test_cli_monitor_exits_1_on_invalid_window(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    log = _write_log(tmp_path, [_rec("a", "2026-07-10T12:00:00Z", [0.9])])

    result = CliRunner().invoke(
        app,
        [
            "monitor",
            str(log),
            "--endpoint-name",
            "ep",
            "--monitoring-window",
            "bogus",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
