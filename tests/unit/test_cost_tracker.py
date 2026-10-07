"""Unit tests for CostTracker, FakeCostTracker, and the cost-report CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.contracts.cost import ComputePricing, CostEntry
from agentic_mlops.tools.cost_tracker import CostTracker, FakeCostTracker, _parse_duration

# ── helpers ──────────────────────────────────────────────────────────────────


def _ts(offset_seconds: float = 0.0) -> str:
    """Return a fixed ISO timestamp offset by `offset_seconds`."""
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=offset_seconds)).isoformat()


# ── _parse_duration ───────────────────────────────────────────────────────────


def test_parse_duration_basic():
    duration = _parse_duration(_ts(0), _ts(3600))
    assert duration == pytest.approx(3600.0)


def test_parse_duration_fractional():
    duration = _parse_duration(_ts(0), _ts(90.5))
    assert duration == pytest.approx(90.5)


def test_parse_duration_none_start():
    assert _parse_duration(None, _ts(100)) is None


def test_parse_duration_none_end():
    assert _parse_duration(_ts(0), None) is None


def test_parse_duration_both_none():
    assert _parse_duration(None, None) is None


def test_parse_duration_invalid_strings():
    assert _parse_duration("not-a-date", _ts(10)) is None


def test_parse_duration_negative_clamped_to_zero():
    # end before start → clamp to 0
    assert _parse_duration(_ts(100), _ts(0)) == pytest.approx(0.0)


# ── ComputePricing ────────────────────────────────────────────────────────────


def test_pricing_default_has_gpu_types():
    p = ComputePricing()
    assert p.rate_for("Standard_NC6s_v3") == pytest.approx(3.06)
    assert p.rate_for("Standard_DS3_v2") == pytest.approx(0.27)


def test_pricing_local_is_zero():
    p = ComputePricing()
    assert p.rate_for("local") == 0.0
    assert p.rate_for("fake") == 0.0


def test_pricing_unknown_is_zero():
    p = ComputePricing()
    assert p.rate_for("some-custom-cluster") == 0.0


def test_pricing_custom_rate(tmp_path: Path):
    yaml_path = tmp_path / "pricing.yaml"
    yaml_path.write_text("rates:\n  gpu-cluster: 4.50\ncurrency: USD\n", encoding="utf-8")
    p = ComputePricing.from_yaml(yaml_path)
    assert p.rate_for("gpu-cluster") == pytest.approx(4.50)


def test_pricing_from_yaml_overrides_defaults(tmp_path: Path):
    yaml_path = tmp_path / "pricing.yaml"
    yaml_path.write_text("rates:\n  Standard_NC6: 1.00\ncurrency: EUR\n", encoding="utf-8")
    p = ComputePricing.from_yaml(yaml_path)
    assert p.currency == "EUR"
    # Only the overridden key is present (full replacement, not merge)
    assert p.rate_for("Standard_NC6") == pytest.approx(1.00)


# ── CostTracker.record ────────────────────────────────────────────────────────


def test_record_azure_ml_computes_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record(
        "training",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(3600),  # 1 hour
    )
    assert entry.step == "training"
    assert entry.runner == "azure-ml"
    assert entry.compute_type == "Standard_DS3_v2"
    assert entry.duration_seconds == pytest.approx(3600.0)
    # $0.27/hr * 1 hr * 1 instance = $0.27
    assert entry.estimated_cost == pytest.approx(0.27, rel=1e-5)
    assert entry.currency == "USD"


def test_record_multi_instance_scales_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record(
        "training",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        instance_count=4,
        started_at=_ts(0),
        completed_at=_ts(3600),  # 1 hour
    )
    # $0.27 * 4 instances * 1 hr = $1.08
    assert entry.estimated_cost == pytest.approx(1.08, rel=1e-5)
    assert entry.instance_count == 4


def test_record_no_timestamps_gives_zero_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record("training", "azure-ml", compute_type="Standard_NC6s_v3")
    assert entry.duration_seconds is None
    assert entry.estimated_cost == 0.0


def test_record_fake_runner_zero_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record(
        "training",
        "fake",
        compute_type="fake",
        started_at=_ts(0),
        completed_at=_ts(7200),
    )
    assert entry.estimated_cost == 0.0


def test_record_local_runner_zero_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record(
        "evaluation",
        "local-yolo",
        compute_type="local",
        started_at=_ts(0),
        completed_at=_ts(1800),
    )
    assert entry.estimated_cost == 0.0


def test_record_accumulates_multiple_entries():
    tracker = CostTracker(ComputePricing())
    tracker.record(
        "training",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )
    tracker.record(
        "evaluation",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(1800),
    )
    assert len(tracker.entries) == 2


def test_record_notes_stored():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record("training", "fake", notes="test run")
    assert entry.notes == "test run"


def test_record_unknown_compute_type_zero_cost():
    tracker = CostTracker(ComputePricing())
    entry = tracker.record(
        "training",
        "azure-ml",
        compute_type="custom-gpu-cluster",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )
    assert entry.estimated_cost == 0.0


def test_record_custom_pricing_maps_cluster_name():
    p = ComputePricing(rates={"custom-gpu-cluster": 5.0})
    tracker = CostTracker(p)
    entry = tracker.record(
        "training",
        "azure-ml",
        compute_type="custom-gpu-cluster",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )
    assert entry.estimated_cost == pytest.approx(5.0, rel=1e-5)


# ── CostTracker.summary ───────────────────────────────────────────────────────


def test_summary_total_is_sum():
    tracker = CostTracker(ComputePricing())
    tracker.record(
        "training",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )  # $0.27
    tracker.record(
        "evaluation",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(1800),
    )  # $0.135
    summary = tracker.summary("wf_test")
    assert summary.workflow_id == "wf_test"
    assert summary.total_cost == pytest.approx(0.27 + 0.135, rel=1e-5)
    assert summary.currency == "USD"
    assert len(summary.entries) == 2


def test_summary_empty_tracker_zero_total():
    tracker = CostTracker(ComputePricing())
    summary = tracker.summary("wf_empty")
    assert summary.total_cost == 0.0
    assert summary.entries == []


def test_summary_has_generated_at():
    tracker = CostTracker(ComputePricing())
    summary = tracker.summary("wf_ts")
    assert summary.generated_at != ""


# ── CostTracker.write_report ──────────────────────────────────────────────────


def test_write_report_creates_file(tmp_path: Path):
    tracker = CostTracker(ComputePricing())
    tracker.record("training", "fake")
    path = tracker.write_report("wf_001", tmp_path)
    assert path.exists()
    assert path.name == "cost_summary.json"


def test_write_report_valid_json(tmp_path: Path):
    tracker = CostTracker(ComputePricing())
    tracker.record(
        "training",
        "azure-ml",
        compute_type="Standard_DS3_v2",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )
    path = tracker.write_report("wf_001", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "wf_001"
    assert "total_cost" in data
    assert isinstance(data["entries"], list)


def test_write_report_creates_dir_if_missing(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    tracker = CostTracker(ComputePricing())
    path = tracker.write_report("wf_001", nested)
    assert path.exists()


# ── FakeCostTracker ───────────────────────────────────────────────────────────


def test_fake_records_calls():
    fake = FakeCostTracker()
    fake.record("training", "fake", compute_type="fake", started_at=_ts(0), completed_at=_ts(10))
    fake.record("evaluation", "azure-ml")
    assert len(fake.recorded) == 2
    assert fake.recorded[0]["step"] == "training"
    assert fake.recorded[1]["step"] == "evaluation"


def test_fake_returns_cost_entry():
    fake = FakeCostTracker()
    entry = fake.record("training", "fake")
    assert isinstance(entry, CostEntry)
    assert entry.step == "training"


def test_fake_entries_property():
    fake = FakeCostTracker()
    fake.record("training", "fake")
    fake.record("evaluation", "fake")
    assert len(fake.entries) == 2


def test_fake_summary_zero_cost():
    fake = FakeCostTracker()
    fake.record(
        "training",
        "azure-ml",
        compute_type="Standard_NC6s_v3",
        started_at=_ts(0),
        completed_at=_ts(3600),
    )
    summary = fake.summary("wf_test")
    assert summary.total_cost == 0.0  # FakeCostTracker never estimates real cost


def test_fake_write_report(tmp_path: Path):
    fake = FakeCostTracker()
    fake.record("training", "fake")
    path = fake.write_report("wf_001", tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "wf_001"


# ── OrchestratorWorkflow integration ─────────────────────────────────────────


def test_orchestrator_wires_cost_tracker(tmp_path: Path):
    """Cost tracker receives records after training + evaluation complete."""
    from agentic_mlops.contracts.orchestrator import (
        OrchestratorInput,
        OrchestratorStatus,
    )
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    fake = FakeCostTracker()
    workflow = OrchestratorWorkflow(cost_tracker=fake)

    dataset_dir = tmp_path / "ds"
    data_yaml = dataset_dir / "data.yaml"
    dataset_dir.mkdir()
    data_yaml.write_text(
        "nc: 1\nnames: [obj]\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)

    training_cfg = tmp_path / "training.yaml"
    training_cfg.write_text("epochs: 1\nbatch_size: 4\nimgsz: 640\n", encoding="utf-8")

    promotion_cfg = tmp_path / "policy.yaml"
    promotion_cfg.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n",
        encoding="utf-8",
    )

    inp = OrchestratorInput(
        workflow_id="wf_cost_test",
        runs_dir=str(tmp_path / "runs"),
        output_dir=str(tmp_path / "runs" / "wf_cost_test" / "artifacts"),
        dataset_path=str(dataset_dir),
        data_yaml_path=str(data_yaml),
        training_config_path=str(training_cfg),
        promotion_policy_path=str(promotion_cfg),
        dry_run=True,
        interactive_approval=False,
        approval_action="approve_model",
        steps=["training", "evaluation"],
    )

    result = workflow.run(inp)
    assert result.status == OrchestratorStatus.COMPLETED
    # Both training and evaluation steps should have been recorded
    steps_recorded = [r["step"] for r in fake.recorded]
    assert "training" in steps_recorded
    assert "evaluation" in steps_recorded


def test_orchestrator_writes_cost_summary_to_artifacts(tmp_path: Path):
    """_finish writes cost_summary.json into the artifacts dir."""
    from agentic_mlops.contracts.orchestrator import OrchestratorInput
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    fake = FakeCostTracker()
    workflow = OrchestratorWorkflow(cost_tracker=fake)

    dataset_dir = tmp_path / "ds"
    data_yaml = dataset_dir / "data.yaml"
    dataset_dir.mkdir()
    data_yaml.write_text(
        "nc: 1\nnames: [obj]\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)

    training_cfg = tmp_path / "training.yaml"
    training_cfg.write_text("epochs: 1\nbatch_size: 4\nimgsz: 640\n", encoding="utf-8")

    promotion_cfg = tmp_path / "policy.yaml"
    promotion_cfg.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n",
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "runs" / "wf_cost_artifacts" / "artifacts"
    inp = OrchestratorInput(
        workflow_id="wf_cost_artifacts",
        runs_dir=str(tmp_path / "runs"),
        output_dir=str(artifacts_dir),
        dataset_path=str(dataset_dir),
        data_yaml_path=str(data_yaml),
        training_config_path=str(training_cfg),
        promotion_policy_path=str(promotion_cfg),
        dry_run=True,
        interactive_approval=False,
        approval_action="approve_model",
        steps=["training", "evaluation"],
    )

    workflow.run(inp)
    cost_path = Path(tmp_path / "runs" / "wf_cost_artifacts" / "artifacts" / "cost_summary.json")
    assert cost_path.exists(), "cost_summary.json should be written by _finish"
    data = json.loads(cost_path.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "wf_cost_artifacts"


def test_orchestrator_no_cost_tracker_no_crash(tmp_path: Path):
    """Workflow runs cleanly without a cost_tracker injected."""
    from agentic_mlops.contracts.orchestrator import OrchestratorInput, OrchestratorStatus
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    workflow = OrchestratorWorkflow()  # no cost_tracker

    dataset_dir = tmp_path / "ds"
    data_yaml = dataset_dir / "data.yaml"
    dataset_dir.mkdir()
    data_yaml.write_text(
        "nc: 1\nnames: [obj]\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)

    training_cfg = tmp_path / "training.yaml"
    training_cfg.write_text("epochs: 1\nbatch_size: 4\nimgsz: 640\n", encoding="utf-8")

    promotion_cfg = tmp_path / "policy.yaml"
    promotion_cfg.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n",
        encoding="utf-8",
    )

    inp = OrchestratorInput(
        workflow_id="wf_no_cost",
        runs_dir=str(tmp_path / "runs"),
        output_dir=str(tmp_path / "runs" / "wf_no_cost" / "artifacts"),
        dataset_path=str(dataset_dir),
        data_yaml_path=str(data_yaml),
        training_config_path=str(training_cfg),
        promotion_policy_path=str(promotion_cfg),
        dry_run=True,
        interactive_approval=False,
        approval_action="approve_model",
        steps=["training", "evaluation"],
    )

    result = workflow.run(inp)
    assert result.status == OrchestratorStatus.COMPLETED
    # No cost_summary.json since no tracker
    cost_path = tmp_path / "runs" / "wf_no_cost" / "artifacts" / "cost_summary.json"
    assert not cost_path.exists()


# ── CLI cost-report command ───────────────────────────────────────────────────


def test_cost_report_command_missing_state(tmp_path: Path):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["cost-report", "wf_missing", "--runs-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "State file not found" in result.output


def test_cost_report_command_outputs_summary(tmp_path: Path):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    # Create a minimal state.json with timing data
    wf_dir = tmp_path / "wf_cli_test"
    wf_dir.mkdir()
    state = {
        "workflow_id": "wf_cli_test",
        "step_outputs": {
            "training": {
                "started_at": _ts(0),
                "completed_at": _ts(3600),
                "compute_type": "Standard_DS3_v2",
            },
            "evaluation": {
                "started_at": _ts(0),
                "completed_at": _ts(1800),
                "compute_type": "Standard_DS3_v2",
            },
        },
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["cost-report", "wf_cli_test", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "wf_cli_test" in result.output
    assert "training" in result.output
    assert "evaluation" in result.output


def test_cost_report_command_writes_json(tmp_path: Path):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    wf_dir = tmp_path / "wf_write_test"
    wf_dir.mkdir()
    state = {
        "workflow_id": "wf_write_test",
        "step_outputs": {
            "training": {"started_at": _ts(0), "completed_at": _ts(3600)},
        },
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    out_file = tmp_path / "out" / "cost_summary.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "cost-report",
            "wf_write_test",
            "--runs-dir",
            str(tmp_path),
            "--output-file",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["workflow_id"] == "wf_write_test"


def test_cost_report_command_custom_pricing(tmp_path: Path):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    wf_dir = tmp_path / "wf_pricing_test"
    wf_dir.mkdir()
    state = {
        "workflow_id": "wf_pricing_test",
        "step_outputs": {
            "training": {
                "started_at": _ts(0),
                "completed_at": _ts(3600),
                "compute_type": "custom-cluster",
            },
        },
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    pricing_cfg = tmp_path / "pricing.yaml"
    pricing_cfg.write_text("rates:\n  custom-cluster: 10.0\ncurrency: USD\n", encoding="utf-8")

    out_file = tmp_path / "cost_summary.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "cost-report",
            "wf_pricing_test",
            "--runs-dir",
            str(tmp_path),
            "--pricing-config",
            str(pricing_cfg),
            "--output-file",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    # $10.0/hr * 1hr = $10.0
    assert data["total_cost"] == pytest.approx(10.0, rel=1e-4)


def test_cost_report_command_skipped_step_not_recorded(tmp_path: Path):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    wf_dir = tmp_path / "wf_skip_test"
    wf_dir.mkdir()
    state = {
        "workflow_id": "wf_skip_test",
        "step_outputs": {
            "training": {"skipped": True},
            "evaluation": {"skipped": True},
        },
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    out_file = tmp_path / "cost_skip.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "cost-report",
            "wf_skip_test",
            "--runs-dir",
            str(tmp_path),
            "--output-file",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["total_cost"] == 0.0
    assert data["entries"] == []


def test_cost_report_command_json_format(tmp_path: Path):
    """--format json prints a machine-readable JSON summary to stdout."""
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    wf_dir = tmp_path / "wf_json_fmt"
    wf_dir.mkdir()
    state = {
        "workflow_id": "wf_json_fmt",
        "step_outputs": {
            "training": {
                "started_at": _ts(0),
                "completed_at": _ts(3600),
                "compute_type": "Standard_DS3_v2",
            },
        },
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["cost-report", "wf_json_fmt", "--runs-dir", str(tmp_path), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["workflow_id"] == "wf_json_fmt"
    assert "total_cost" in data
    assert isinstance(data["entries"], list)
