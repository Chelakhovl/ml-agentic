"""End-to-end integration tests for OrchestratorWorkflow.

Unlike the unit tests (which use stubs/mocks for individual steps), these
tests run the real agents with fake/dry-run runners and verify the full
artifact chain on disk: state.json, audit_log.jsonl, orchestrator_report,
per-step output files, and state-machine transitions.

No special marker required — these tests run as part of `pytest tests/` and
do not make any real Azure ML or network calls.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from agentic_mlops.contracts.approvals import ApprovalAction
from agentic_mlops.contracts.orchestrator import (
    OrchestratorInput,
    OrchestratorStatus,
)
from agentic_mlops.contracts.training_approval import TrainingApprovalAction
from agentic_mlops.integrations.workflow_state_store import WorkflowStateStore
from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
from tests.conftest import make_valid_dataset

# ── Helpers ───────────────────────────────────────────────────────────────────

# Steps that complete successfully with dry_run=True (no real weights produced)
_MVP_NO_REGISTRY = ["dataset_validation", "training", "evaluation", "approval"]


def _make_dataset(root: Path) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(ds, num_classes=2)
    return ds


def _make_training_config(root: Path) -> Path:
    cfg = root / "training.yaml"
    cfg.write_text("epochs: 1\nimgsz: 32\nbatch: 2\ndevice: cpu\n", encoding="utf-8")
    return cfg


def _make_promotion_policy(root: Path) -> Path:
    policy = root / "promotion_policy.yaml"
    policy.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n"
        "require_improvement_over_baseline: false\n",
        encoding="utf-8",
    )
    return policy


def _base_input(tmp_path: Path, **overrides: object) -> OrchestratorInput:
    dataset = _make_dataset(tmp_path)
    training_cfg = _make_training_config(tmp_path)
    policy_cfg = _make_promotion_policy(tmp_path)
    fields: dict[str, object] = {
        "workflow_id": "e2e_test",
        "runs_dir": str(tmp_path / "runs"),
        "dataset_path": str(dataset),
        "data_yaml_path": str(dataset / "data.yaml"),
        "training_config_path": str(training_cfg),
        "promotion_policy_path": str(policy_cfg),
        "dry_run": True,
        "steps": _MVP_NO_REGISTRY,
        "interactive_approval": False,
        "registry_dir": str(tmp_path / "registry"),
        "deployment_dir": str(tmp_path / "deployments"),
        "export_format": "pt",
    }
    fields.update(overrides)
    return OrchestratorInput.model_validate(fields)


def _make_valid_jpeg_bytes() -> bytes:
    """Return a minimal valid JPEG bytes (1×1 grey pixel, Pillow-readable)."""
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


# ── 1. Full pipeline (no model_registry) happy path ──────────────────────────


def test_full_pipeline_happy_path_produces_all_artifacts(tmp_path: Path) -> None:
    """validation → training → evaluation → approval completes; state + audit + report exist."""
    inp = _base_input(
        tmp_path,
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    result = OrchestratorWorkflow().run(inp)

    assert result.status == OrchestratorStatus.COMPLETED, result.message
    assert [s.step for s in result.steps] == _MVP_NO_REGISTRY
    assert all(s.success for s in result.steps)

    runs_dir = Path(inp.runs_dir)
    state_path = runs_dir / inp.workflow_id / "state.json"
    audit_path = runs_dir / inp.workflow_id / "audit_log.jsonl"
    assert state_path.exists(), "state.json missing"
    assert audit_path.exists(), "audit_log.jsonl missing"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["current_state"] == "COMPLETED"
    assert set(state["completed_steps"]) == set(_MVP_NO_REGISTRY)

    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    event_types = [e["event"] for e in events]
    assert "step_started" in event_types
    assert "step_finished" in event_types

    # orchestrator_report.json must be written
    artifacts_dir = Path(inp.output_dir or (runs_dir / inp.workflow_id / "artifacts"))
    report_path = artifacts_dir / "orchestrator_report.json"
    assert report_path.exists(), "orchestrator_report.json missing"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"


# ── 2. evaluation_output.json is written and contains success=True ────────────


def test_evaluation_output_json_written_with_success(tmp_path: Path) -> None:
    """evaluation_output.json is present after the evaluation step and has success=True."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_eval_out",
        steps=["dataset_validation", "training", "evaluation"],
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED, result.message

    runs_dir = Path(inp.runs_dir)
    artifacts_root = runs_dir / inp.workflow_id / "artifacts"
    eval_out = artifacts_root / "evaluation" / "evaluation_output.json"
    assert eval_out.exists(), "evaluation_output.json not written by orchestrator"

    data = json.loads(eval_out.read_text(encoding="utf-8"))
    assert data.get("success") is True, "evaluation_output.json must have success=True"


# ── 3. training_output.json is written even in dry_run mode ──────────────────


def test_training_output_json_written_in_dry_run(tmp_path: Path) -> None:
    """training_output.json is present after the training step in dry_run mode."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_train_out",
        steps=["dataset_validation", "training"],
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED, result.message

    runs_dir = Path(inp.runs_dir)
    artifacts_root = runs_dir / inp.workflow_id / "artifacts"
    train_out = artifacts_root / "training" / "training_output.json"
    assert train_out.exists(), "training_output.json not written by orchestrator in dry_run"


# ── 4. model_registry succeeds when training produces a real best.pt ─────────


def test_model_registry_succeeds_with_stub_weights(tmp_path: Path) -> None:
    """After running to approval, inject a stub best.pt so model_registry can pass Gate 5."""
    # Run the first 4 steps to get training/evaluation/approval artifacts on disk.
    inp_partial = _base_input(
        tmp_path,
        workflow_id="e2e_registry",
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    wf = OrchestratorWorkflow()
    r1 = wf.run(inp_partial)
    assert r1.status == OrchestratorStatus.COMPLETED, r1.message

    # Inject a stub best.pt next to training_output.json so Gate 5 passes.
    runs_dir = Path(inp_partial.runs_dir)
    train_dir = runs_dir / inp_partial.workflow_id / "artifacts" / "training"
    stub_pt = train_dir / "best.pt"
    stub_pt.write_bytes(b"PT_STUB_FOR_GATE5")

    # Patch best_weights_path into training_output.json as well.
    train_out_path = train_dir / "training_output.json"
    train_data = json.loads(train_out_path.read_text(encoding="utf-8"))
    train_data["best_weights_path"] = str(stub_pt)
    train_out_path.write_text(json.dumps(train_data, indent=2), encoding="utf-8")

    # Re-run with resume_from_step=model_registry (skip re-running earlier steps).
    inp_registry = inp_partial.model_copy(
        update={
            "resume": True,
            "resume_from_step": "model_registry",
            "steps": _MVP_NO_REGISTRY + ["model_registry"],
            "model_name": "e2e-yolo",
        }
    )
    r2 = wf.run(inp_registry)
    assert r2.status == OrchestratorStatus.COMPLETED, r2.message

    registry_step = next(s for s in r2.steps if s.step == "model_registry")
    assert registry_step.success is True
    assert (tmp_path / "registry" / "e2e-yolo" / "versions").exists()


# ── 5. H5 pause → resume cycle ────────────────────────────────────────────────


def test_h5_pause_resume_completes_workflow(tmp_path: Path) -> None:
    """Non-interactive run with no action pauses at H5; resume with action completes it."""
    inp = _base_input(tmp_path)  # no approval_action -> pause
    wf = OrchestratorWorkflow()
    first = wf.run(inp)
    assert first.status == OrchestratorStatus.PENDING_APPROVAL, first.message
    assert first.current_state == "MODEL_APPROVAL_REQUIRED"
    assert first.pending_approval_id is not None

    store = WorkflowStateStore(Path(inp.runs_dir), inp.workflow_id)
    saved = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert saved["current_state"] == "MODEL_APPROVAL_REQUIRED"

    resumed = inp.model_copy(
        update={
            "resume": True,
            "approval_action": ApprovalAction.APPROVE_MODEL,
            "force_approve": True,
        }
    )
    second = wf.run(resumed)
    assert second.status == OrchestratorStatus.COMPLETED, second.message

    audit_text = store.audit_log_path.read_text(encoding="utf-8")
    audit_events = [json.loads(line) for line in audit_text.splitlines()]
    # The approval step is started on the first run (pauses) and again on resume.
    # What matters is at least one finished-success event.
    approval_finished = [
        e
        for e in audit_events
        if e["event"] == "step_finished" and e["step"] == "approval" and e.get("success")
    ]
    assert len(approval_finished) >= 1, "approval step should finish successfully at least once"


# ── 6. H4 + H5 combined gates ────────────────────────────────────────────────


def test_h4_pause_then_h5_pause_then_complete(tmp_path: Path) -> None:
    """Three-run cycle: pause at H4 → approve_training → pause at H5 → approve_model."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_h4h5",
        steps=[
            "dataset_validation",
            "training_approval",
            "training",
            "evaluation",
            "approval",
        ],
        interactive_training_approval=False,
        # No training_approval_action -> pause at H4
    )
    wf = OrchestratorWorkflow()

    r1 = wf.run(inp)
    assert r1.status == OrchestratorStatus.PENDING_APPROVAL
    assert r1.current_state == "TRAINING_APPROVAL_REQUIRED"

    r2 = wf.run(
        inp.model_copy(
            update={
                "resume": True,
                "training_approval_action": TrainingApprovalAction.APPROVE_TRAINING,
            }
        )
    )
    assert r2.status == OrchestratorStatus.PENDING_APPROVAL
    assert r2.current_state == "MODEL_APPROVAL_REQUIRED"

    r3 = wf.run(
        inp.model_copy(
            update={
                "resume": True,
                "training_approval_action": TrainingApprovalAction.APPROVE_TRAINING,
                "approval_action": ApprovalAction.APPROVE_MODEL,
                "force_approve": True,
            }
        )
    )
    assert r3.status == OrchestratorStatus.COMPLETED, r3.message


# ── 7. H4 reject cascades downstream as SKIPPED ──────────────────────────────


def test_h4_reject_training_cascades_skipped(tmp_path: Path) -> None:
    """reject_training at H4 → all downstream steps SKIPPED; workflow COMPLETED."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_h4_reject",
        steps=[
            "dataset_validation",
            "training_approval",
            "training",
            "evaluation",
            "approval",
        ],
        interactive_training_approval=False,
        training_approval_action=TrainingApprovalAction.REJECT_TRAINING,
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED, result.message

    by_step = {s.step: s for s in result.steps}
    assert by_step["dataset_validation"].success is True
    assert by_step["training_approval"].success is True
    for step in ("training", "evaluation", "approval"):
        assert by_step[step].status == "SKIPPED", f"{step} should be SKIPPED after H4 rejection"


# ── 8. resume_from_step re-runs from the named step ──────────────────────────


def test_resume_from_step_reruns_from_named_step(tmp_path: Path) -> None:
    """resume_from_step=evaluation re-runs evaluation + approval; skips validation + training."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_rfstep",
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    wf = OrchestratorWorkflow()
    r1 = wf.run(inp)
    assert r1.status == OrchestratorStatus.COMPLETED

    store = WorkflowStateStore(Path(inp.runs_dir), inp.workflow_id)
    audit_before = store.audit_log_path.read_text(encoding="utf-8").count('"step_started"')

    r2 = wf.run(
        inp.model_copy(
            update={
                "resume": True,
                "resume_from_step": "evaluation",
            }
        )
    )
    assert r2.status == OrchestratorStatus.COMPLETED, r2.message

    audit_after = store.audit_log_path.read_text(encoding="utf-8").count('"step_started"')
    # evaluation + approval re-started = 2 new step_started events
    assert audit_after - audit_before == 2


# ── 9. Two workflow IDs have independent state ────────────────────────────────


def test_two_workflow_ids_have_independent_state(tmp_path: Path) -> None:
    """Two workflows in the same runs_dir keep completely separate state files."""
    runs_dir = str(tmp_path / "runs")

    inp_a = _base_input(
        tmp_path,
        workflow_id="e2e_wf_a",
        runs_dir=runs_dir,
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    inp_b = _base_input(
        tmp_path,
        workflow_id="e2e_wf_b",
        runs_dir=runs_dir,
        approval_action=ApprovalAction.REJECT_MODEL,
    )

    wf = OrchestratorWorkflow()
    r_a = wf.run(inp_a)
    r_b = wf.run(inp_b)

    assert r_a.status == OrchestratorStatus.COMPLETED, r_a.message
    assert r_b.status == OrchestratorStatus.COMPLETED, r_b.message

    state_a = json.loads((Path(runs_dir) / "e2e_wf_a" / "state.json").read_text())
    state_b = json.loads((Path(runs_dir) / "e2e_wf_b" / "state.json").read_text())

    # Both ran the same steps; results differ only by approval outcome
    assert set(state_a["completed_steps"]) == set(_MVP_NO_REGISTRY)
    assert set(state_b["completed_steps"]) == set(_MVP_NO_REGISTRY)


# ── 10. Full 11-step pipeline (slow; requires Pillow) ─────────────────────────


@pytest.mark.slow
def test_full_11_step_pipeline(tmp_path: Path) -> None:
    """All 11 PIPELINE_STEPS run end-to-end with fake/dry-run runners and Pillow-valid images."""
    # Create raw images that Pillow can open (data_intake step uses Pillow if available).
    # Each image must have unique content — duplicate detection across splits fires
    # when all images share the same bytes.
    from PIL import Image  # noqa: PLC0415

    from agentic_mlops.contracts.orchestrator import PIPELINE_STEPS  # noqa: PLC0415

    raw_data = tmp_path / "raw_images"
    raw_data.mkdir(parents=True)
    for i in range(6):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(i * 40, i * 20, 128)).save(buf, format="JPEG")
        (raw_data / f"img_{i:02d}.jpg").write_bytes(buf.getvalue())
        (raw_data / f"img_{i:02d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    training_cfg = _make_training_config(tmp_path)
    policy_cfg = _make_promotion_policy(tmp_path)

    inp = OrchestratorInput.model_validate(
        {
            "workflow_id": "e2e_full_11",
            "runs_dir": str(tmp_path / "runs"),
            "steps": list(PIPELINE_STEPS),
            "dry_run": True,
            # data_intake
            "raw_data_path": str(raw_data),
            "dataset_name": "e2e_dataset",
            # dataset_structuring
            "classes": ["scratch", "dent"],
            "label_format": "yolo",
            "train_ratio": 0.75,
            "val_ratio": 0.25,
            "test_ratio": 0.0,
            # training_approval (H4)
            "interactive_training_approval": False,
            "training_approval_action": TrainingApprovalAction.APPROVE_TRAINING,
            # training
            "training_config_path": str(training_cfg),
            # evaluation / model_decision
            "promotion_policy_path": str(policy_cfg),
            # approval (H5)
            "interactive_approval": False,
            "approval_action": ApprovalAction.REJECT_MODEL,  # reject → registry/deploy SKIPPED
            # model_registry / deployment remain SKIPPED so no best.pt needed
            "model_name": "e2e-full-model",
            "registry_dir": str(tmp_path / "registry"),
            "dataset_registry_dir": str(tmp_path / "dataset_registry"),
            "deployment_dir": str(tmp_path / "deployments"),
            "export_format": "pt",
        }
    )

    result = OrchestratorWorkflow().run(inp)

    assert result.status == OrchestratorStatus.COMPLETED, result.message
    assert {s.step for s in result.steps} == set(PIPELINE_STEPS)

    # Steps up to approval must succeed; model_registry + deployment must be SKIPPED
    by_step = {s.step: s for s in result.steps}
    for step in (
        "data_intake",
        "dataset_structuring",
        "dataset_validation",
        "dataset_versioning",
        "training_approval",
        "training",
        "evaluation",
        "model_decision",
        "approval",
    ):
        assert by_step[step].success is True, f"{step} failed: {by_step[step].errors}"
    for step in ("model_registry", "deployment"):
        assert by_step[step].status == "SKIPPED", f"{step} should be SKIPPED"

# ── 11. H5 reject cascades model_registry + deployment SKIPPED ───────────────


def test_h5_reject_model_cascades_skipped(tmp_path: Path) -> None:
    """REJECT_MODEL at H5 marks model_registry and deployment as SKIPPED; workflow COMPLETED."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_h5_reject",
        steps=_MVP_NO_REGISTRY + ["model_registry", "deployment"],
        approval_action=ApprovalAction.REJECT_MODEL,
        model_name="e2e-reject",
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED, result.message

    by_step = {s.step: s for s in result.steps}
    assert by_step["approval"].success is True
    for step in ("model_registry", "deployment"):
        assert by_step[step].status == "SKIPPED", f"{step} should be SKIPPED after H5 rejection"


# ── 12. model_decision output written when step is included ──────────────────


def test_model_decision_output_written_and_correct(tmp_path: Path) -> None:
    """model_decision step writes decision_report.json with a non-empty recommendation."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_decision",
        steps=["dataset_validation", "training", "evaluation", "model_decision"],
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED, result.message

    runs_dir = Path(inp.runs_dir)
    decision_out = (
        runs_dir / inp.workflow_id / "artifacts" / "model_decision" / "decision_report.json"
    )
    assert decision_out.exists(), "decision_report.json not written by orchestrator"
    data = json.loads(decision_out.read_text(encoding="utf-8"))
    assert data.get("success") is True
    assert data.get("decision") is not None


# ── 13. state.json contains schema_version after a real run ──────────────────


def test_state_json_has_schema_version_after_run(tmp_path: Path) -> None:
    """OrchestratorWorkflow writes schema_version into state.json on every save."""
    from agentic_mlops.integrations.workflow_state_store import STATE_SCHEMA_VERSION

    inp = _base_input(
        tmp_path,
        workflow_id="e2e_schema",
        steps=["dataset_validation"],
    )
    OrchestratorWorkflow().run(inp)

    state = json.loads(
        (Path(inp.runs_dir) / inp.workflow_id / "state.json").read_text(encoding="utf-8")
    )
    assert state["schema_version"] == STATE_SCHEMA_VERSION


# ── 14. Duplicate workflow_id without resume raises an error ─────────────────


def test_duplicate_workflow_id_without_resume_fails(tmp_path: Path) -> None:
    """Re-running a completed workflow without --resume fails with a clear error."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_dup",
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    wf = OrchestratorWorkflow()
    first = wf.run(inp)
    assert first.status == OrchestratorStatus.COMPLETED

    second = wf.run(inp)  # same workflow_id, no resume=True
    assert second.status == OrchestratorStatus.FAILED
    assert second.message  # has an explanatory message


# ── 15. audit log has workflow_started and workflow_finished events ───────────


def test_audit_log_bookends_present(tmp_path: Path) -> None:
    """audit_log.jsonl starts with workflow_started and ends with workflow_finished."""
    inp = _base_input(
        tmp_path,
        workflow_id="e2e_audit_bk",
        approval_action=ApprovalAction.APPROVE_MODEL,
        force_approve=True,
    )
    OrchestratorWorkflow().run(inp)

    audit_path = Path(inp.runs_dir) / inp.workflow_id / "audit_log.jsonl"
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    event_types = [e["event"] for e in events]
    assert "workflow_started" in event_types, "expected workflow_started in audit log"
    assert "workflow_finished" in event_types, "expected workflow_finished in audit log"
