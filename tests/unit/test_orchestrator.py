"""Unit tests for the Orchestrator: OrchestratorWorkflow, WorkflowStateStore, and the CLI.

Coverage matrix:
    1.  Unknown step name -> immediate FAILED, no crash
    2.  Default steps (None) resolve to the 5 classic MVP steps, in order
    3.  dataset_validation failure stops the chain before training runs
    4.  Full default pipeline pauses at approval (non-interactive, no action) ->
        PENDING_APPROVAL, current_state MODEL_APPROVAL_REQUIRED, pending_approval_id set
    5.  Resume after a pause completes remaining steps without re-running earlier ones
    6.  approval_action=REJECT_MODEL -> workflow COMPLETED (business outcome, not a failure)
    7.  model_registry is skipped gracefully when approval was not approve_model
    8.  deployment is skipped gracefully when approval was not approve_model
    9.  Re-running the same workflow_id without resume=True fails clearly
   10.  resume=True with no prior state starts a fresh run
   11.  resume=True on an already-completed workflow returns immediately (idempotent)
   12.  state.json and audit_log.jsonl are written with the expected shape
   13.  dataset_structuring's output feeds dataset_validation automatically
   14.  dataset_validation without dataset_structuring and without dataset_path fails clearly
   15.  dataset_versioning step wiring registers a version
   16.  data_intake without raw_data_path fails clearly
   17.  training without training_config_path fails clearly
   18.  evaluation without a prior training step fails clearly
   19.  _is_legal: valid and invalid state transitions (white-box policy engine test)
   20.  orchestrator_report.json/.md are written
   21.  OrchestratorInput.from_yaml loads a config file and applies overrides
   22.  CLI: run-workflow succeeds and pauses at approval (exit 0)
   23.  CLI: run-workflow exits 1 on a bad config path
   24.  CLI: run-workflow exits 1 when re-run without --resume
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agentic_mlops.contracts.approvals import ApprovalAction
from agentic_mlops.contracts.orchestrator import (
    DEFAULT_STEPS,
    OrchestratorInput,
    OrchestratorStatus,
)
from agentic_mlops.integrations.workflow_state_store import WorkflowStateStore
from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
from tests.conftest import make_valid_dataset

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    make_valid_dataset(dataset, num_classes=3)
    return dataset


def _make_training_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "training.yaml"
    cfg.write_text("epochs: 1\nimgsz: 640\nbatch: 2\nname: test_run\n", encoding="utf-8")
    return cfg


def _base_input(tmp_path: Path, **overrides: object) -> OrchestratorInput:
    dataset = _make_dataset(tmp_path)
    training_cfg = _make_training_config(tmp_path)
    fields: dict[str, object] = {
        "workflow_id": "wf_test",
        "runs_dir": str(tmp_path / "runs"),
        "dataset_path": str(dataset),
        "data_yaml_path": str(dataset / "data.yaml"),
        "training_config_path": str(training_cfg),
        "dry_run": True,
        "interactive_approval": False,
    }
    fields.update(overrides)
    return OrchestratorInput.model_validate(fields)


# ── 1-2. Step resolution ─────────────────────────────────────────────────────────


def test_unknown_step_fails_immediately(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["not_a_real_step"])
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    assert "Unknown step" in result.message


def test_default_steps_are_the_five_mvp_steps_in_order(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, approval_action=ApprovalAction.APPROVE_MODEL, force_approve=True)
    result = OrchestratorWorkflow().run(inp)
    assert [s.step for s in result.steps] == list(DEFAULT_STEPS)


# ── 3. Failure stops the chain ────────────────────────────────────────────────────


def test_dataset_validation_failure_stops_chain(tmp_path: Path) -> None:
    bad_dataset = tmp_path / "empty_dataset"
    bad_dataset.mkdir(parents=True)
    inp = _base_input(
        tmp_path, dataset_path=str(bad_dataset), data_yaml_path=str(bad_dataset / "data.yaml")
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    assert [s.step for s in result.steps] == ["dataset_validation"]


# ── 4-5. Pause / resume ─────────────────────────────────────────────────────────


def test_pauses_at_approval_when_no_action_given(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation", "training", "evaluation", "approval"])
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.PENDING_APPROVAL
    assert result.current_state == "MODEL_APPROVAL_REQUIRED"
    assert result.pending_approval_id == "appr_wf_test"
    assert result.success is True


def test_resume_completes_without_rerunning_earlier_steps(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation", "training", "evaluation", "approval"])
    wf = OrchestratorWorkflow()
    wf.run(inp)

    store = WorkflowStateStore(Path(inp.runs_dir), inp.workflow_id)
    audit_before = store.audit_log_path.read_text(encoding="utf-8").count('"step_started"')

    resumed = inp.model_copy(
        update={
            "resume": True,
            "approval_action": ApprovalAction.APPROVE_MODEL,
            "force_approve": True,
        }
    )
    result = wf.run(resumed)
    assert result.status == OrchestratorStatus.COMPLETED
    assert [s.step for s in result.steps] == [
        "dataset_validation", "training", "evaluation", "approval"
    ]

    audit_after = store.audit_log_path.read_text(encoding="utf-8").count('"step_started"')
    # Only "approval" should have (re-)started on resume — the other 3 steps were skipped.
    assert audit_after - audit_before == 1


# ── 6-8. Non-approve outcomes ────────────────────────────────────────────────────


def test_reject_model_completes_workflow_not_fails(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path,
        steps=["dataset_validation", "training", "evaluation", "approval", "model_registry"],
        approval_action=ApprovalAction.REJECT_MODEL,
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED
    approval_step = next(s for s in result.steps if s.step == "approval")
    assert approval_step.success is True


def test_model_registry_skipped_when_not_approved(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path,
        steps=["dataset_validation", "training", "evaluation", "approval", "model_registry"],
        approval_action=ApprovalAction.REJECT_MODEL,
    )
    result = OrchestratorWorkflow().run(inp)
    registry_step = next(s for s in result.steps if s.step == "model_registry")
    assert registry_step.status == "SKIPPED"
    assert registry_step.success is True


def test_deployment_skipped_when_not_approved(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path,
        steps=["dataset_validation", "training", "evaluation", "approval", "deployment"],
        approval_action=ApprovalAction.REJECT_MODEL,
    )
    result = OrchestratorWorkflow().run(inp)
    deploy_step = next(s for s in result.steps if s.step == "deployment")
    assert deploy_step.status == "SKIPPED"


# ── 9-11. State file guards ──────────────────────────────────────────────────────


def test_rerun_without_resume_fails_clearly(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation"])
    wf = OrchestratorWorkflow()
    first = wf.run(inp)
    assert first.status == OrchestratorStatus.COMPLETED

    second = wf.run(inp)
    assert second.status == OrchestratorStatus.FAILED
    assert "already exists" in second.message
    assert "resume" in second.message.lower()


def test_resume_with_no_prior_state_starts_fresh(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation"], resume=True)
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED


def test_resume_on_already_completed_workflow_is_idempotent(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation"])
    wf = OrchestratorWorkflow()
    wf.run(inp)

    resumed = inp.model_copy(update={"resume": True})
    result = wf.run(resumed)
    assert result.status == OrchestratorStatus.COMPLETED
    assert "already completed" in result.message.lower()


# ── 12. State/audit file shape ──────────────────────────────────────────────────


def test_state_and_audit_files_written(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation"])
    result = OrchestratorWorkflow().run(inp)

    state = json.loads(Path(result.state_path).read_text(encoding="utf-8"))
    assert state["workflow_id"] == "wf_test"
    assert state["completed_steps"] == ["dataset_validation"]
    assert state["status"] == "completed"

    audit_lines = Path(result.audit_log_path).read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in audit_lines]
    assert "workflow_started" in events
    assert "step_started" in events
    assert "step_finished" in events
    assert "workflow_finished" in events


# ── 13-14. Dataset wiring ─────────────────────────────────────────────────────────


def test_dataset_structuring_feeds_dataset_validation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(6):
        img = raw / f"img_{i}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8\xff\xe0" + img.name.encode() + b"\xff\xd9")
        (raw / f"img_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    inp = _base_input(
        tmp_path,
        steps=["dataset_structuring", "dataset_validation"],
        dataset_path=None,
        data_yaml_path=None,
        raw_data_path=str(raw),
        classes=["scratch", "dent", "crack"],
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED
    val_step = next(s for s in result.steps if s.step == "dataset_validation")
    assert val_step.success is True


def test_dataset_validation_without_dataset_path_fails_clearly(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path, steps=["dataset_validation"], dataset_path=None, data_yaml_path=None
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    assert "dataset_structuring" in result.errors[0]


# ── 15. dataset_versioning wiring ─────────────────────────────────────────────────


def test_dataset_versioning_registers_a_version(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path,
        steps=["dataset_validation", "dataset_versioning"],
        dataset_name="test_dataset",
        dataset_registry_dir=str(tmp_path / "dataset_registry"),
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.COMPLETED
    versioning_step = next(s for s in result.steps if s.step == "dataset_versioning")
    assert versioning_step.status == "REGISTERED"


# ── 16-18. Required-field guards per step ─────────────────────────────────────────


def test_data_intake_without_raw_data_path_fails_clearly(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["data_intake"])
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    assert "raw_data_path" in result.errors[0]


def test_training_without_config_path_fails_clearly(tmp_path: Path) -> None:
    inp = _base_input(
        tmp_path, steps=["dataset_validation", "training"], training_config_path=None
    )
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    train_step = next(s for s in result.steps if s.step == "training")
    assert "training_config_path" in train_step.errors[0]


def test_evaluation_without_prior_training_fails_clearly(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["evaluation"])
    result = OrchestratorWorkflow().run(inp)
    assert result.status == OrchestratorStatus.FAILED
    assert "training" in result.errors[0]


# ── 19. Transition legality (white-box) ────────────────────────────────────────────


def test_transition_legality_table() -> None:
    wf = OrchestratorWorkflow()
    steps = ["dataset_validation", "training"]
    assert wf._is_legal("NEW", "DATASET_VALIDATION_RUNNING", steps) is True
    assert wf._is_legal("NEW", "TRAINING_RUNNING", steps) is False
    assert wf._is_legal("DATASET_VALIDATION_RUNNING", "DATASET_VALIDATION_COMPLETED", steps) is True
    assert wf._is_legal("DATASET_VALIDATION_RUNNING", "TRAINING_RUNNING", steps) is False
    assert wf._is_legal("DATASET_VALIDATION_COMPLETED", "TRAINING_RUNNING", steps) is True
    assert wf._is_legal("TRAINING_COMPLETED", "COMPLETED", steps) is True
    assert wf._is_legal("TRAINING_FAILED", "COMPLETED", steps) is False
    assert wf._is_legal("APPROVAL_RUNNING", "MODEL_APPROVAL_REQUIRED", ["approval"]) is True
    assert wf._is_legal("MODEL_APPROVAL_REQUIRED", "APPROVAL_RUNNING", ["approval"]) is True


# ── 20. Report artifacts ────────────────────────────────────────────────────────────


def test_orchestrator_report_written(tmp_path: Path) -> None:
    inp = _base_input(tmp_path, steps=["dataset_validation"])
    result = OrchestratorWorkflow().run(inp)
    report_paths = [a for a in result.artifacts if "orchestrator_report" in a]
    assert any(p.endswith(".json") for p in report_paths)
    assert any(p.endswith(".md") for p in report_paths)
    for p in report_paths:
        assert Path(p).exists()


# ── 21. from_yaml ────────────────────────────────────────────────────────────────


def test_orchestrator_input_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "orchestrator.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "steps": ["dataset_validation"],
                "dataset_path": "/x",
                "data_yaml_path": "/x/data.yaml",
            }
        ),
        encoding="utf-8",
    )
    inp = OrchestratorInput.from_yaml(cfg_path, workflow_id="wf_yaml", resume=True)
    assert inp.workflow_id == "wf_yaml"
    assert inp.resume is True
    assert inp.steps == ["dataset_validation"]
    assert inp.dataset_path == "/x"


# ── 22-24. CLI ─────────────────────────────────────────────────────────────────


def test_cli_run_workflow_pauses_at_approval(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    dataset = _make_dataset(tmp_path)
    training_cfg = _make_training_config(tmp_path)
    cfg_path = tmp_path / "orchestrator.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "steps": ["dataset_validation", "training", "evaluation", "approval"],
                "dry_run": True,
                "dataset_path": str(dataset),
                "data_yaml_path": str(dataset / "data.yaml"),
                "training_config_path": str(training_cfg),
                "interactive_approval": False,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-workflow",
            "--workflow-id", "wf_cli_test",
            "--config", str(cfg_path),
            "--runs-dir", str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PENDING_APPROVAL" in result.output


def test_cli_run_workflow_exits_1_on_bad_config(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "run-workflow",
            "--workflow-id", "wf_cli_bad",
            "--config", str(tmp_path / "nope.yaml"),
            "--runs-dir", str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1


def test_cli_run_workflow_exits_1_without_resume_on_rerun(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    dataset = _make_dataset(tmp_path)
    cfg_path = tmp_path / "orchestrator.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "steps": ["dataset_validation"],
                "dataset_path": str(dataset),
                "data_yaml_path": str(dataset / "data.yaml"),
            }
        ),
        encoding="utf-8",
    )
    args = [
        "run-workflow",
        "--workflow-id", "wf_cli_rerun",
        "--config", str(cfg_path),
        "--runs-dir", str(tmp_path / "runs"),
    ]
    runner = CliRunner()
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, args)
    assert second.exit_code == 1
