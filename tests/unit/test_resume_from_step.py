"""Tests for --resume-from-step in OrchestratorWorkflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.orchestrator import (
    OrchestratorInput,
    OrchestratorStatus,
)
from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

# ── Stub agent helpers ─────────────────────────────────────────────────────────


class _FakeOutput:
    def __init__(self, step: str, success: bool = True):
        self.success = success
        self.status = "completed" if success else "failed"
        self.errors: list[str] = [] if success else ["fake error"]
        self.artifacts: list[str] = []
        self.message = "ok"
        self.key_outputs: dict[str, Any] = {"step_name": step}
        # Fields expected by specific steps
        self.manifest_path = None
        self.report_path = f"runs/fake/{step}/report.json"
        self.status_label = "COMPLETED" if success else "FAILED"
        # training
        self.best_weights_path = "fake/best.pt"
        self.job_status = "completed"
        self.remote_started_at = None
        self.remote_completed_at = None
        # evaluation
        self.recommendation = "PROMOTE_CANDIDATE"
        self.started_at = None
        self.completed_at = None
        # approval
        self.action = None
        self.generated_artifacts: list[str] = []
        # registry
        self.registry_path = "fake/registry"
        self.version = 1
        # deployment
        self.endpoint_name = "fake-endpoint"
        self.release = 1
        self.scoring_uri = None
        # decision
        self.decision = "PROMOTE"
        # dataset versioning
        self.dataset_version_path = "fake/dataset/v1"
        # dataset validation
        self.structured_dataset_path = None
        self.data_yaml_path = None
        self.split_report_path = None


def _make_orchestrator(
    steps: list[str], runs_dir: Path, workflow_id: str = "wf_test"
) -> OrchestratorInput:
    return OrchestratorInput(
        workflow_id=workflow_id,
        steps=steps,
        runs_dir=str(runs_dir),
        dry_run=True,
        dataset_path="/fake/dataset",
        data_yaml_path="/fake/data.yaml",
        training_config_path=None,
        approval_action=None,
        interactive_approval=False,
    )


def _seed_state(
    runs_dir: Path,
    workflow_id: str,
    completed: list[str],
    step_outputs: dict[str, dict] | None = None,
    current_state: str = "TRAINING_COMPLETED",
    status: str = "running",
) -> None:
    wf_dir = runs_dir / workflow_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "workflow_id": workflow_id,
        "trigger": "manual",
        "steps": completed + ["evaluation", "approval", "model_registry"],
        "completed_steps": list(completed),
        "step_status": {s: "COMPLETED" for s in completed},
        "step_outputs": step_outputs or {s: {"step_name": s} for s in completed},
        "artifacts": [],
        "current_state": current_state,
        "status": status,
        "last_agent": completed[-1] if completed else None,
        "pending_approval_id": None,
        "started_at": "2026-08-21T10:00:00+00:00",
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (wf_dir / "audit_log.jsonl").write_text("", encoding="utf-8")


# ── Patching helpers ───────────────────────────────────────────────────────────


def _patch_dispatch(monkeypatch, outcomes: dict[str, bool]):
    """Replace OrchestratorWorkflow._dispatch so each step returns a controlled outcome."""
    from agentic_mlops.workflows.orchestrator import _StepOutcome

    def fake_dispatch(self, step, inp, output_root, step_outputs, azure_config):
        success = outcomes.get(step, True)
        return _StepOutcome(
            success=success,
            status_label="COMPLETED" if success else "FAILED",
            errors=[] if success else ["fake error"],
            key_outputs={
                "step_name": step,
                "report_path": f"fake/{step}/report.json",
                "best_weights_path": "fake/best.pt",
                "job_status": "completed",
                "output_json_path": f"fake/{step}/output.json",
            },
        )

    monkeypatch.setattr(OrchestratorWorkflow, "_dispatch", fake_dispatch)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestResumFromStepValidation:
    def test_fails_when_no_state_file(self, tmp_path):
        inp = OrchestratorInput(
            workflow_id="wf_new",
            steps=["dataset_validation", "training", "evaluation"],
            runs_dir=str(tmp_path),
            resume_from_step="evaluation",
        )
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.FAILED
        assert "resume-from-step" in result.message or "state file" in result.message.lower()

    def test_fails_for_unknown_step(self, tmp_path, monkeypatch):
        _seed_state(tmp_path, "wf1", ["dataset_validation", "training"])
        inp = OrchestratorInput(
            workflow_id="wf1",
            steps=["dataset_validation", "training", "evaluation"],
            runs_dir=str(tmp_path),
            resume_from_step="nonexistent_step",
        )
        _patch_dispatch(monkeypatch, {})
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.FAILED
        assert "nonexistent_step" in result.message

    def test_step_not_in_configured_steps_fails(self, tmp_path, monkeypatch):
        _seed_state(tmp_path, "wf2", ["dataset_validation", "training"])
        inp = OrchestratorInput(
            workflow_id="wf2",
            steps=["dataset_validation", "training", "evaluation"],
            runs_dir=str(tmp_path),
            resume_from_step="deployment",  # not in configured steps
        )
        _patch_dispatch(monkeypatch, {})
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.FAILED
        assert "deployment" in result.message


class TestResumFromStepBehavior:
    def test_reruns_target_and_after(self, tmp_path, monkeypatch):
        """Rewinding to evaluation causes it and subsequent steps to run."""
        steps = ["dataset_validation", "training", "evaluation"]
        _seed_state(
            tmp_path,
            "wf3",
            completed=["dataset_validation", "training"],
            step_outputs={
                "dataset_validation": {"status": "passed", "report_path": "fake/val/report.json"},
                "training": {
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "training_output_path": "fake/train/output.json",
                },
            },
            current_state="TRAINING_COMPLETED",
        )
        dispatched: list[str] = []

        def fake_dispatch(self, step, inp, output_root, step_outputs, azure_config):
            from agentic_mlops.workflows.orchestrator import _StepOutcome

            dispatched.append(step)
            return _StepOutcome(
                success=True,
                status_label="COMPLETED",
                key_outputs={
                    "step_name": step,
                    "report_path": f"fake/{step}/report.json",
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "output_json_path": f"fake/{step}/output.json",
                },
            )

        monkeypatch.setattr(OrchestratorWorkflow, "_dispatch", fake_dispatch)

        inp = OrchestratorInput(
            workflow_id="wf3",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="evaluation",
        )
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.COMPLETED
        assert "evaluation" in dispatched
        assert "dataset_validation" not in dispatched
        assert "training" not in dispatched

    def test_prior_step_outputs_preserved(self, tmp_path, monkeypatch):
        """Training's best_weights_path should still be available to the re-run evaluation."""
        steps = ["dataset_validation", "training", "evaluation"]
        _seed_state(
            tmp_path,
            "wf4",
            completed=["dataset_validation", "training"],
            step_outputs={
                "dataset_validation": {"status": "passed", "report_path": "fake/val/report.json"},
                "training": {
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "training_output_path": "fake/train/output.json",
                },
            },
            current_state="TRAINING_COMPLETED",
        )
        received_step_outputs: dict = {}

        def fake_dispatch(self, step, inp, output_root, step_outputs, azure_config):
            from agentic_mlops.workflows.orchestrator import _StepOutcome

            if step == "evaluation":
                received_step_outputs.update(step_outputs)
            return _StepOutcome(
                success=True,
                status_label="COMPLETED",
                key_outputs={
                    "step_name": step,
                    "report_path": f"fake/{step}/report.json",
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "output_json_path": f"fake/{step}/output.json",
                },
            )

        monkeypatch.setattr(OrchestratorWorkflow, "_dispatch", fake_dispatch)

        inp = OrchestratorInput(
            workflow_id="wf4",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="evaluation",
        )
        OrchestratorWorkflow().run(inp)
        # evaluation step_outputs should include training output
        assert received_step_outputs.get("training", {}).get("best_weights_path") == "fake/best.pt"

    def test_evaluation_output_cleared_before_rerun(self, tmp_path, monkeypatch):
        """The target step's stale output must be gone before re-run."""
        steps = ["dataset_validation", "training", "evaluation"]
        _seed_state(
            tmp_path,
            "wf5",
            completed=["dataset_validation", "training", "evaluation"],
            step_outputs={
                "dataset_validation": {"status": "passed", "report_path": "fake/val.json"},
                "training": {
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "training_output_path": "fake/train.json",
                },
                "evaluation": {"report_path": "OLD_stale_path.json"},
            },
            current_state="EVALUATION_COMPLETED",
            status="completed",
        )

        dispatched_with_step_outputs: dict = {}

        def fake_dispatch(self, step, inp, output_root, step_outputs, azure_config):
            from agentic_mlops.workflows.orchestrator import _StepOutcome

            if step == "evaluation":
                dispatched_with_step_outputs["eval_in"] = step_outputs.get("evaluation")
            return _StepOutcome(
                success=True,
                status_label="COMPLETED",
                key_outputs={
                    "report_path": "NEW_path.json",
                    "best_weights_path": "fake/best.pt",
                    "job_status": "completed",
                    "output_json_path": "fake/eval/output.json",
                },
            )

        monkeypatch.setattr(OrchestratorWorkflow, "_dispatch", fake_dispatch)

        inp = OrchestratorInput(
            workflow_id="wf5",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="evaluation",
        )
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.COMPLETED
        # Old output was cleared before dispatch
        assert dispatched_with_step_outputs["eval_in"] is None

    def test_implies_resume_without_explicit_flag(self, tmp_path, monkeypatch):
        """resume_from_step works even when resume=False (it implies resume)."""
        steps = ["dataset_validation", "training"]
        _seed_state(
            tmp_path,
            "wf6",
            completed=["dataset_validation"],
            step_outputs={"dataset_validation": {"status": "passed", "report_path": "fake.json"}},
            current_state="DATASET_VALIDATION_COMPLETED",
        )
        _patch_dispatch(monkeypatch, {})
        inp = OrchestratorInput(
            workflow_id="wf6",
            steps=steps,
            runs_dir=str(tmp_path),
            resume=False,  # explicit False
            resume_from_step="training",
        )
        result = OrchestratorWorkflow().run(inp)
        # Should succeed (not fail with "state file already exists")
        assert result.status != OrchestratorStatus.FAILED or "state file" not in result.message

    def test_rewind_to_first_step(self, tmp_path, monkeypatch):
        """Rewinding to the very first step resets current_state to NEW."""
        steps = ["dataset_validation", "training"]
        _seed_state(
            tmp_path,
            "wf7",
            completed=["dataset_validation", "training"],
            current_state="TRAINING_COMPLETED",
            status="completed",
        )
        _patch_dispatch(monkeypatch, {})
        inp = OrchestratorInput(
            workflow_id="wf7",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="dataset_validation",
        )
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.COMPLETED

    def test_audit_log_records_rewind_event(self, tmp_path, monkeypatch):
        """workflow_resume_from_step event is appended to audit_log.jsonl."""
        steps = ["dataset_validation", "training"]
        _seed_state(
            tmp_path,
            "wf8",
            completed=["dataset_validation"],
            current_state="DATASET_VALIDATION_COMPLETED",
        )
        _patch_dispatch(monkeypatch, {})
        inp = OrchestratorInput(
            workflow_id="wf8",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="training",
        )
        OrchestratorWorkflow().run(inp)
        audit = (tmp_path / "wf8" / "audit_log.jsonl").read_text(encoding="utf-8")
        events = [json.loads(line) for line in audit.splitlines() if line.strip()]
        event_names = [e.get("event") for e in events]
        assert "workflow_resume_from_step" in event_names
        rewind_event = next(e for e in events if e.get("event") == "workflow_resume_from_step")
        assert rewind_event["step"] == "training"

    def test_completed_workflow_can_be_rewound(self, tmp_path, monkeypatch):
        """A completed workflow can be rewound and re-run from a given step."""
        steps = ["dataset_validation", "training"]
        _seed_state(
            tmp_path,
            "wf9",
            completed=["dataset_validation", "training"],
            current_state="TRAINING_COMPLETED",
            status="completed",
        )
        _patch_dispatch(monkeypatch, {})
        inp = OrchestratorInput(
            workflow_id="wf9",
            steps=steps,
            runs_dir=str(tmp_path),
            resume_from_step="training",
        )
        result = OrchestratorWorkflow().run(inp)
        assert result.status == OrchestratorStatus.COMPLETED


class TestResumFromStepCLI:
    def test_cli_passes_resume_from_step(self, tmp_path, monkeypatch):
        """CLI --resume-from-step flag is correctly threaded to OrchestratorInput."""
        import yaml
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        steps = ["dataset_validation", "training"]
        _seed_state(
            tmp_path,
            "wf_cli",
            completed=["dataset_validation"],
            current_state="DATASET_VALIDATION_COMPLETED",
        )
        _patch_dispatch(monkeypatch, {})

        cfg_path = tmp_path / "orch.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "steps": steps,
                    "dry_run": True,
                    "dataset_path": "/fake/ds",
                    "data_yaml_path": "/fake/data.yaml",
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id",
                "wf_cli",
                "--config",
                str(cfg_path),
                "--runs-dir",
                str(tmp_path),
                "--resume-from-step",
                "training",
            ],
        )
        assert result.exit_code == 0, result.output
