"""Integration tests: MVPWorkflow end-to-end without Azure or live YOLO.

All tests use ``dry_run=True`` so no GPU, no ultralytics required.
Tests exercise the real workflow orchestration with real agents — not stubs —
except where a specific failure mode is under test (training failure case).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.approvals import ApprovalAction
from agentic_mlops.contracts.training import TrainingJobStatus, TrainingOutput
from agentic_mlops.contracts.workflows import MVPWorkflowInput, MVPWorkflowStatus
from agentic_mlops.workflows.mvp_workflow import MVPWorkflow
from tests.conftest import make_valid_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_training_config(path: Path) -> Path:
    cfg_path = path / "training.yaml"
    cfg_path.write_text(
        yaml.dump({"model": "yolo11m.pt", "epochs": 1}),
        encoding="utf-8",
    )
    return cfg_path


def _base_input(
    dataset_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    approval_action: ApprovalAction = ApprovalAction.APPROVE_MODEL,
) -> MVPWorkflowInput:
    return MVPWorkflowInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(dataset_path / "data.yaml"),
        training_config_path=str(config_path),
        output_dir=str(output_dir),
        dry_run=True,
        interactive_approval=False,
        approval_action=approval_action,
    )


class _StubTrainingFailAgent(BaseAgent):
    """Injects a training failure without touching the filesystem."""

    def run(self, inp: object) -> TrainingOutput:  # noqa: A002
        return TrainingOutput(
            success=False,
            message="Simulated GPU OOM",
            job_status=TrainingJobStatus.FAILED,
            errors=["CUDA out of memory"],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMVPWorkflowE2E:
    def test_dry_run_completes_successfully(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        make_valid_dataset(dataset)
        cfg = _make_training_config(tmp_path)

        result = MVPWorkflow().run(_base_input(dataset, cfg, tmp_path / "out"))

        assert result.workflow_status == MVPWorkflowStatus.COMPLETED
        assert result.success

    def test_all_pipeline_dirs_created(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        make_valid_dataset(dataset)
        cfg = _make_training_config(tmp_path)
        out = tmp_path / "out"

        MVPWorkflow().run(_base_input(dataset, cfg, out))

        for step_dir in ("validation", "training", "evaluation"):
            assert (out / step_dir).is_dir(), f"Missing {step_dir}/ dir"

    def test_workflow_stops_on_dataset_validation_failure(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        make_valid_dataset(dataset)
        (dataset / "data.yaml").unlink()  # corrupt the dataset
        cfg = _make_training_config(tmp_path)

        result = MVPWorkflow().run(_base_input(dataset, cfg, tmp_path / "out"))

        assert result.workflow_status == MVPWorkflowStatus.FAILED
        step_names = {s.step for s in result.steps}
        assert "validation" in step_names
        assert "training" not in step_names

    def test_workflow_stops_when_training_fails(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        make_valid_dataset(dataset)
        cfg = _make_training_config(tmp_path)

        wf = MVPWorkflow(_training_factory=lambda d: _StubTrainingFailAgent(d))
        result = wf.run(_base_input(dataset, cfg, tmp_path / "out"))

        assert result.workflow_status == MVPWorkflowStatus.FAILED
        by_step = {s.step: s for s in result.steps}
        assert by_step["training"].success is False
        assert "evaluation" not in by_step

    def test_output_dir_created_automatically(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        make_valid_dataset(dataset)
        cfg = _make_training_config(tmp_path)
        nested = tmp_path / "a" / "b" / "c"

        result = MVPWorkflow().run(_base_input(dataset, cfg, nested))

        assert nested.is_dir()
        assert result.workflow_status in (
            MVPWorkflowStatus.COMPLETED,
            MVPWorkflowStatus.FAILED,
        )
