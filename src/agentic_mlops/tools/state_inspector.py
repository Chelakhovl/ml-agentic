"""WorkflowStateInspector — reads state.json + audit_log.jsonl for display."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepSummary:
    name: str
    status: str  # "completed" | "failed" | "skipped" | "running" | "pending"
    artifacts: list[str] = field(default_factory=list)


@dataclass
class WorkflowSnapshot:
    workflow_id: str
    status: str  # OrchestratorStatus value
    current_state: str
    steps: list[str]
    step_summaries: list[StepSummary]
    pending_approval_id: str | None
    started_at: str | None
    updated_at: str | None
    mlflow_run_id: str | None
    audit_events: list[dict[str, Any]]
    state_path: Path
    audit_log_path: Path

    @property
    def num_completed(self) -> int:
        return sum(1 for s in self.step_summaries if s.status == "completed")

    @property
    def num_failed(self) -> int:
        return sum(1 for s in self.step_summaries if s.status == "failed")

    @property
    def num_skipped(self) -> int:
        return sum(1 for s in self.step_summaries if s.status == "skipped")


class WorkflowStateInspector:
    """Reads workflow state from disk and returns a WorkflowSnapshot."""

    def inspect(
        self,
        runs_dir: str | Path,
        workflow_id: str,
        max_audit_events: int = 20,
    ) -> WorkflowSnapshot | None:
        runs_dir = Path(runs_dir)
        wf_dir = runs_dir / workflow_id
        state_path = wf_dir / "state.json"
        audit_log_path = wf_dir / "audit_log.jsonl"

        if not state_path.exists():
            return None

        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))

        configured_steps: list[str] = state.get("steps", [])
        completed_steps: set[str] = set(state.get("completed_steps", []))
        raw_step_status: dict[str, str] = state.get("step_status", {})
        step_outputs: dict[str, Any] = state.get("step_outputs", {})

        step_summaries: list[StepSummary] = []
        for step in configured_steps:
            raw_status = raw_step_status.get(step, "").upper()

            if step in completed_steps:
                if raw_status in ("FAILED", "BLOCKED"):
                    status = "failed"
                elif raw_status == "SKIPPED":
                    status = "skipped"
                else:
                    status = "completed"
            elif raw_status == "RUNNING":
                status = "running"
            elif raw_status in ("FAILED", "BLOCKED"):
                status = "failed"
            elif raw_status == "SKIPPED":
                status = "skipped"
            else:
                status = "pending"

            # collect notable artifact paths from step_outputs
            artifacts: list[str] = []
            out = step_outputs.get(step, {})
            for key in (
                "report_path",
                "best_weights_path",
                "training_output_path",
                "evaluation_output_path",
                "dataset_version_path",
                "registry_path",
                "output_json_path",
            ):
                v = out.get(key)
                if v:
                    artifacts.append(v)

            step_summaries.append(StepSummary(name=step, status=status, artifacts=artifacts))

        # audit events — last N lines
        audit_events: list[dict[str, Any]] = []
        if audit_log_path.exists():
            lines = audit_log_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-max_audit_events:]:
                line = line.strip()
                if line:
                    try:
                        audit_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        return WorkflowSnapshot(
            workflow_id=workflow_id,
            status=state.get("status", "unknown"),
            current_state=state.get("current_state", "unknown"),
            steps=configured_steps,
            step_summaries=step_summaries,
            pending_approval_id=state.get("pending_approval_id"),
            started_at=state.get("started_at"),
            updated_at=state.get("updated_at"),
            mlflow_run_id=state.get("mlflow_run_id"),
            audit_events=audit_events,
            state_path=state_path,
            audit_log_path=audit_log_path,
        )

    def list_workflows(self, runs_dir: str | Path) -> list[str]:
        """Return workflow IDs (subdirs containing state.json), newest-modified first."""
        runs_dir = Path(runs_dir)
        if not runs_dir.exists():
            return []
        ids = [
            d.name
            for d in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            if d.is_dir() and (d / "state.json").exists()
        ]
        return ids
