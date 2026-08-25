"""WorkflowRunDiffer — compares two workflow run state files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepDiff:
    name: str
    status_a: str | None  # None = step not in that run
    status_b: str | None
    changed: bool


@dataclass
class MetricDiff:
    metric: str  # e.g. "evaluation.map50"
    value_a: float | None
    value_b: float | None
    delta: float | None  # value_b - value_a; None when either is missing
    improved: bool | None  # True = higher is better and b > a; None = unknown


@dataclass
class RunDiff:
    wf_a: str
    wf_b: str
    status_a: str
    status_b: str
    started_at_a: str | None
    started_at_b: str | None
    step_diffs: list[StepDiff] = field(default_factory=list)
    metric_diffs: list[MetricDiff] = field(default_factory=list)
    # Steps present in one run but not the other
    steps_only_in_a: list[str] = field(default_factory=list)
    steps_only_in_b: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return (
            self.status_a != self.status_b
            or any(d.changed for d in self.step_diffs)
            or bool(self.steps_only_in_a or self.steps_only_in_b)
        )


# Metrics to extract per step, with (lower_is_better) flag.
_STEP_METRICS: dict[str, list[tuple[str, bool]]] = {
    "evaluation": [
        ("map50", False),
        ("precision", False),
        ("recall", False),
        ("map50_95", False),
    ],
    "dataset_validation": [
        ("total_images", False),
        ("total_labels", False),
        ("blocking_issues_count", True),  # lower is better
    ],
    "model_registry": [
        ("version", False),
    ],
}


class WorkflowRunDiffer:
    """Reads two state.json files and returns a RunDiff."""

    def diff(
        self,
        runs_dir: str | Path,
        wf_id_a: str,
        wf_id_b: str,
    ) -> RunDiff | None:
        """Return a RunDiff, or None if either workflow_id is not found."""
        runs_dir = Path(runs_dir)
        state_a = self._read_state(runs_dir, wf_id_a)
        state_b = self._read_state(runs_dir, wf_id_b)
        if state_a is None or state_b is None:
            return None

        steps_a = set(state_a.get("steps", []))
        steps_b = set(state_b.get("steps", []))
        all_steps = [s for s in _PIPELINE_ORDER if s in steps_a | steps_b]

        step_status_a = self._resolve_step_statuses(state_a)
        step_status_b = self._resolve_step_statuses(state_b)

        step_diffs: list[StepDiff] = []
        for step in all_steps:
            sa = step_status_a.get(step)
            sb = step_status_b.get(step)
            step_diffs.append(
                StepDiff(
                    name=step,
                    status_a=sa,
                    status_b=sb,
                    changed=(sa != sb),
                )
            )

        metric_diffs = self._extract_metric_diffs(
            state_a.get("step_outputs", {}),
            state_b.get("step_outputs", {}),
        )

        return RunDiff(
            wf_a=wf_id_a,
            wf_b=wf_id_b,
            status_a=state_a.get("status", "unknown"),
            status_b=state_b.get("status", "unknown"),
            started_at_a=state_a.get("started_at"),
            started_at_b=state_b.get("started_at"),
            step_diffs=step_diffs,
            metric_diffs=metric_diffs,
            steps_only_in_a=sorted(steps_a - steps_b),
            steps_only_in_b=sorted(steps_b - steps_a),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _read_state(runs_dir: Path, wf_id: str) -> dict[str, Any] | None:
        p = runs_dir / wf_id / "state.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _resolve_step_statuses(state: dict) -> dict[str, str]:
        """Return {step: resolved_status} for steps in this run."""
        completed = set(state.get("completed_steps", []))
        raw_status = state.get("step_status", {})
        result: dict[str, str] = {}
        for step in state.get("steps", []):
            rs = raw_status.get(step, "").upper()
            if step in completed:
                if rs in ("FAILED", "BLOCKED"):
                    result[step] = "failed"
                elif rs == "SKIPPED":
                    result[step] = "skipped"
                else:
                    result[step] = "completed"
            elif rs == "RUNNING":
                result[step] = "running"
            elif rs in ("FAILED", "BLOCKED"):
                result[step] = "failed"
            elif rs == "SKIPPED":
                result[step] = "skipped"
            else:
                result[step] = "pending"
        return result

    @staticmethod
    def _extract_metric_diffs(
        outputs_a: dict[str, Any],
        outputs_b: dict[str, Any],
    ) -> list[MetricDiff]:
        diffs: list[MetricDiff] = []
        for step, metrics in _STEP_METRICS.items():
            out_a = outputs_a.get(step, {})
            out_b = outputs_b.get(step, {})
            if not out_a and not out_b:
                continue
            for key, lower_is_better in metrics:
                va = _to_float(out_a.get(key))
                vb = _to_float(out_b.get(key))
                delta = (vb - va) if va is not None and vb is not None else None
                if delta is not None:
                    improved = (delta < 0) if lower_is_better else (delta > 0)
                else:
                    improved = None
                if va is not None or vb is not None:
                    diffs.append(
                        MetricDiff(
                            metric=f"{step}.{key}",
                            value_a=va,
                            value_b=vb,
                            delta=delta,
                            improved=improved,
                        )
                    )
        return diffs


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Canonical pipeline order for display (matches PIPELINE_STEPS in contracts).
_PIPELINE_ORDER = [
    "data_intake",
    "dataset_structuring",
    "dataset_validation",
    "dataset_versioning",
    "training_approval",
    "training",
    "evaluation",
    "model_decision",
    "approval",
    "model_registry",
    "deployment",
]
