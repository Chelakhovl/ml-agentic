"""CostTracker — accumulates per-step cost estimates for a workflow run.

Lightweight; no Azure SDK dependency.  Cost estimates are based on:
  - duration (parsed from ISO-format started_at / completed_at strings)
  - hourly rate for the compute type (ComputePricing)
  - instance count

For Azure ML training the relevant timing fields in TrainingOutput are
``remote_started_at`` / ``remote_completed_at``; for EvaluationOutput they
are ``started_at`` / ``completed_at``.  If either timestamp is None (e.g.
for fake or local runners), the cost is recorded as zero.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.contracts.cost import ComputePricing, CostEntry, CostSummary
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


def _parse_duration(started_at: str | None, completed_at: str | None) -> float | None:
    if not started_at or not completed_at:
        return None
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.fromisoformat(completed_at)
        return max(0.0, (t1 - t0).total_seconds())
    except (ValueError, TypeError):
        return None


class CostTracker:
    """Accumulates CostEntry records and writes a JSON cost summary."""

    def __init__(self, pricing: ComputePricing | None = None) -> None:
        self._pricing = pricing or ComputePricing()
        self._entries: list[CostEntry] = []

    def record(
        self,
        step: str,
        runner: str,
        *,
        compute_type: str = "fake",
        instance_count: int = 1,
        started_at: str | None = None,
        completed_at: str | None = None,
        notes: str = "",
    ) -> CostEntry:
        """Record cost for one completed step and return the entry."""
        duration = _parse_duration(started_at, completed_at)
        hourly_rate = self._pricing.rate_for(compute_type)
        cost = 0.0
        if duration is not None and hourly_rate > 0:
            cost = round((duration / 3600.0) * hourly_rate * instance_count, 6)
        entry = CostEntry(
            step=step,
            runner=runner,
            compute_type=compute_type,
            instance_count=instance_count,
            duration_seconds=duration,
            estimated_cost=cost,
            currency=self._pricing.currency,
            notes=notes,
        )
        self._entries.append(entry)
        logger.debug(
            "Cost recorded",
            extra={"step": step, "cost": cost, "currency": self._pricing.currency},
        )
        return entry

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    def summary(self, workflow_id: str) -> CostSummary:
        total = round(sum(e.estimated_cost for e in self._entries), 6)
        return CostSummary(
            workflow_id=workflow_id,
            total_cost=total,
            currency=self._pricing.currency,
            entries=list(self._entries),
            generated_at=datetime.now(tz=UTC).isoformat(),
        )

    def write_report(self, workflow_id: str, output_dir: Path) -> Path:
        """Serialise the cost summary to ``output_dir/cost_summary.json``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.summary(workflow_id).model_dump()
        path = output_dir / "cost_summary.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class FakeCostTracker:
    """In-memory test double for CostTracker.

    Exposes the same public API.  Does not apply pricing; all cost estimates
    are 0.0.  Recorded calls are stored in ``self.recorded`` for assertions.
    """

    def __init__(self) -> None:
        self.recorded: list[dict] = []
        self._entries: list[CostEntry] = []

    def record(
        self,
        step: str,
        runner: str,
        *,
        compute_type: str = "fake",
        instance_count: int = 1,
        started_at: str | None = None,
        completed_at: str | None = None,
        notes: str = "",
    ) -> CostEntry:
        self.recorded.append(
            dict(
                step=step,
                runner=runner,
                compute_type=compute_type,
                instance_count=instance_count,
                started_at=started_at,
                completed_at=completed_at,
                notes=notes,
            )
        )
        entry = CostEntry(
            step=step,
            runner=runner,
            compute_type=compute_type,
            instance_count=instance_count,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    def summary(self, workflow_id: str) -> CostSummary:
        return CostSummary(
            workflow_id=workflow_id,
            total_cost=0.0,
            currency="USD",
            entries=list(self._entries),
            generated_at=datetime.now(tz=UTC).isoformat(),
        )

    def write_report(self, workflow_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "cost_summary.json"
        path.write_text(
            json.dumps(self.summary(workflow_id).model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
