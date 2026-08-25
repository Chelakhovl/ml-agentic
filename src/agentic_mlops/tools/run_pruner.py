"""WorkflowPruner — delete old workflow run directories based on age or count."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path


@dataclass
class PruneResult:
    deleted: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # could not read / active
    dry_run: bool = False

    @property
    def deleted_count(self) -> int:
        return len(self.deleted)


class WorkflowPruner:
    """Delete workflow run directories from runs_dir based on age or count.

    At most one of keep_last / older_than_days may be given at once; using both
    raises ValueError. Runs whose state.json cannot be read are always kept (safe
    default).

    Args:
        keep_last: retain only the N most-recently-started runs; delete the rest.
        older_than_days: delete runs whose started_at is older than this many days.
        statuses: if given, only delete runs with these statuses (e.g. ["completed",
            "failed"]). None means no status filter.
        dry_run: compute what *would* be deleted without touching the filesystem.
    """

    def __init__(
        self,
        keep_last: int | None = None,
        older_than_days: float | None = None,
        statuses: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        if keep_last is not None and older_than_days is not None:
            raise ValueError("Specify keep_last OR older_than_days, not both.")
        if keep_last is not None and keep_last < 0:
            raise ValueError("keep_last must be >= 0.")
        self.keep_last = keep_last
        self.older_than_days = older_than_days
        self.statuses = [s.lower() for s in statuses] if statuses else None
        self.dry_run = dry_run

    def prune(self, runs_dir: str | Path) -> PruneResult:
        runs_dir = Path(runs_dir)
        result = PruneResult(dry_run=self.dry_run)

        if not runs_dir.exists():
            return result

        # Collect candidate dirs (any subdir containing state.json)
        candidates = [d for d in sorted(runs_dir.iterdir()) if d.is_dir()]
        run_info: list[tuple[str, str, str | None]] = []  # (wf_id, status, started_at)

        for d in candidates:
            state = _read_state(d)
            if state is None:
                result.skipped.append(d.name)
                continue
            run_info.append((d.name, state.get("status", "unknown"), state.get("started_at")))

        # Sort by started_at ascending (oldest first); None sorts last (kept)
        run_info.sort(key=lambda t: (t[2] is None, t[2] or ""))

        to_delete: set[str] = set()

        if self.keep_last is not None:
            # Keep the keep_last most-recent (end of sorted list)
            deletable = run_info[: max(0, len(run_info) - self.keep_last)]
            to_delete = {wf for wf, _, _ in deletable}

        elif self.older_than_days is not None:
            from datetime import datetime

            cutoff = datetime.now(tz=UTC).timestamp() - self.older_than_days * 86400
            for wf, _, started_at in run_info:
                if started_at is None:
                    continue
                try:
                    ts = datetime.fromisoformat(started_at).timestamp()
                    if ts < cutoff:
                        to_delete.add(wf)
                except (ValueError, TypeError):
                    pass

        # Apply status filter
        if self.statuses is not None and to_delete:
            status_map = {wf: st for wf, st, _ in run_info}
            to_delete = {wf for wf in to_delete if status_map.get(wf, "").lower() in self.statuses}

        for wf, status, _ in run_info:
            if wf in to_delete:
                if not self.dry_run:
                    shutil.rmtree(runs_dir / wf, ignore_errors=True)
                result.deleted.append(wf)
            else:
                result.kept.append(wf)

        return result


# ── helpers ───────────────────────────────────────────────────────────────────


def _read_state(run_dir: Path) -> dict | None:
    p = run_dir / "state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
