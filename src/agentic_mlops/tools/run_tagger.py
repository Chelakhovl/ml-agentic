"""WorkflowRunTagger — read/write arbitrary tags on a workflow's state.json."""

from __future__ import annotations

import json
from pathlib import Path


class WorkflowRunTagger:
    """Add, remove, or read tags stored in runs/<wf_id>/state.json.

    Tags live under the ``"tags"`` key in state.json as a flat
    ``{str: str}`` dict.  Values are always strings; callers that want
    typed values should parse them after reading.
    """

    def add(self, runs_dir: str | Path, workflow_id: str, tags: dict[str, str]) -> dict[str, str]:
        """Merge *tags* into the run's existing tags and return the full tag dict."""
        state = self._load(runs_dir, workflow_id)
        existing: dict[str, str] = state.get("tags") or {}
        existing.update(tags)
        state["tags"] = existing
        self._save(runs_dir, workflow_id, state)
        return dict(existing)

    def remove(self, runs_dir: str | Path, workflow_id: str, keys: list[str]) -> dict[str, str]:
        """Delete *keys* from the run's tags and return the remaining tag dict."""
        state = self._load(runs_dir, workflow_id)
        existing: dict[str, str] = state.get("tags") or {}
        for k in keys:
            existing.pop(k, None)
        state["tags"] = existing
        self._save(runs_dir, workflow_id, state)
        return dict(existing)

    def get(self, runs_dir: str | Path, workflow_id: str) -> dict[str, str]:
        """Return the current tags for the run (empty dict if none set)."""
        state = self._load(runs_dir, workflow_id)
        return dict(state.get("tags") or {})

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _state_path(runs_dir: str | Path, workflow_id: str) -> Path:
        return Path(runs_dir) / workflow_id / "state.json"

    def _load(self, runs_dir: str | Path, workflow_id: str) -> dict:
        p = self._state_path(runs_dir, workflow_id)
        if not p.exists():
            raise FileNotFoundError(
                f"state.json not found for workflow '{workflow_id}' in {runs_dir}"
            )
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _save(runs_dir: str | Path, workflow_id: str, state: dict) -> None:
        p = Path(runs_dir) / workflow_id / "state.json"
        p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
