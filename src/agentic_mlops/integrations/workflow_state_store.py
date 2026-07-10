"""Workflow state store for the Orchestrator — a local JSON state file + JSONL audit log.

Per agentic_mlops_workflow_docs/agents/00_orchestrator_agent.md's "MVP
implementation" note: no networked State Store / Audit Log service exists —
just two files per workflow run:

    runs/<workflow_id>/state.json        current state snapshot (overwritten each step)
    runs/<workflow_id>/audit_log.jsonl   append-only event history, one JSON object per line
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class WorkflowStateStore:
    """Reads/writes state.json and appends to audit_log.jsonl for one workflow_id."""

    def __init__(self, runs_dir: Path, workflow_id: str) -> None:
        self.workflow_dir = Path(runs_dir) / workflow_id
        self.state_path = self.workflow_dir / "state.json"
        self.audit_log_path = self.workflow_dir / "audit_log.jsonl"

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict[str, Any]) -> None:
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now(tz=UTC).isoformat()
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def append_audit(self, event: dict[str, Any]) -> None:
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": datetime.now(tz=UTC).isoformat(), **event}
        with self.audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        # "message"/"asctime" are reserved LogRecord attribute names — logging
        # a dict that happens to contain one as `extra` raises a KeyError.
        # The JSONL line above (the actual audit record) keeps the original keys.
        log_extra = {("detail" if k == "message" else k): v for k, v in event.items()}
        logger.info("Workflow audit event", extra=log_extra)
