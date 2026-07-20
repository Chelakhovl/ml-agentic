"""Pydantic contract for webhook notification configuration.

Used by OrchestratorWorkflow to send Teams/Slack notifications on key pipeline events.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NotificationConfig(BaseModel):
    """Webhook notification settings for the Orchestrator.

    Set ``teams_webhook_url`` and/or ``slack_webhook_url`` to enable
    platform-specific notifications. ``notify_on`` controls which events
    trigger a message — default is workflow-level events + step failures.

    Verbose mode: add "step_started" and "step_completed" to notify_on to
    receive a message for every individual step.
    """

    enabled: bool = True
    teams_webhook_url: str | None = None
    slack_webhook_url: str | None = None
    notify_on: list[str] = Field(
        default_factory=lambda: [
            "step_failed",
            "workflow_completed",
            "workflow_failed",
            "workflow_blocked",
            "workflow_pending_approval",
        ]
    )
