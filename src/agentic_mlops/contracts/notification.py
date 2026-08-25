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
    # Base URL of the web dashboard (e.g. "https://mlops.example.com").
    # When set, approval notifications include action buttons that open or
    # POST to this URL.  For Teams the button opens /approve/{workflow_id}
    # in the browser.  For Slack without slack_signing_secret it does the
    # same; with slack_signing_secret it sends interactive POST-back buttons
    # instead (requires the Slack app's Interactivity URL configured to
    # {callback_base_url}/webhooks/approve/slack).
    callback_base_url: str | None = None
    # Slack signing secret from the Slack app's "Basic Information" page.
    # Required for validating incoming interactive button callbacks.
    # When set together with callback_base_url, Slack approval messages use
    # interactive POST-back buttons instead of browser-link buttons.
    slack_signing_secret: str | None = None
    # Teams security token (base64-encoded symmetric key from the connector
    # configuration page). When set together with callback_base_url, Teams
    # approval messages include HttpPOST action buttons that POST the decision
    # directly to {callback_base_url}/webhooks/approve/teams.
    # Without it, only an OpenUri browser-link button is shown.
    teams_signing_secret: str | None = None

    # ── Email (SMTP) ──────────────────────────────────────────────────────────
    # Stdlib-only (smtplib / email.mime) — no extra packages required.
    # Leave smtp_host empty to disable email notifications.
    email_to: list[str] = Field(default_factory=list)
    email_from: str = "mlops-noreply@example.com"
    email_subject_prefix: str = "[MLOps]"
    smtp_host: str = ""
    smtp_port: int = 0  # 0 = auto (465 for ssl, 587 for tls, 25 plain)
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_tls: bool = True  # STARTTLS (typical for port 587)
    smtp_ssl: bool = False  # SMTPS (port 465); takes precedence over smtp_tls
