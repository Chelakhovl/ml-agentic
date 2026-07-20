"""Webhook notification clients for the Orchestrator.

Three implementations, following the same injection pattern as
AzureMLTrainingRunner / ArtifactStore:

  NoOpNotificationClient    — safe default; does nothing
  WebhookNotificationClient — POSTs to Teams (MessageCard) and/or Slack
                               webhooks; stdlib-only (urllib.request)
  FakeNotificationClient    — in-memory test double; records all calls

All implementations are best-effort: ``send()`` swallows every exception
and logs a warning so a broken webhook can never abort the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_mlops.contracts.notification import NotificationConfig

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────

_HEX_COLOUR: dict[str, str] = {
    "step_started": "0076D7",         # blue
    "step_completed": "00B050",       # green
    "step_failed": "FF0000",          # red
    "workflow_completed": "00B050",   # green
    "workflow_failed": "FF0000",      # red
    "workflow_blocked": "CC4400",     # dark orange
    "workflow_pending_approval": "FFA500",  # orange
}
_COLOUR_DEFAULT = "808080"  # grey

# Slack uses #rrggbb
_SLACK_COLOUR: dict[str, str] = {k: f"#{v}" for k, v in _HEX_COLOUR.items()}
_SLACK_COLOUR_DEFAULT = "#808080"


# ── Base ──────────────────────────────────────────────────────────────────────


class NotificationClientBase:
    """Base class — all notification backends implement this interface."""

    def send(self, event_name: str, payload: dict) -> None:
        """Fire a notification for *event_name* with *payload* context dict.

        Implementations MUST NOT raise — they should log warnings on failure
        and return silently so the calling pipeline is never blocked.
        """
        raise NotImplementedError


# ── No-op ─────────────────────────────────────────────────────────────────────


class NoOpNotificationClient(NotificationClientBase):
    """Safe default — does nothing, requires no configuration."""

    def send(self, event_name: str, payload: dict) -> None:
        pass


# ── Webhook ───────────────────────────────────────────────────────────────────


class WebhookNotificationClient(NotificationClientBase):
    """Sends Teams (MessageCard) and/or Slack notifications via incoming webhooks.

    Requires no extra packages — uses ``urllib.request`` from the standard
    library.  Both URLs are optional; configure either or both.

    The ``notify_on`` list in ``NotificationConfig`` controls which event names
    trigger an actual POST.  Events not in the list are silently ignored.
    """

    def __init__(self, config: NotificationConfig) -> None:
        self._config = config

    def send(self, event_name: str, payload: dict) -> None:
        if event_name not in self._config.notify_on:
            return
        if self._config.teams_webhook_url:
            self._post(
                self._config.teams_webhook_url,
                _format_teams(event_name, payload),
            )
        if self._config.slack_webhook_url:
            self._post(
                self._config.slack_webhook_url,
                _format_slack(event_name, payload),
            )

    def _post(self, url: str, body: dict) -> None:
        """HTTP POST *body* as JSON to *url*. Logs on failure, never raises."""
        import urllib.request  # noqa: PLC0415

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                status = resp.status
                if status >= 300:
                    logger.warning("Webhook POST returned HTTP %s for %s", status, url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook POST failed (%s): %s", url, exc)


# ── Fake (test double) ────────────────────────────────────────────────────────


class FakeNotificationClient(NotificationClientBase):
    """In-memory test double. Records every ``send()`` call in ``self.events``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def send(self, event_name: str, payload: dict) -> None:
        self.events.append((event_name, dict(payload)))


# ── Message formatters ────────────────────────────────────────────────────────


def _format_teams(event_name: str, payload: dict) -> dict:
    """Build an O365 Incoming Webhook MessageCard payload."""
    workflow_id = payload.get("workflow_id", "unknown")
    step = payload.get("step", "")
    status = payload.get("status", "")
    message_text = payload.get("message", "")
    colour = _HEX_COLOUR.get(event_name, _COLOUR_DEFAULT)

    subtitle = event_name.replace("_", " ").title()
    if step:
        subtitle += f" — step: {step}"

    facts: list[dict] = [
        {"name": "Workflow ID", "value": workflow_id},
        {"name": "Event", "value": event_name},
    ]
    if step:
        facts.append({"name": "Step", "value": step})
    if status:
        facts.append({"name": "Status", "value": status})
    if message_text:
        facts.append({"name": "Message", "value": message_text})
    if payload.get("error"):
        facts.append({"name": "Error", "value": str(payload["error"])})

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": colour,
        "summary": subtitle,
        "sections": [
            {
                "activityTitle": f"MLOps Workflow: {workflow_id}",
                "activitySubtitle": subtitle,
                "facts": facts,
            }
        ],
    }


def _format_slack(event_name: str, payload: dict) -> dict:
    """Build a Slack Incoming Webhook payload with a colour attachment."""
    workflow_id = payload.get("workflow_id", "unknown")
    step = payload.get("step", "")
    status = payload.get("status", "")
    colour = _SLACK_COLOUR.get(event_name, _SLACK_COLOUR_DEFAULT)

    text = f"*{workflow_id}* | {event_name.replace('_', ' ')}"
    if step:
        text += f" — `{step}`"
    if status:
        text += f" ({status})"
    if payload.get("error"):
        text += f"\n> {payload['error']}"

    return {
        "text": text,
        "attachments": [{"color": colour, "text": text}],
    }
