"""Notification clients for the Orchestrator.

Five implementations, following the same injection pattern as
AzureMLTrainingRunner / ArtifactStore:

  NoOpNotificationClient    — safe default; does nothing
  WebhookNotificationClient — POSTs to Teams (MessageCard) and/or Slack
                               webhooks; stdlib-only (urllib.request)
  SmtpEmailNotificationClient — sends plain-text emails via SMTP;
                               stdlib-only (smtplib / email.mime)
  CompositeNotificationClient — delegates to a list of other clients
  FakeNotificationClient    — in-memory test double; records all calls

All implementations are best-effort: ``send()`` swallows every exception
and logs a warning so a broken channel can never abort the pipeline.
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
    "step_started": "0076D7",  # blue
    "step_completed": "00B050",  # green
    "step_failed": "FF0000",  # red
    "workflow_completed": "00B050",  # green
    "workflow_failed": "FF0000",  # red
    "workflow_blocked": "CC4400",  # dark orange
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

    When ``config.callback_base_url`` is set and ``event_name`` is
    ``workflow_pending_approval``, the Teams message includes an OpenUri button
    that opens the dashboard approval page in the browser, and the Slack message
    includes action buttons (interactive POST-back when ``slack_signing_secret``
    is also set; browser-link otherwise).
    """

    def __init__(self, config: NotificationConfig) -> None:
        self._config = config

    def send(self, event_name: str, payload: dict) -> None:
        if event_name not in self._config.notify_on:
            return
        is_approval = event_name == "workflow_pending_approval"
        cb = self._config.callback_base_url
        if self._config.teams_webhook_url:
            body = (
                _format_teams_approval(event_name, payload, cb, self._config.teams_signing_secret)
                if is_approval and cb
                else _format_teams(event_name, payload)
            )
            self._post(self._config.teams_webhook_url, body)
        if self._config.slack_webhook_url:
            body = (
                _format_slack_approval(event_name, payload, cb, self._config.slack_signing_secret)
                if is_approval and cb
                else _format_slack(event_name, payload)
            )
            self._post(self._config.slack_webhook_url, body)

    @staticmethod
    def _redact_url(url: str) -> str:
        """Return a redacted form of a webhook URL suitable for logging.

        Teams URLs look like ``https://…/webhook/<guid>/IncomingWebhook/<token1>/<token2>``
        and Slack URLs look like ``https://hooks.slack.com/services/<T>/<B>/<secret>``.
        Both embed access tokens in the path, so the full URL must never appear in
        logs. Only the scheme + host are kept, plus a ``/…`` suffix.
        """
        try:
            from urllib.parse import urlparse  # noqa: PLC0415

            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}/…"
        except Exception:  # noqa: BLE001
            return "<webhook>"

    def _post(self, url: str, body: dict) -> None:
        """HTTP POST *body* as JSON to *url* with exponential-backoff retries.

        Retries on network errors and 5xx responses. 4xx are not retried (client
        errors won't recover on retry). Never raises — logs a warning on final failure.
        """
        import time  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        max_retries = self._config.webhook_max_retries
        delay = self._config.webhook_retry_delay
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        safe_url = self._redact_url(url)

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                    status = resp.status
                    if status < 300:
                        return
                    # 5xx: transient — retry; 4xx: permanent — give up now
                    if status < 500:
                        logger.warning(
                            "Webhook POST returned HTTP %s for %s (not retrying)",
                            status,
                            safe_url,
                        )
                        return
                    err_msg = f"HTTP {status}"
            except urllib.error.URLError as exc:
                err_msg = str(exc)
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)

            if attempt < max_retries:
                logger.warning(
                    "Webhook POST failed (%s, attempt %d/%d): %s — retrying in %.1fs",
                    safe_url, attempt + 1, max_retries + 1, err_msg, delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.warning("Webhook POST failed (%s): %s", safe_url, err_msg)


# ── SMTP email ────────────────────────────────────────────────────────────────


class SmtpEmailNotificationClient(NotificationClientBase):
    """Sends plain-text email notifications via SMTP.

    Stdlib-only (``smtplib``, ``email.mime``) — no extra packages required.
    Skips silently when ``smtp_host`` or ``email_to`` is not set.
    Supports STARTTLS (``smtp_tls=True``, default) and SMTPS (``smtp_ssl=True``).
    Best-effort: ``send()`` never raises.
    """

    def __init__(self, config: NotificationConfig) -> None:
        self._config = config

    def send(self, event_name: str, payload: dict) -> None:
        cfg = self._config
        if event_name not in cfg.notify_on:
            return
        if not cfg.smtp_host or not cfg.email_to:
            return
        try:
            self._send_email(event_name, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMTP email notification failed: %s", exc)

    def _send_email(self, event_name: str, payload: dict) -> None:
        import smtplib  # noqa: PLC0415
        from email.mime.text import MIMEText  # noqa: PLC0415

        cfg = self._config
        workflow_id = payload.get("workflow_id", "unknown")
        status = payload.get("status", "")
        step = payload.get("step", "")
        message = payload.get("message", "")

        prefix = cfg.email_subject_prefix or "[MLOps]"
        event_label = event_name.replace("_", " ").title()
        subject = f"{prefix} {event_label} — {workflow_id}"
        if step:
            subject += f" ({step})"

        lines = [
            f"Workflow ID: {workflow_id}",
            f"Event:       {event_name}",
        ]
        if step:
            lines.append(f"Step:        {step}")
        if status:
            lines.append(f"Status:      {status}")
        if message:
            lines.append(f"Message:     {message}")
        if payload.get("error"):
            lines.append(f"Error:       {payload['error']}")
        if payload.get("pending_approval_id"):
            lines.append(f"Approval ID: {payload['pending_approval_id']}")

        msg = MIMEText("\n".join(lines), "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = cfg.email_from
        msg["To"] = ", ".join(cfg.email_to)

        port = cfg.smtp_port or (465 if cfg.smtp_ssl else 587 if cfg.smtp_tls else 25)
        smtp_cls = smtplib.SMTP_SSL if cfg.smtp_ssl else smtplib.SMTP

        with smtp_cls(cfg.smtp_host, port, timeout=15) as smtp:
            if cfg.smtp_tls and not cfg.smtp_ssl:
                smtp.starttls()
            if cfg.smtp_user and cfg.smtp_password:
                smtp.login(cfg.smtp_user, cfg.smtp_password)
            smtp.sendmail(cfg.email_from, cfg.email_to, msg.as_string())


# ── Composite ─────────────────────────────────────────────────────────────────


class CompositeNotificationClient(NotificationClientBase):
    """Delegates ``send()`` to every client in *clients* in order.

    Each child handles its own exceptions; one failing channel never prevents
    the others from firing.
    """

    def __init__(self, clients: list[NotificationClientBase]) -> None:
        self._clients = list(clients)

    def send(self, event_name: str, payload: dict) -> None:
        for client in self._clients:
            try:
                client.send(event_name, payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CompositeNotificationClient child failed: %s", exc)


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


# ── Approval gate action sets ─────────────────────────────────────────────────

# Actions per current_state; each entry is (action_id, button_label, style).
# style: "primary" (green), "danger" (red), or "" (default grey).
_APPROVAL_ACTIONS: dict[str, list[tuple[str, str, str]]] = {
    "TRAINING_APPROVAL_REQUIRED": [
        ("approve_training", "Approve Training", "primary"),
        ("reject_training", "Reject Training", "danger"),
    ],
    "MODEL_APPROVAL_REQUIRED": [
        ("approve_model", "Approve Model", "primary"),
        ("reject_model", "Reject Model", "danger"),
        ("request_retraining", "Request Retraining", ""),
    ],
}
_APPROVAL_GATE_LABEL: dict[str, str] = {
    "TRAINING_APPROVAL_REQUIRED": "H4 — Training Approval",
    "MODEL_APPROVAL_REQUIRED": "H5 — Model Approval",
}


def _format_teams_approval(
    event_name: str,
    payload: dict,
    callback_base_url: str,
    teams_signing_secret: str | None = None,
) -> dict:
    """Build a Teams MessageCard with approval action buttons.

    When *teams_signing_secret* is ``None``: includes a single ``OpenUri``
    button that opens ``{callback_base_url}/approve/{workflow_id}`` in the
    browser — works with any Teams incoming webhook, no extra configuration.

    When *teams_signing_secret* is set: adds ``HttpPOST`` action buttons
    (one per approval action) that POST the decision directly to
    ``{callback_base_url}/webhooks/approve/teams``.  The dashboard verifies
    the ``Authorization: HMAC <base64-hmac>`` header Teams attaches using the
    same key.  Requires the Teams connector's security token to be configured.
    """
    workflow_id = payload.get("workflow_id", "unknown")
    current_state = payload.get("current_state", "")
    gate_label = _APPROVAL_GATE_LABEL.get(current_state, "Approval Required")
    message_text = payload.get("message", "")
    colour = _HEX_COLOUR.get(event_name, _COLOUR_DEFAULT)

    facts: list[dict] = [
        {"name": "Workflow ID", "value": workflow_id},
        {"name": "Gate", "value": gate_label},
    ]
    if payload.get("pending_approval_id"):
        facts.append({"name": "Approval ID", "value": payload["pending_approval_id"]})
    if message_text:
        facts.append({"name": "Message", "value": message_text})

    approval_url = f"{callback_base_url.rstrip('/')}/approve/{workflow_id}"
    teams_webhook_url = f"{callback_base_url.rstrip('/')}/webhooks/approve/teams"

    if teams_signing_secret:
        # HttpPOST action buttons — Teams POSTs to our webhook handler.
        # Each action sends {"workflow_id": ..., "action": ...} as the body.
        actions_for_state = _APPROVAL_ACTIONS.get(current_state, [])
        http_actions: list[dict] = [
            {
                "@type": "HttpPOST",
                "name": label,
                "target": teams_webhook_url,
                "bodyContentType": "application/json",
                "body": json.dumps({"workflow_id": workflow_id, "action": action_id}),
            }
            for action_id, label, _style in actions_for_state
        ]
        if not http_actions:
            # Unknown gate — fall back to browser link
            http_actions = []

        potential_action: list[dict] = [
            {
                "@type": "OpenUri",
                "name": "Open Dashboard",
                "targets": [{"os": "default", "uri": approval_url}],
            }
        ]
        if http_actions:
            potential_action.append(
                {
                    "@type": "ActionCard",
                    "name": "Take Action",
                    "actions": http_actions,
                }
            )
    else:
        potential_action = [
            {
                "@type": "OpenUri",
                "name": "Open Approval Page",
                "targets": [{"os": "default", "uri": approval_url}],
            }
        ]

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": colour,
        "summary": f"Approval Required — {workflow_id}",
        "sections": [
            {
                "activityTitle": f"MLOps Workflow: {workflow_id}",
                "activitySubtitle": gate_label,
                "facts": facts,
            }
        ],
        "potentialAction": potential_action,
    }


def _format_slack_approval(
    event_name: str,
    payload: dict,
    callback_base_url: str,
    slack_signing_secret: str | None = None,
) -> dict:
    """Build a Slack Block Kit payload with approval action buttons.

    When *slack_signing_secret* is ``None``: buttons use the ``url`` field to
    open ``{callback_base_url}/approve/{workflow_id}`` in the browser — works
    with any Slack incoming webhook, no app configuration needed.

    When *slack_signing_secret* is set: buttons use ``action_id``/``value``
    fields (interactive POST-back) so Slack can send the decision to
    ``{callback_base_url}/webhooks/approve/slack``.  Requires the Slack app's
    "Interactivity Request URL" configured to that address.
    """
    workflow_id = payload.get("workflow_id", "unknown")
    current_state = payload.get("current_state", "")
    gate_label = _APPROVAL_GATE_LABEL.get(current_state, "Approval Required")
    message_text = payload.get("message", "")
    colour = _SLACK_COLOUR.get(event_name, _SLACK_COLOUR_DEFAULT)

    summary = f":large_orange_circle: *Approval Required* — `{workflow_id}`"
    detail_lines = [f"*Gate:* {gate_label}"]
    if payload.get("pending_approval_id"):
        detail_lines.append(f"*Approval ID:* `{payload['pending_approval_id']}`")
    if message_text:
        detail_lines.append(f"_{message_text}_")
    detail = "\n".join(detail_lines)

    blocks: list[dict] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{summary}\n{detail}"},
        }
    ]

    approval_url = f"{callback_base_url.rstrip('/')}/approve/{workflow_id}"
    actions_for_state = _APPROVAL_ACTIONS.get(current_state, [])

    if slack_signing_secret:
        # Interactive POST-back buttons — Slack sends payload to our webhook endpoint.
        elements = []
        for action_id, label, style in actions_for_state:
            btn: dict = {
                "type": "button",
                "action_id": action_id,
                "value": f"{workflow_id}:{action_id}",
                "text": {"type": "plain_text", "text": label},
            }
            if style:
                btn["style"] = style
            elements.append(btn)
        if not elements:
            # Fallback when current_state is unknown — generic link button
            elements = [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Open Approval Page"},
                    "url": approval_url,
                    "style": "primary",
                }
            ]
    else:
        # Browser-link buttons — works with plain incoming webhook, no app config needed.
        elements = [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Open Approval Page"},
                "url": approval_url,
                "style": "primary",
            }
        ]

    blocks.append({"type": "actions", "elements": elements})

    return {
        "text": f"Approval Required: {workflow_id} — {gate_label}",
        "attachments": [{"color": colour, "blocks": blocks}],
    }
