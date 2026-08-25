"""Tests for integrations/notification_client.py and OrchestratorWorkflow notification wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_mlops.contracts.notification import NotificationConfig
from agentic_mlops.contracts.orchestrator import OrchestratorInput, OrchestratorStatus
from agentic_mlops.integrations.notification_client import (
    FakeNotificationClient,
    NoOpNotificationClient,
    WebhookNotificationClient,
    _format_slack,
    _format_slack_approval,
    _format_teams,
    _format_teams_approval,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _config(**kwargs) -> NotificationConfig:
    defaults = dict(
        enabled=True,
        teams_webhook_url="https://outlook.office.com/webhook/TEAMS",
        slack_webhook_url="https://hooks.slack.com/services/SLACK",
    )
    defaults.update(kwargs)
    return NotificationConfig(**defaults)


def _fake_response(status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── NoOpNotificationClient ────────────────────────────────────────────────────


class TestNoOpNotificationClient:
    def test_send_never_raises(self):
        client = NoOpNotificationClient()
        client.send("step_failed", {"workflow_id": "wf1"})  # just must not raise


# ── FakeNotificationClient ────────────────────────────────────────────────────


class TestFakeNotificationClient:
    def test_records_events(self):
        client = FakeNotificationClient()
        client.send("step_started", {"workflow_id": "wf1", "step": "training"})
        client.send("workflow_completed", {"workflow_id": "wf1", "status": "completed"})
        assert len(client.events) == 2
        assert client.events[0] == ("step_started", {"workflow_id": "wf1", "step": "training"})
        assert client.events[1][0] == "workflow_completed"

    def test_payload_is_copy(self):
        client = FakeNotificationClient()
        payload = {"workflow_id": "wf1"}
        client.send("step_started", payload)
        payload["mutated"] = True
        assert "mutated" not in client.events[0][1]


# ── Message formatters ────────────────────────────────────────────────────────


class TestFormatTeams:
    def test_message_card_structure(self):
        card = _format_teams(
            "step_failed", {"workflow_id": "wf1", "step": "training", "error": "OOM"}
        )
        assert card["@type"] == "MessageCard"
        assert card["themeColor"] == "FF0000"  # red for step_failed
        assert card["sections"][0]["activityTitle"] == "MLOps Workflow: wf1"
        facts = {f["name"]: f["value"] for f in card["sections"][0]["facts"]}
        assert facts["Workflow ID"] == "wf1"
        assert facts["Step"] == "training"
        assert facts["Error"] == "OOM"

    def test_default_colour_for_unknown_event(self):
        card = _format_teams("unknown_event", {"workflow_id": "wf1"})
        assert card["themeColor"] == "808080"

    def test_workflow_completed_green(self):
        card = _format_teams("workflow_completed", {"workflow_id": "wf1"})
        assert card["themeColor"] == "00B050"


class TestFormatSlack:
    def test_text_present(self):
        body = _format_slack("step_failed", {"workflow_id": "wf1", "step": "training"})
        assert "text" in body
        assert "attachments" in body

    def test_colour_in_attachment(self):
        body = _format_slack("workflow_completed", {"workflow_id": "wf1"})
        assert body["attachments"][0]["color"] == "#00B050"

    def test_error_in_text(self):
        body = _format_slack("step_failed", {"workflow_id": "wf1", "error": "GPU OOM"})
        assert "GPU OOM" in body["text"]


# ── WebhookNotificationClient ─────────────────────────────────────────────────


class TestWebhookNotificationClient:
    def _make_client(self, **cfg_kwargs) -> WebhookNotificationClient:
        return WebhookNotificationClient(_config(**cfg_kwargs))

    def test_notify_on_filter_skips_unlisted_events(self):
        client = self._make_client(notify_on=["workflow_completed"])
        with patch("urllib.request.urlopen") as mock_urlopen:
            client.send("step_started", {"workflow_id": "wf1"})
            mock_urlopen.assert_not_called()

    def test_posts_to_teams(self):
        client = self._make_client(slack_webhook_url=None)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock_urlopen:
            client.send("step_failed", {"workflow_id": "wf1"})
            assert mock_urlopen.call_count == 1
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://outlook.office.com/webhook/TEAMS"
            body = json.loads(req.data.decode())
            assert body["@type"] == "MessageCard"

    def test_posts_to_slack(self):
        client = self._make_client(teams_webhook_url=None)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock_urlopen:
            client.send("step_failed", {"workflow_id": "wf1"})
            assert mock_urlopen.call_count == 1
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://hooks.slack.com/services/SLACK"
            body = json.loads(req.data.decode())
            assert "text" in body

    def test_posts_to_both_urls(self):
        client = self._make_client()
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock_urlopen:
            client.send("workflow_failed", {"workflow_id": "wf1"})
            assert mock_urlopen.call_count == 2

    def test_http_error_logs_warning_no_raise(self, caplog):
        client = self._make_client(slack_webhook_url=None)
        import logging

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            with caplog.at_level(logging.WARNING):
                client.send("step_failed", {"workflow_id": "wf1"})  # must not raise
        assert "connection refused" in caplog.text

    def test_non_200_status_logs_warning(self, caplog):
        client = self._make_client(slack_webhook_url=None)
        import logging

        with patch("urllib.request.urlopen", return_value=_fake_response(status=400)):
            with caplog.at_level(logging.WARNING):
                client.send("step_failed", {"workflow_id": "wf1"})
        assert "400" in caplog.text

    def test_disabled_config_not_filtered_here(self):
        # WebhookNotificationClient.send() only filters notify_on, not enabled flag —
        # the enabled check lives in OrchestratorWorkflow.run() when building _notifier.
        client = WebhookNotificationClient(
            NotificationConfig(enabled=False, teams_webhook_url="https://t.ms/x")
        )
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock_urlopen:
            client.send("step_failed", {"workflow_id": "wf1"})
            mock_urlopen.assert_called_once()


# ── OrchestratorWorkflow notification wiring ──────────────────────────────────


def _make_stub_step_result(success: bool = True, status: str = "COMPLETED"):
    from agentic_mlops.workflows.orchestrator import _StepOutcome

    return _StepOutcome(
        success=success,
        status_label=status,
        errors=[] if success else ["mock error"],
    )


def _build_orchestrator_with_fake_notifier(tmp_path: Path):
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    notifier = FakeNotificationClient()
    orch = OrchestratorWorkflow(notification_client=notifier)
    return orch, notifier


class TestOrchestratorNotifications:
    def _run_minimal(self, tmp_path: Path, steps=None, extra_inp=None):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        notifier = FakeNotificationClient()
        orch = OrchestratorWorkflow(notification_client=notifier)

        inp = OrchestratorInput(
            workflow_id="wf_notify_test",
            steps=steps or ["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
        )
        if extra_inp:
            inp = inp.model_copy(update=extra_inp)
        return orch, notifier, inp

    def test_completed_workflow_fires_workflow_completed(self, tmp_path: Path):
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)
        orch, notifier, inp = self._run_minimal(
            tmp_path,
            extra_inp={"dataset_path": str(ds), "data_yaml_path": str(ds / "data.yaml")},
        )
        orch.run(inp)
        event_names = [e for e, _ in notifier.events]
        assert "workflow_completed" in event_names

    def test_failed_step_fires_step_failed(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow, _StepOutcome

        notifier = FakeNotificationClient()
        orch = OrchestratorWorkflow(notification_client=notifier)

        # Patch _dispatch to simulate a failing step
        failing_outcome = _StepOutcome(
            success=False, status_label="FAILED", errors=["forced failure"]
        )

        inp = OrchestratorInput(
            workflow_id="wf_fail_test",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
        )
        with patch.object(orch, "_dispatch", return_value=failing_outcome):
            orch.run(inp)

        event_names = [e for e, _ in notifier.events]
        assert "step_failed" in event_names

    def test_exception_in_step_fires_step_failed(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        notifier = FakeNotificationClient()
        orch = OrchestratorWorkflow(notification_client=notifier)

        inp = OrchestratorInput(
            workflow_id="wf_exc_test",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
        )
        with patch.object(orch, "_dispatch", side_effect=RuntimeError("crash")):
            orch.run(inp)

        event_names = [e for e, _ in notifier.events]
        assert "step_failed" in event_names
        assert "workflow_failed" in event_names

    def test_no_notification_client_no_errors(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)
        orch = OrchestratorWorkflow()  # no notification_client, no inp.notifications
        inp = OrchestratorInput(
            workflow_id="wf_no_notify",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
        )
        result = orch.run(inp)
        # Should complete without raising, notifications just silently disabled
        assert result.status in (OrchestratorStatus.COMPLETED, OrchestratorStatus.FAILED)

    def test_inp_notifications_builds_webhook_client(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        orch = OrchestratorWorkflow()  # no injected client

        inp = OrchestratorInput(
            workflow_id="wf_yaml_notify",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
            notifications=NotificationConfig(
                enabled=True,
                teams_webhook_url="https://t.ms/hook",
            ),
        )
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock_urlopen:
            with patch.object(orch, "_dispatch", side_effect=RuntimeError("boom")):
                orch.run(inp)
            # step_failed + workflow_failed should have fired
            assert mock_urlopen.call_count >= 1

    def test_disabled_notifications_no_posts(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="wf_disabled_notify",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
            notifications=NotificationConfig(enabled=False, teams_webhook_url="https://t.ms/x"),
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            with patch.object(orch, "_dispatch", side_effect=RuntimeError("boom")):
                orch.run(inp)
            mock_urlopen.assert_not_called()

    def test_step_started_fires_when_in_notify_on(self, tmp_path: Path):
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow, _StepOutcome

        notifier = FakeNotificationClient()
        orch = OrchestratorWorkflow(notification_client=notifier)

        success_outcome = _StepOutcome(success=True, status_label="COMPLETED")
        inp = OrchestratorInput(
            workflow_id="wf_verbose",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
        )
        with patch.object(orch, "_dispatch", return_value=success_outcome):
            orch.run(inp)

        event_names = [e for e, _ in notifier.events]
        assert "step_started" in event_names
        assert "step_completed" in event_names


# ── Approval formatters ───────────────────────────────────────────────────────


_PENDING_PAYLOAD = {
    "workflow_id": "wf_approve",
    "status": "pending_approval",
    "message": "Awaiting decision",
    "pending_approval_id": "appr_wf_approve",
    "current_state": "MODEL_APPROVAL_REQUIRED",
}


class TestFormatTeamsApproval:
    def test_has_potential_action(self):
        card = _format_teams_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "http://localhost:8000",
        )
        assert card["@type"] == "MessageCard"
        assert "potentialAction" in card
        action = card["potentialAction"][0]
        assert action["@type"] == "OpenUri"
        assert "http://localhost:8000/approve/wf_approve" in action["targets"][0]["uri"]

    def test_gate_label_in_facts(self):
        card = _format_teams_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "https://mlops.example.com",
        )
        facts = {f["name"]: f["value"] for f in card["sections"][0]["facts"]}
        assert facts["Gate"] == "H5 — Model Approval"
        assert facts["Approval ID"] == "appr_wf_approve"

    def test_trailing_slash_stripped_from_base_url(self):
        card = _format_teams_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "https://mlops.example.com/",
        )
        uri = card["potentialAction"][0]["targets"][0]["uri"]
        assert not uri.startswith("https://mlops.example.com//")

    def test_h4_gate_label(self):
        payload = {**_PENDING_PAYLOAD, "current_state": "TRAINING_APPROVAL_REQUIRED"}
        card = _format_teams_approval("workflow_pending_approval", payload, "http://localhost:8000")
        facts = {f["name"]: f["value"] for f in card["sections"][0]["facts"]}
        assert facts["Gate"] == "H4 — Training Approval"


class TestFormatSlackApproval:
    def test_url_button_when_no_signing_secret(self):
        body = _format_slack_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "http://localhost:8000",
            slack_signing_secret=None,
        )
        # Should have a url-based button in the blocks attachments
        blocks = body["attachments"][0]["blocks"]
        actions = next(b for b in blocks if b["type"] == "actions")
        btn = actions["elements"][0]
        assert btn["type"] == "button"
        assert "url" in btn
        assert "http://localhost:8000/approve/wf_approve" in btn["url"]

    def test_interactive_buttons_when_signing_secret_set(self):
        body = _format_slack_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "http://localhost:8000",
            slack_signing_secret="secret123",
        )
        blocks = body["attachments"][0]["blocks"]
        actions = next(b for b in blocks if b["type"] == "actions")
        btns = actions["elements"]
        # Should have approve_model, reject_model, request_retraining
        action_ids = [b["action_id"] for b in btns]
        assert "approve_model" in action_ids
        assert "reject_model" in action_ids
        # Values encode workflow_id
        btn_values = [b["value"] for b in btns]
        assert all(v.startswith("wf_approve:") for v in btn_values)

    def test_h4_interactive_buttons(self):
        payload = {**_PENDING_PAYLOAD, "current_state": "TRAINING_APPROVAL_REQUIRED"}
        body = _format_slack_approval(
            "workflow_pending_approval", payload, "http://localhost:8000", "sec"
        )
        blocks = body["attachments"][0]["blocks"]
        actions = next(b for b in blocks if b["type"] == "actions")
        action_ids = [b["action_id"] for b in actions["elements"]]
        assert "approve_training" in action_ids
        assert "reject_training" in action_ids

    def test_text_present(self):
        body = _format_slack_approval(
            "workflow_pending_approval",
            _PENDING_PAYLOAD,
            "http://localhost:8000",
        )
        assert "wf_approve" in body["text"]


class TestWebhookApprovalDispatch:
    """WebhookNotificationClient routes approval events to the new formatters."""

    def test_approval_event_with_callback_uses_approval_formatter_teams(self):
        cfg = NotificationConfig(
            teams_webhook_url="https://t.ms/hook",
            slack_webhook_url=None,
            callback_base_url="http://localhost:8000",
        )
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.send("workflow_pending_approval", _PENDING_PAYLOAD)
        body = json.loads(mock.call_args[0][0].data.decode())
        # Should be an approval card with potentialAction, not a plain MessageCard
        assert "potentialAction" in body

    def test_approval_event_with_callback_uses_approval_formatter_slack(self):
        cfg = NotificationConfig(
            teams_webhook_url=None,
            slack_webhook_url="https://hooks.slack.com/TEST",
            callback_base_url="http://localhost:8000",
        )
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.send("workflow_pending_approval", _PENDING_PAYLOAD)
        body = json.loads(mock.call_args[0][0].data.decode())
        # Slack approval body uses blocks in attachment, not plain text attachment
        assert "blocks" in body["attachments"][0]

    def test_approval_event_without_callback_uses_plain_formatter(self):
        cfg = NotificationConfig(
            teams_webhook_url="https://t.ms/hook",
            slack_webhook_url=None,
            # No callback_base_url
        )
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.send("workflow_pending_approval", _PENDING_PAYLOAD)
        body = json.loads(mock.call_args[0][0].data.decode())
        # Falls back to plain MessageCard (no potentialAction)
        assert "potentialAction" not in body

    def test_non_approval_event_with_callback_uses_plain_formatter(self):
        cfg = NotificationConfig(
            teams_webhook_url="https://t.ms/hook",
            callback_base_url="http://localhost:8000",
        )
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.send("workflow_completed", {"workflow_id": "wf1", "status": "completed"})
        body = json.loads(mock.call_args[0][0].data.decode())
        assert "potentialAction" not in body


# ── Slack webhook endpoint ────────────────────────────────────────────────────


class TestSlackApprovalWebhook:
    """Tests for POST /webhooks/approve/slack in web/routes.py."""

    def _make_client(self, tmp_path, signing_secret: str = ""):
        """Build a TestClient for the FastAPI app pointing at tmp_path."""
        from fastapi.testclient import TestClient

        from agentic_mlops.web.app import create_app

        web_app = create_app(
            runs_dir=str(tmp_path / "runs"),
            registry_dir=str(tmp_path / "registry"),
            datasets_dir=str(tmp_path / "datasets"),
            slack_signing_secret=signing_secret,
        )
        return TestClient(web_app)

    def _make_slack_payload(self, workflow_id: str, action: str) -> dict:
        return {
            "type": "block_actions",
            "user": {"id": "U123", "name": "alice"},
            "actions": [
                {
                    "action_id": action,
                    "value": f"{workflow_id}:{action}",
                }
            ],
        }

    def _write_pending_state(self, tmp_path: Path, workflow_id: str, gate: str = "h5"):
        import json

        runs_dir = tmp_path / "runs" / workflow_id
        runs_dir.mkdir(parents=True)
        cs = "TRAINING_APPROVAL_REQUIRED" if gate == "h4" else "MODEL_APPROVAL_REQUIRED"
        state = {
            "workflow_id": workflow_id,
            "status": "pending_approval",
            "current_state": cs,
            "completed_steps": [],
            "step_status": {},
        }
        (runs_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (runs_dir / "input.json").write_text(
            json.dumps(
                {
                    "workflow_id": workflow_id,
                    "steps": ["approval"],
                    "runs_dir": str(tmp_path / "runs"),
                    "dataset_path": str(tmp_path / "ds"),
                    "data_yaml_path": str(tmp_path / "ds" / "data.yaml"),
                    "dry_run": True,
                    "resume": False,
                    "interactive_approval": False,
                    "interactive_training_approval": False,
                }
            ),
            encoding="utf-8",
        )

    def test_valid_approve_model_returns_200(self, tmp_path: Path):
        import urllib.parse

        self._write_pending_state(tmp_path, "wf_slack_test", gate="h5")
        client = self._make_client(tmp_path)
        payload = self._make_slack_payload("wf_slack_test", "approve_model")
        form_body = urllib.parse.urlencode({"payload": json.dumps(payload)})
        with patch("agentic_mlops.workflows.orchestrator.OrchestratorWorkflow.run"):
            resp = client.post(
                "/webhooks/approve/slack",
                content=form_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["action"] == "approve_model"

    def test_invalid_action_returns_400(self, tmp_path: Path):
        import urllib.parse

        self._write_pending_state(tmp_path, "wf_bad_action", gate="h5")
        client = self._make_client(tmp_path)
        payload = self._make_slack_payload("wf_bad_action", "not_a_real_action")
        form_body = urllib.parse.urlencode({"payload": json.dumps(payload)})
        resp = client.post(
            "/webhooks/approve/slack",
            content=form_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 400

    def test_missing_workflow_returns_404(self, tmp_path: Path):
        import urllib.parse

        client = self._make_client(tmp_path)
        payload = self._make_slack_payload("wf_nonexistent", "approve_model")
        form_body = urllib.parse.urlencode({"payload": json.dumps(payload)})
        resp = client.post(
            "/webhooks/approve/slack",
            content=form_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 404

    def test_valid_signature_accepted(self, tmp_path: Path):
        import hashlib
        import hmac
        import time
        import urllib.parse

        signing_secret = "test_secret_abc"
        self._write_pending_state(tmp_path, "wf_sig_ok", gate="h5")
        client = self._make_client(tmp_path, signing_secret=signing_secret)
        payload = self._make_slack_payload("wf_sig_ok", "approve_model")
        payload_json = json.dumps(payload)
        form_body = urllib.parse.urlencode({"payload": payload_json})
        ts = str(int(time.time()))
        sig_base = f"v0:{ts}:{form_body}"
        sig = (
            "v0=" + hmac.new(signing_secret.encode(), sig_base.encode(), hashlib.sha256).hexdigest()
        )
        with patch("agentic_mlops.workflows.orchestrator.OrchestratorWorkflow.run"):
            resp = client.post(
                "/webhooks/approve/slack",
                content=form_body,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "X-Slack-Request-Timestamp": ts,
                    "X-Slack-Signature": sig,
                },
            )
        assert resp.status_code == 200

    def test_invalid_signature_rejected(self, tmp_path: Path):
        import urllib.parse

        self._write_pending_state(tmp_path, "wf_sig_bad", gate="h5")
        client = self._make_client(tmp_path, signing_secret="real_secret")
        payload = self._make_slack_payload("wf_sig_bad", "approve_model")
        form_body = urllib.parse.urlencode({"payload": json.dumps(payload)})
        import time

        ts = str(int(time.time()))
        resp = client.post(
            "/webhooks/approve/slack",
            content=form_body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Slack-Request-Timestamp": ts,
                "X-Slack-Signature": "v0=bad_signature",
            },
        )
        assert resp.status_code == 403


# ── SmtpEmailNotificationClient ───────────────────────────────────────────────


def _decode_email(raw: str) -> tuple[str, str]:
    """Return (decoded_subject, decoded_body) from a raw MIME message string."""
    import email as _email
    import email.header as _hdr

    msg = _email.message_from_string(raw)
    subject = str(_hdr.make_header(_hdr.decode_header(msg["Subject"] or "")))
    body = (msg.get_payload(decode=True) or b"").decode("utf-8")
    return subject, body


def _email_cfg(**kwargs) -> NotificationConfig:
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_port=587,
        email_to=["ops@example.com"],
        email_from="mlops@example.com",
        smtp_tls=True,
        smtp_ssl=False,
    )
    defaults.update(kwargs)
    return NotificationConfig(**defaults)


class TestSmtpEmailNotificationClient:
    def _smtp_mock(self):
        """Return a context-manager-compatible SMTP mock."""
        from unittest.mock import MagicMock

        smtp = MagicMock()
        smtp.__enter__ = lambda s: s
        smtp.__exit__ = MagicMock(return_value=False)
        return smtp

    def test_sends_on_listed_event(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp) as mock_cls:
            SmtpEmailNotificationClient(_email_cfg()).send(
                "workflow_completed", {"workflow_id": "wf1", "status": "completed"}
            )
        mock_cls.assert_called_once()
        smtp.sendmail.assert_called_once()

    def test_skips_unlisted_event(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        with patch("smtplib.SMTP") as mock_cls:
            SmtpEmailNotificationClient(_email_cfg(notify_on=["workflow_failed"])).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        mock_cls.assert_not_called()

    def test_skips_when_no_smtp_host(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        with patch("smtplib.SMTP") as mock_cls:
            SmtpEmailNotificationClient(_email_cfg(smtp_host="")).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        mock_cls.assert_not_called()

    def test_skips_when_no_email_to(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        with patch("smtplib.SMTP") as mock_cls:
            SmtpEmailNotificationClient(_email_cfg(email_to=[])).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        mock_cls.assert_not_called()

    def test_subject_contains_event_and_workflow_id(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg()).send(
                "step_failed", {"workflow_id": "wf_sub", "step": "training"}
            )
        subject, body = _decode_email(smtp.sendmail.call_args[0][2])
        assert "Step Failed" in subject
        assert "wf_sub" in subject
        assert "training" in subject

    def test_body_contains_error(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg()).send(
                "step_failed",
                {"workflow_id": "wf_err", "error": "GPU out of memory"},
            )
        _, body = _decode_email(smtp.sendmail.call_args[0][2])
        assert "GPU out of memory" in body

    def test_starttls_called_when_tls_true(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg(smtp_tls=True, smtp_ssl=False)).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        smtp.starttls.assert_called_once()

    def test_starttls_not_called_when_ssl_true(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP_SSL", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg(smtp_ssl=True, smtp_tls=True)).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        smtp.starttls.assert_not_called()

    def test_ssl_uses_smtp_ssl_class(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with (
            patch("smtplib.SMTP_SSL", return_value=smtp) as mock_ssl,
            patch("smtplib.SMTP") as mock_plain,
        ):
            SmtpEmailNotificationClient(_email_cfg(smtp_ssl=True, smtp_port=465)).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        mock_ssl.assert_called_once_with("smtp.example.com", 465, timeout=15)
        mock_plain.assert_not_called()

    def test_login_called_when_credentials_set(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(
                _email_cfg(smtp_user="user@ex.com", smtp_password="s3cr3t")
            ).send("workflow_completed", {"workflow_id": "wf1"})
        smtp.login.assert_called_once_with("user@ex.com", "s3cr3t")

    def test_login_not_called_without_credentials(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg()).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        smtp.login.assert_not_called()

    def test_multi_recipient_sendmail(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        recipients = ["alice@ex.com", "bob@ex.com"]
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg(email_to=recipients)).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        to_arg = smtp.sendmail.call_args[0][1]
        assert to_arg == recipients

    def test_custom_subject_prefix(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg(email_subject_prefix="[FACTORY-ML]")).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        subject, _ = _decode_email(smtp.sendmail.call_args[0][2])
        assert "[FACTORY-ML]" in subject

    def test_auto_port_tls(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp) as mock_cls:
            SmtpEmailNotificationClient(
                _email_cfg(smtp_port=0, smtp_tls=True, smtp_ssl=False)
            ).send("workflow_completed", {"workflow_id": "wf1"})
        _, port, *_ = mock_cls.call_args[0]
        assert port == 587

    def test_auto_port_ssl(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP_SSL", return_value=smtp) as mock_cls:
            SmtpEmailNotificationClient(_email_cfg(smtp_port=0, smtp_ssl=True)).send(
                "workflow_completed", {"workflow_id": "wf1"}
            )
        _, port, *_ = mock_cls.call_args[0]
        assert port == 465

    def test_smtp_exception_logs_warning_no_raise(self, caplog):
        import logging
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        with patch("smtplib.SMTP", side_effect=Exception("connection refused")):
            with caplog.at_level(logging.WARNING):
                SmtpEmailNotificationClient(_email_cfg()).send(
                    "workflow_completed", {"workflow_id": "wf1"}
                )
        assert "connection refused" in caplog.text

    def test_pending_approval_id_in_body(self):
        from unittest.mock import patch

        from agentic_mlops.integrations.notification_client import SmtpEmailNotificationClient

        smtp = self._smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp):
            SmtpEmailNotificationClient(_email_cfg()).send(
                "workflow_pending_approval",
                {"workflow_id": "wf_pend", "pending_approval_id": "appr_wf_pend"},
            )
        _, body = _decode_email(smtp.sendmail.call_args[0][2])
        assert "appr_wf_pend" in body


# ── CompositeNotificationClient ───────────────────────────────────────────────


class TestCompositeNotificationClient:
    def test_delegates_to_all_clients(self):
        from agentic_mlops.integrations.notification_client import (
            CompositeNotificationClient,
            FakeNotificationClient,
        )

        a, b = FakeNotificationClient(), FakeNotificationClient()
        comp = CompositeNotificationClient([a, b])
        comp.send("workflow_completed", {"workflow_id": "wf1"})
        assert len(a.events) == 1
        assert len(b.events) == 1

    def test_one_failing_child_does_not_block_others(self, caplog):
        import logging
        from unittest.mock import MagicMock

        from agentic_mlops.integrations.notification_client import (
            CompositeNotificationClient,
            FakeNotificationClient,
        )

        bad = MagicMock()
        bad.send.side_effect = RuntimeError("exploded")
        good = FakeNotificationClient()
        comp = CompositeNotificationClient([bad, good])
        with caplog.at_level(logging.WARNING):
            comp.send("workflow_completed", {"workflow_id": "wf1"})
        assert len(good.events) == 1
        assert "exploded" in caplog.text

    def test_empty_composite_is_silent(self):
        from agentic_mlops.integrations.notification_client import CompositeNotificationClient

        CompositeNotificationClient([]).send("workflow_completed", {})  # must not raise


# ── Orchestrator composite wiring ─────────────────────────────────────────────


class TestOrchestratorEmailWiring:
    def _run_with_smtp_config(self, tmp_path, smtp_mock):
        from unittest.mock import patch

        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="wf_email_test",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
            notifications=NotificationConfig(
                smtp_host="smtp.example.com",
                email_to=["ops@example.com"],
                smtp_tls=False,
                smtp_ssl=False,
            ),
        )
        with patch("smtplib.SMTP", return_value=smtp_mock):
            result = orch.run(inp)
        return result

    def test_email_sent_on_workflow_completed(self, tmp_path):
        from unittest.mock import MagicMock

        smtp = MagicMock()
        smtp.__enter__ = lambda s: s
        smtp.__exit__ = MagicMock(return_value=False)
        result = self._run_with_smtp_config(tmp_path, smtp)
        from agentic_mlops.contracts.orchestrator import OrchestratorStatus

        assert result.status == OrchestratorStatus.COMPLETED
        smtp.sendmail.assert_called()

    def test_both_webhook_and_email_fire(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from agentic_mlops.contracts.orchestrator import OrchestratorInput, OrchestratorStatus
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds2"
        ds.mkdir()
        make_valid_dataset(ds)

        smtp = MagicMock()
        smtp.__enter__ = lambda s: s
        smtp.__exit__ = MagicMock(return_value=False)

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="wf_combo",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs2"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
            notifications=NotificationConfig(
                teams_webhook_url="https://t.ms/hook",
                smtp_host="smtp.example.com",
                email_to=["ops@example.com"],
                smtp_tls=False,
                smtp_ssl=False,
            ),
        )
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=fake_resp) as mock_http,
            patch("smtplib.SMTP", return_value=smtp),
        ):
            result = orch.run(inp)

        assert result.status == OrchestratorStatus.COMPLETED
        mock_http.assert_called()
        smtp.sendmail.assert_called()
