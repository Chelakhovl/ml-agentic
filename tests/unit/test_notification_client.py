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
    _format_teams,
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
