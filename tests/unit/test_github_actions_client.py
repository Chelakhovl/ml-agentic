"""Unit tests for GithubActionsClient and OrchestratorWorkflow CI/CD wiring."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agentic_mlops.contracts.github_actions import GithubActionsConfig
from agentic_mlops.integrations.github_actions_client import (
    FakeGithubActionsClient,
    GithubActionsClient,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cfg(**kwargs) -> GithubActionsConfig:
    defaults = dict(token="ghp_FAKE", owner="myorg", repo="ml-pipeline")
    defaults.update(kwargs)
    return GithubActionsConfig(**defaults)


def _fake_response(status: int = 204) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _payload(**kwargs) -> dict:
    base = {"workflow_id": "wf_001", "status": "completed", "step_outputs": {}}
    base.update(kwargs)
    return base


# ── FakeGithubActionsClient ───────────────────────────────────────────────────


class TestFakeGithubActionsClient:
    def test_records_calls(self):
        fake = FakeGithubActionsClient()
        fake.trigger("workflow_completed", {"workflow_id": "wf_1"})
        fake.trigger("workflow_completed", {"workflow_id": "wf_2"})
        assert len(fake.calls) == 2
        assert fake.calls[0] == ("workflow_completed", {"workflow_id": "wf_1"})

    def test_payload_is_copy(self):
        fake = FakeGithubActionsClient()
        p = {"workflow_id": "wf_1"}
        fake.trigger("workflow_completed", p)
        p["mutated"] = True
        assert "mutated" not in fake.calls[0][1]


# ── GithubActionsClient ───────────────────────────────────────────────────────


class TestGithubActionsClient:
    def test_dispatches_on_trigger_on_event(self):
        client = GithubActionsClient(_cfg())
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger("workflow_completed", _payload())
        assert mock.call_count == 1

    def test_skips_unlisted_events(self):
        client = GithubActionsClient(_cfg())
        with patch("urllib.request.urlopen") as mock:
            client.trigger("step_failed", _payload())
        mock.assert_not_called()

    def test_disabled_config_skips(self):
        client = GithubActionsClient(_cfg(enabled=False))
        with patch("urllib.request.urlopen") as mock:
            client.trigger("workflow_completed", _payload())
        mock.assert_not_called()

    def test_missing_token_logs_warning_no_raise(self, caplog, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS_TOKEN", raising=False)
        client = GithubActionsClient(_cfg(token=""))
        import logging

        with caplog.at_level(logging.WARNING):
            client.trigger("workflow_completed", _payload())
        assert "token" in caplog.text.lower()

    def test_request_url_and_method(self, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg())
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger("workflow_completed", _payload())
        req = mock.call_args[0][0]
        assert req.full_url == (
            "https://api.github.test/repos/myorg/ml-pipeline"
            "/actions/workflows/mlops-deploy.yml/dispatches"
        )
        assert req.get_method() == "POST"

    def test_request_headers_include_auth(self, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg(token="ghp_TEST123"))
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger("workflow_completed", _payload())
        req = mock.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer ghp_TEST123"
        assert req.get_header("Accept") == "application/vnd.github+json"

    def test_request_body_contains_inputs(self, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg())
        step_outputs = {
            "model_registry": {
                "model_name": "my-yolo",
                "version": 3,
                "registry_backend": "azure_ml",
            }
        }
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger(
                "workflow_completed",
                _payload(step_outputs=step_outputs),
            )
        body = json.loads(mock.call_args[0][0].data.decode())
        assert body["ref"] == "main"
        inputs = body["inputs"]
        assert inputs["mlops_workflow_id"] == "wf_001"
        assert inputs["model_name"] == "my-yolo"
        assert inputs["model_version"] == "3"
        assert inputs["registry_backend"] == "azure_ml"
        assert inputs["trigger_event"] == "workflow_completed"

    def test_extra_inputs_forwarded(self, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg(extra_inputs={"environment": "eu-prod"}))
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger("workflow_completed", _payload())
        body = json.loads(mock.call_args[0][0].data.decode())
        assert body["inputs"]["environment"] == "eu-prod"

    def test_non_204_status_logs_warning(self, caplog, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg())
        import logging

        with patch("urllib.request.urlopen", return_value=_fake_response(status=422)):
            with caplog.at_level(logging.WARNING):
                client.trigger("workflow_completed", _payload())
        assert "422" in caplog.text

    def test_network_exception_logs_warning_no_raise(self, caplog, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg())
        import logging

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            with caplog.at_level(logging.WARNING):
                client.trigger("workflow_completed", _payload())
        assert "connection refused" in caplog.text

    def test_custom_ref_used(self, monkeypatch):
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        client = GithubActionsClient(_cfg(ref="v2.0"))
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            client.trigger("workflow_completed", _payload())
        body = json.loads(mock.call_args[0][0].data.decode())
        assert body["ref"] == "v2.0"


# ── OrchestratorWorkflow wiring ───────────────────────────────────────────────


class TestOrchestratorGithubActionsWiring:
    def _run_dry(self, tmp_path, gh_client, workflow_id="wf_gh_test"):
        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)

        orch = OrchestratorWorkflow(github_actions_client=gh_client)
        inp = OrchestratorInput(
            workflow_id=workflow_id,
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
        )
        return orch.run(inp)

    def test_trigger_called_on_completed(self, tmp_path):
        fake = FakeGithubActionsClient()
        result = self._run_dry(tmp_path, fake)
        from agentic_mlops.contracts.orchestrator import OrchestratorStatus

        assert result.status == OrchestratorStatus.COMPLETED
        assert len(fake.calls) == 1
        event, payload = fake.calls[0]
        assert event == "workflow_completed"
        assert payload["workflow_id"] == "wf_gh_test"

    def test_trigger_not_called_on_failed(self, tmp_path):
        from unittest.mock import patch

        from agentic_mlops.contracts.orchestrator import OrchestratorInput
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow, _StepOutcome

        fake = FakeGithubActionsClient()
        orch = OrchestratorWorkflow(github_actions_client=fake)
        inp = OrchestratorInput(
            workflow_id="wf_gh_fail",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(tmp_path / "ds"),
            data_yaml_path=str(tmp_path / "ds" / "data.yaml"),
            dry_run=True,
        )
        failing = _StepOutcome(success=False, status_label="FAILED", errors=["forced"])
        with patch.object(orch, "_dispatch", return_value=failing):
            orch.run(inp)
        assert len(fake.calls) == 0

    def test_inp_github_actions_builds_client(self, tmp_path, monkeypatch):
        """When no client is injected, OrchestratorInput.github_actions builds one."""
        monkeypatch.setenv("GITHUB_API_BASE_URL", "https://api.github.test")
        from agentic_mlops.contracts.github_actions import GithubActionsConfig
        from agentic_mlops.contracts.orchestrator import OrchestratorInput, OrchestratorStatus
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="wf_gh_yaml",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
            github_actions=GithubActionsConfig(
                token="ghp_TEST",
                owner="myorg",
                repo="ml-pipeline",
            ),
        )
        with patch("urllib.request.urlopen", return_value=_fake_response()) as mock:
            result = orch.run(inp)
        assert result.status == OrchestratorStatus.COMPLETED
        assert mock.call_count == 1
        req = mock.call_args[0][0]
        assert "myorg/ml-pipeline" in req.full_url

    def test_no_github_client_no_errors(self, tmp_path):
        """Pipeline completes without errors when no GitHub Actions config."""
        from agentic_mlops.contracts.orchestrator import OrchestratorInput, OrchestratorStatus
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow
        from tests.conftest import make_valid_dataset

        ds = tmp_path / "ds"
        ds.mkdir()
        make_valid_dataset(ds)

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="wf_no_gh",
            steps=["dataset_validation"],
            runs_dir=str(tmp_path / "runs"),
            dataset_path=str(ds),
            data_yaml_path=str(ds / "data.yaml"),
            dry_run=True,
        )
        result = orch.run(inp)
        assert result.status == OrchestratorStatus.COMPLETED
