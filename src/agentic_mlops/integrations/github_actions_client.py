"""GitHub Actions CI/CD trigger integration.

Sends a ``workflow_dispatch`` event to a GitHub Actions workflow via the
GitHub REST API when the Orchestrator pipeline completes.  Stdlib-only
(``urllib.request``) — no extra packages required.  Best-effort: every
exception is swallowed and logged so a broken token can never abort the
pipeline.

Usage (injected into OrchestratorWorkflow):

    from agentic_mlops.contracts.github_actions import GithubActionsConfig
    from agentic_mlops.integrations.github_actions_client import GithubActionsClient

    cfg = GithubActionsConfig(
        token="ghp_...",
        owner="myorg",
        repo="ml-pipeline",
        workflow_file="mlops-deploy.yml",
        ref="main",
    )
    client = GithubActionsClient(cfg)
    # Called automatically by OrchestratorWorkflow._finish() on workflow_completed.
    client.trigger("workflow_completed", {"workflow_id": "wf_001", ...})
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_mlops.contracts.github_actions import GithubActionsConfig

logger = logging.getLogger(__name__)

# GitHub REST API base URL; override in tests via GITHUB_API_BASE_URL env var.
_GITHUB_API_BASE = "https://api.github.com"


class GithubActionsClient:
    """Triggers a GitHub Actions ``workflow_dispatch`` event.

    The dispatched workflow receives the following inputs (all strings, as
    required by the GitHub Actions API):

    ``mlops_workflow_id``
        The Orchestrator run ID (same as ``workflow_id`` in state.json).
    ``mlops_status``
        Terminal pipeline status (e.g. ``"completed"``).
    ``model_name``
        Model name from the ``model_registry`` step, if that step ran.
    ``model_version``
        Registered version number (as a string), if available.
    ``registry_backend``
        Registry backend used (``"local"``, ``"mlflow"``, ``"azure_ml"``).
    ``trigger_event``
        The Orchestrator event name that fired the dispatch.

    Any ``extra_inputs`` from ``GithubActionsConfig`` are merged on top.
    """

    def __init__(self, config: GithubActionsConfig) -> None:
        self._config = config

    def trigger(self, event_name: str, payload: dict) -> None:
        """Fire a workflow_dispatch if *event_name* is in ``trigger_on``.

        Never raises — logs a warning on any failure.
        """
        if not self._config.enabled:
            return
        if event_name not in self._config.trigger_on:
            return
        token = self._config.token or os.environ.get("GITHUB_ACTIONS_TOKEN", "")
        if not token:
            logger.warning(
                "GithubActionsClient: no token configured — "
                "set GithubActionsConfig.token or GITHUB_ACTIONS_TOKEN env var"
            )
            return
        try:
            self._dispatch(event_name, payload, token)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GitHub Actions workflow_dispatch failed: %s", exc)

    # ------------------------------------------------------------------

    def _dispatch(self, event_name: str, payload: dict, token: str) -> None:
        import urllib.request  # noqa: PLC0415

        cfg = self._config
        workflow_id = payload.get("workflow_id", "unknown")
        status = payload.get("status", "")

        # Extract model info persisted in step_outputs by the orchestrator.
        step_outputs: dict = payload.get("step_outputs", {})
        registry_out: dict = step_outputs.get("model_registry", {})
        model_name = registry_out.get("model_name") or payload.get("model_name") or ""
        model_version = str(registry_out.get("version") or "")
        registry_backend = (
            registry_out.get("registry_backend") or payload.get("registry_backend") or "local"
        )

        inputs: dict[str, str] = {
            "mlops_workflow_id": workflow_id,
            "mlops_status": status,
            "model_name": model_name,
            "model_version": model_version,
            "registry_backend": registry_backend,
            "trigger_event": event_name,
        }
        inputs.update(cfg.extra_inputs)

        api_base = os.environ.get("GITHUB_API_BASE_URL", _GITHUB_API_BASE).rstrip("/")
        url = (
            f"{api_base}/repos/{cfg.owner}/{cfg.repo}"
            f"/actions/workflows/{cfg.workflow_file}/dispatches"
        )
        body = json.dumps({"ref": cfg.ref, "inputs": inputs}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            status_code = resp.status
            if status_code not in (200, 204):
                logger.warning(
                    "GitHub Actions dispatch returned HTTP %s for %s/%s",
                    status_code,
                    cfg.owner,
                    cfg.repo,
                )
            else:
                logger.info(
                    "GitHub Actions workflow dispatched: %s/%s @ %s (workflow_id=%s)",
                    cfg.owner,
                    cfg.repo,
                    cfg.ref,
                    workflow_id,
                )


class FakeGithubActionsClient:
    """In-memory test double. Records every trigger() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def trigger(self, event_name: str, payload: dict) -> None:
        self.calls.append((event_name, dict(payload)))
