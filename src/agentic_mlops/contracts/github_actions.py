"""Pydantic contract for GitHub Actions CI/CD trigger configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GithubActionsConfig(BaseModel):
    """Configuration for triggering a GitHub Actions workflow dispatch.

    When attached to ``OrchestratorInput.github_actions`` and the pipeline
    completes successfully, ``OrchestratorWorkflow`` calls the GitHub REST API
    to dispatch the configured workflow with the pipeline run details as inputs.

    Token requirements: the PAT (or Actions-generated ``GITHUB_TOKEN``) needs
    the ``actions:write`` permission on the target repo.  Store the token in
    ``GITHUB_ACTIONS_TOKEN`` env var or pass it at runtime — never commit it.
    """

    enabled: bool = True

    # GitHub PAT with actions:write scope.  Read from GITHUB_ACTIONS_TOKEN if
    # not set here (the field is optional so the contract can be constructed
    # from a YAML that omits it and falls back to the env var at call time).
    token: str = ""

    # Target repository (owner and repo name, e.g. "myorg" / "ml-pipeline").
    owner: str
    repo: str

    # Workflow to dispatch — may be the filename (e.g. "mlops-deploy.yml") or
    # the workflow ID (an integer) as a string.
    workflow_file: str = "mlops-deploy.yml"

    # Branch, tag, or SHA to run the dispatched workflow on.
    ref: str = "main"

    # Pipeline event names that trigger a dispatch.  Only "workflow_completed"
    # makes sense for a deployment trigger; add others for custom integrations.
    trigger_on: list[str] = Field(default_factory=lambda: ["workflow_completed"])

    # Additional key-value pairs forwarded verbatim as GitHub Actions inputs
    # (on top of the standard ones injected by GithubActionsClient).
    extra_inputs: dict[str, str] = Field(default_factory=dict)
