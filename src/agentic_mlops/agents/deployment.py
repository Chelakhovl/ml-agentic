"""Deployment Agent — prepares a registered model for serving and deploys it.

Two backends (DeploymentInput.backend): "local" (default — export + smoke test +
versioned local release directory) or "azure_ml" (real Managed Online Endpoint via
AzureMLOnlineEndpointDeployer, injected into ModelDeployer). See tools/deployer.py.

Staging can proceed automatically once smoke tests pass (spec: "semi-automatic").
Production requires a rollback plan and an approved production_approval_path — the
same "read a status='approved' JSON from disk" gate ModelRegistryAgent already
uses. Never deploys to production without that approval, matching the spec's
"no auto-production-promote"-style safety rule for this agent.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.deployment import DeploymentInput, DeploymentOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.deployer import ModelDeployer
from agentic_mlops.tools.report_writer import ReportWriter


class DeploymentAgent(BaseAgent):
    """Exports, smoke-tests, and deploys a registered model to staging/production.

    Output artifacts:
        artifacts_dir/deployment_report.json
        artifacts_dir/deployment_report.md
    Deployment artifacts (in deployment_dir/<endpoint_name>/):
        releases/<N>/<exported_model>, deployment_manifest.json
        current.json
    """

    def __init__(
        self,
        artifacts_dir: Path,
        deployer: ModelDeployer | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._deployer = deployer or ModelDeployer()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: DeploymentInput) -> DeploymentOutput:
        self._log_start(model_name=input.model_name, target=str(input.target))

        output = self._deployer.deploy(input, self.artifacts_dir)

        json_path, md_path = self._report_writer.write_deployment_report(output, self.artifacts_dir)
        output.deployment_report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id and output.success:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=str(output.status),
            endpoint_name=output.endpoint_name,
            release=str(output.release),
        )
        return output

    def _log_to_mlflow(self, inp: DeploymentInput, output: DeploymentOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "deployment.model_name": inp.model_name,
            "deployment.target": str(inp.target),
            "deployment.export_format": str(inp.export_format),
            "deployment.endpoint_name": output.endpoint_name or "",
        }
        client.log_params(rid, params)

        if output.release is not None:
            client.log_metrics(rid, {"deployment.release": float(output.release)})

        client.log_tags(
            rid,
            {
                "workflow_step": "deployment",
                "deployment_status": str(output.status),
            },
        )

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
