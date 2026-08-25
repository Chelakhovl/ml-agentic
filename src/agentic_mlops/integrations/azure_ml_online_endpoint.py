"""Azure ML Managed Online Endpoint deployer.

Real SDK v2 calls (MLClient.online_endpoints / online_deployments) — this backend
needs external connection info (subscription, resource group, workspace) that
DeploymentInput does not carry, so it must be constructed explicitly with an
AzureMLConfig and injected into DeploymentAgent, same pattern as
AzureMLModelRegistryClient / AzureMLTrainingRunner / AzureMLEvaluationRunner. See
CLI: ``deploy-model --backend azure_ml --azure-config configs/azure_ml.yaml``.

Deploys an already-registered Azure ML Model asset (azure_model_name +
azure_model_version — typically produced by a prior `register-model --backend
azure_ml` / model_registry step) to a Managed Online Endpoint, using
azure_jobs/score.py as the scoring script. Creates the endpoint if it doesn't
exist, creates/updates the deployment, then routes 100% traffic to it —
"deploying" here always means a single full-traffic cutover, not a blue/green
or canary rollout (not implemented).

Inject a client_factory (e.g. FakeAzureMLClientFactory) for tests — no real
Azure calls made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.azure_ml import AzureMLConfig, AzureMLEnvironmentConfig
from agentic_mlops.contracts.deployment import (
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
    DeploymentTarget,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class AzureMLOnlineEndpointDeployer:
    """Creates/updates a Managed Online Endpoint + Deployment and cuts traffic over to it."""

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )

            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    def deploy(self, inp: DeploymentInput, artifacts_dir: Path) -> DeploymentOutput:
        if not inp.azure_model_name or inp.azure_model_version is None:
            msg = "azure_model_name and azure_model_version are required for " "backend='azure_ml'."
            return DeploymentOutput(
                success=False, message=msg, status=DeploymentStatus.FAILED, errors=[msg]
            )

        endpoint_name = inp.endpoint_name or f"{inp.model_name}-{inp.target.value}"
        deployment_name = f"{inp.model_name}-v{inp.azure_model_version}".replace("_", "-")
        serving = self._config.serving

        try:
            ml_client = self._factory.create(self._config)

            endpoint = self._build_endpoint(endpoint_name)
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info(
                "Azure ML online endpoint provisioned", extra={"endpoint_name": endpoint_name}
            )

            deployment = self._build_deployment(deployment_name, endpoint_name, inp)
            ml_client.online_deployments.begin_create_or_update(deployment).result()
            logger.info(
                "Azure ML online deployment created",
                extra={"deployment_name": deployment_name, "endpoint_name": endpoint_name},
            )

            endpoint.traffic = {deployment_name: 100}
            final_endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            scoring_uri = getattr(final_endpoint, "scoring_uri", None)

            deployed_at = datetime.now(tz=UTC).isoformat()
            status = (
                DeploymentStatus.DEPLOYED_TO_PRODUCTION
                if inp.target == DeploymentTarget.PRODUCTION
                else DeploymentStatus.DEPLOYED_TO_STAGING
            )

            manifest = {
                "backend": "azure_ml",
                "endpoint_name": endpoint_name,
                "deployment_name": deployment_name,
                "azure_model_name": inp.azure_model_name,
                "azure_model_version": inp.azure_model_version,
                "scoring_uri": scoring_uri,
                "instance_type": serving.instance_type,
                "instance_count": serving.instance_count,
                "deployed_at": deployed_at,
                "rollback_plan": inp.rollback_plan,
            }
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = artifacts_dir / "azure_deployment_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

            logger.info(
                "Azure ML online endpoint deployment complete",
                extra={
                    "endpoint_name": endpoint_name,
                    "deployment_name": deployment_name,
                    "scoring_uri": scoring_uri,
                },
            )

            return DeploymentOutput(
                success=True,
                message=(
                    f"Model deployed to Azure ML Managed Online Endpoint "
                    f"'{endpoint_name}' (deployment '{deployment_name}')."
                ),
                status=status,
                endpoint_name=endpoint_name,
                scoring_uri=scoring_uri,
                azure_deployment_name=deployment_name,
                artifacts=[str(manifest_path)],
            )

        except Exception as exc:
            logger.error("AzureMLOnlineEndpointDeployer failed", extra={"error": str(exc)})
            return DeploymentOutput(
                success=False,
                message=f"Azure ML deployment failed: {exc}",
                status=DeploymentStatus.FAILED,
                errors=[str(exc)],
            )

    # ── SDK object construction ─────────────────────────────────────────────────

    def _build_endpoint(self, endpoint_name: str) -> Any:
        from azure.ai.ml.entities import ManagedOnlineEndpoint  # noqa: PLC0415

        return ManagedOnlineEndpoint(name=endpoint_name, auth_mode=self._config.serving.auth_mode)

    def _build_deployment(
        self, deployment_name: str, endpoint_name: str, inp: DeploymentInput
    ) -> Any:
        from azure.ai.ml.entities import (  # noqa: PLC0415
            CodeConfiguration,
            ManagedOnlineDeployment,
        )

        code_dir = str(Path(__file__).parent.parent / "azure_jobs")
        serving = self._config.serving
        env = self._build_environment(serving.environment or self._config.environment)

        return ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"azureml:{inp.azure_model_name}:{inp.azure_model_version}",
            environment=env,
            code_configuration=CodeConfiguration(code=code_dir, scoring_script="score.py"),
            instance_type=serving.instance_type,
            instance_count=serving.instance_count,
        )

    def _build_environment(self, env_cfg: AzureMLEnvironmentConfig) -> Any:
        from azure.ai.ml.entities import Environment  # noqa: PLC0415

        if env_cfg.mode.value == "registered":
            return env_cfg.registered_environment
        return Environment(image=env_cfg.base_image, conda_file=env_cfg.conda_file)
