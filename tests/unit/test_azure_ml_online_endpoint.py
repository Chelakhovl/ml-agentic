"""Unit tests for AzureMLOnlineEndpointDeployer.

azure-ai-ml is NOT required to be installed — azure.ai.ml.entities is mocked;
Azure connectivity is replaced with FakeAzureMLClientFactory/FakeMLClient, same
pattern as TestAzureMLModelRegistryClient in test_model_registry.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.contracts.deployment import (
    DeploymentBackend,
    DeploymentInput,
    DeploymentStatus,
    DeploymentTarget,
)
from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory, FakeMLClient
from agentic_mlops.integrations.azure_ml_online_endpoint import AzureMLOnlineEndpointDeployer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _minimal_azure_config(**overrides: object) -> AzureMLConfig:
    defaults: dict = {
        "subscription_id": "sub-123",
        "resource_group": "rg-test",
        "workspace_name": "ws-test",
        "compute_name": "gpu-cluster",
        "environment": {"mode": "registered", "registered_environment": "azureml:yolo-env:1"},
    }
    defaults.update(overrides)
    return AzureMLConfig.model_validate(defaults)


def _mock_azure_ml_entities_module() -> MagicMock:
    def fake_endpoint(**kwargs: object) -> MagicMock:
        m = MagicMock()
        m.name = kwargs.get("name")
        m.auth_mode = kwargs.get("auth_mode")
        m.traffic = {}
        return m

    def fake_deployment(**kwargs: object) -> MagicMock:
        m = MagicMock()
        m.name = kwargs.get("name")
        m.endpoint_name = kwargs.get("endpoint_name")
        m.model = kwargs.get("model")
        m.instance_type = kwargs.get("instance_type")
        m.instance_count = kwargs.get("instance_count")
        return m

    def fake_code_configuration(**kwargs: object) -> MagicMock:
        m = MagicMock()
        m.code = kwargs.get("code")
        m.scoring_script = kwargs.get("scoring_script")
        return m

    def fake_environment(**kwargs: object) -> MagicMock:
        m = MagicMock()
        m.image = kwargs.get("image")
        m.conda_file = kwargs.get("conda_file")
        return m

    return MagicMock(
        ManagedOnlineEndpoint=fake_endpoint,
        ManagedOnlineDeployment=fake_deployment,
        CodeConfiguration=fake_code_configuration,
        Environment=fake_environment,
    )


def _make_deployment_input(**overrides: object) -> DeploymentInput:
    defaults: dict = {
        "model_name": "factory-defects",
        "backend": DeploymentBackend.AZURE_ML,
        "azure_model_name": "factory-defects-model",
        "azure_model_version": 3,
        "target": DeploymentTarget.STAGING,
    }
    defaults.update(overrides)
    return DeploymentInput(**defaults)


def _deploy(deployer: AzureMLOnlineEndpointDeployer, inp: DeploymentInput, tmp_path: Path):
    mock_entities = _mock_azure_ml_entities_module()
    with patch.dict(sys.modules, {"azure.ai.ml.entities": mock_entities}):
        return deployer.deploy(inp, tmp_path / "artifacts")


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_deploy_succeeds_and_returns_scoring_uri(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    result = _deploy(deployer, _make_deployment_input(), tmp_path)

    assert result.success is True
    assert result.status == DeploymentStatus.DEPLOYED_TO_STAGING
    assert result.endpoint_name == "factory-defects-staging"
    assert result.scoring_uri is not None
    assert result.azure_deployment_name == "factory-defects-v3"


def test_deploy_to_production_status(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    inp = _make_deployment_input(
        target=DeploymentTarget.PRODUCTION,
        rollback_plan="Route traffic back to previous deployment.",
    )
    result = _deploy(deployer, inp, tmp_path)

    assert result.success is True
    assert result.status == DeploymentStatus.DEPLOYED_TO_PRODUCTION


def test_missing_azure_model_name_fails(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    inp = _make_deployment_input(azure_model_name=None)
    result = _deploy(deployer, inp, tmp_path)

    assert result.success is False
    assert result.status == DeploymentStatus.FAILED
    assert any("azure_model_name" in e for e in result.errors)


def test_deployment_references_correct_model_version(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    _deploy(deployer, _make_deployment_input(), tmp_path)

    client: FakeMLClient = factory.last_client
    deployment_obj = client.online_deployments.created[0]
    assert deployment_obj.model == "azureml:factory-defects-model:3"
    assert deployment_obj.instance_type == "Standard_DS2_v2"


def test_traffic_routed_100_percent_to_new_deployment(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    _deploy(deployer, _make_deployment_input(), tmp_path)

    client: FakeMLClient = factory.last_client
    final_endpoint = client.online_endpoints.created[-1]
    assert final_endpoint.traffic == {"factory-defects-v3": 100}


def test_custom_instance_type_and_count_used(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    cfg = _minimal_azure_config(serving={"instance_type": "Standard_DS3_v2", "instance_count": 2})
    deployer = AzureMLOnlineEndpointDeployer(cfg, client_factory=factory)
    _deploy(deployer, _make_deployment_input(), tmp_path)

    client: FakeMLClient = factory.last_client
    deployment_obj = client.online_deployments.created[0]
    assert deployment_obj.instance_type == "Standard_DS3_v2"
    assert deployment_obj.instance_count == 2


def test_manifest_written(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    result = _deploy(deployer, _make_deployment_input(), tmp_path)

    manifest_path = Path(result.artifacts[0])
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["backend"] == "azure_ml"
    assert data["azure_model_name"] == "factory-defects-model"
    assert data["azure_model_version"] == 3


def test_custom_endpoint_name_respected(tmp_path: Path) -> None:
    factory = FakeAzureMLClientFactory()
    deployer = AzureMLOnlineEndpointDeployer(_minimal_azure_config(), client_factory=factory)
    result = _deploy(deployer, _make_deployment_input(endpoint_name="custom-endpoint"), tmp_path)

    assert result.endpoint_name == "custom-endpoint"


def test_sdk_failure_produces_failed_output(tmp_path: Path) -> None:
    class _RaisingFactory:
        def create(self, config: AzureMLConfig) -> FakeMLClient:
            raise RuntimeError("Azure authentication failed.")

    deployer = AzureMLOnlineEndpointDeployer(
        _minimal_azure_config(), client_factory=_RaisingFactory()
    )
    result = _deploy(deployer, _make_deployment_input(), tmp_path)

    assert result.success is False
    assert result.status == DeploymentStatus.FAILED
    assert any("authentication" in e.lower() for e in result.errors)
