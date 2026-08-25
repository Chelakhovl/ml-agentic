"""Integration test — Azure ML Model Registry (opt-in, real Azure ML calls).

Registers a stub model file as an Azure ML Model asset, then retrieves it to
confirm it exists.  Read-only retrieval after registration — effectively free.

Run:
    pytest tests/integration -m azure_integration --azure-config configs/azure_ml.yaml
"""

from __future__ import annotations

import pytest

from agentic_mlops.contracts.azure_ml import AzureMLConfig

pytestmark = pytest.mark.azure_integration


@pytest.fixture
def azure_config(request: pytest.FixtureRequest) -> AzureMLConfig:
    path = request.config.getoption("--azure-config")
    if not path:
        pytest.skip("--azure-config not provided")
    return AzureMLConfig.from_yaml(path)


def test_register_model_creates_asset(azure_config: AzureMLConfig, tmp_path) -> None:
    """AzureMLModelRegistryClient.register() should create a Model asset."""
    from agentic_mlops.integrations.model_registry import AzureMLModelRegistryClient

    # Create a stub weights file
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"stub weights")

    client = AzureMLModelRegistryClient(azure_config)
    result = client.register(
        model_name="integration-test-model",
        weights_path=weights,
        metadata={"source": "pytest-integration-test"},
    )

    assert result.success, f"Registration failed: {result.message}"
    assert result.model_version is not None


def test_get_model_returns_registered_version(azure_config: AzureMLConfig, tmp_path) -> None:
    """After registration, get_model() should return the registered version."""
    from agentic_mlops.integrations.model_registry import AzureMLModelRegistryClient

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"stub get test")

    client = AzureMLModelRegistryClient(azure_config)
    reg = client.register(
        model_name="integration-test-get",
        weights_path=weights,
        metadata={},
    )
    assert reg.success

    fetched = client.get_model("integration-test-get", reg.model_version)
    assert fetched is not None
    assert fetched.get("version") == reg.model_version
