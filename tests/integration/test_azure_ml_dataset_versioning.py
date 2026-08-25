"""Integration test — Azure ML Dataset Versioning (opt-in, real Azure ML calls).

Requires: pip install -e ".[dev,azure]", az login (or SP env vars),
and --azure-config configs/azure_ml.yaml.

Scope: registers a tiny temp dataset directory as an Azure ML Data asset and
verifies the asset name / version are visible via the workspace API.
No training or compute spin-up — this is effectively free (API call only).

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
        pytest.skip(
            "--azure-config not provided -- skipping Azure ML dataset versioning integration test."
        )
    return AzureMLConfig.from_yaml(path)


def test_dataset_registration_creates_asset(azure_config: AzureMLConfig, tmp_path) -> None:
    """Register a local directory as an Azure ML Data asset and verify it appears."""
    from agentic_mlops.integrations.dataset_registry import AzureMLDatasetRegistryClient

    # Minimal dataset structure
    dataset_dir = tmp_path / "test_dataset"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0stub")
    (dataset_dir / "data.yaml").write_text("nc: 1\nnames: [test]\n")

    client = AzureMLDatasetRegistryClient(azure_config)
    result = client.register(
        dataset_name="integration-test-dataset",
        dataset_path=dataset_dir,
        version_note="integration test from pytest",
    )

    assert result.success, f"Registration failed: {result.message}"
    assert result.version is not None
    assert result.dataset_name == "integration-test-dataset"


def test_list_dataset_versions(azure_config: AzureMLConfig, tmp_path) -> None:
    """After registration, list_versions should return at least one version."""
    from agentic_mlops.integrations.dataset_registry import AzureMLDatasetRegistryClient

    dataset_dir = tmp_path / "ds_list_test"
    dataset_dir.mkdir()
    (dataset_dir / "data.yaml").write_text("nc: 1\nnames: [item]\n")

    client = AzureMLDatasetRegistryClient(azure_config)
    # Register first
    client.register(
        dataset_name="integration-test-list",
        dataset_path=dataset_dir,
        version_note="list test",
    )
    versions = client.list_versions("integration-test-list")
    assert len(versions) >= 1
