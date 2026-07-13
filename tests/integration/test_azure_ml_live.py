"""Real, opt-in Azure ML connectivity check.

Requires a real Azure ML workspace + credentials (`az login`, or
AZURE_CLIENT_ID/AZURE_TENANT_ID/AZURE_CLIENT_SECRET) and `pip install
agentic-mlops-yolo[azure]`. Automatically skipped unless --azure-config is
passed, so a plain `pytest`, `pytest tests/unit`, or CI run never triggers
real Azure calls or cost.

Scope: authenticates and confirms the configured workspace + compute
target actually exist and are reachable -- read-only API calls, no compute
spin-up, no training job submitted, effectively free. This is deliberately
lighter than a full training-job run: it answers "is my azure_ml.yaml +
auth actually correct" before anyone spends money on a real `train
--runner azure-ml` job, which is a separate, manual, explicitly-costly
step documented in README.md's "Azure ML Training" section (~$0.50-1.50
for a 1-epoch coco8 run) -- deliberately not automated into a test here.

Run:
    pip install -e ".[dev,azure]"
    az login
    pytest tests/integration -m azure_integration --azure-config configs/azure_ml.yaml
"""

from __future__ import annotations

import pytest

from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.integrations.azure_ml_client import DefaultAzureMLClientFactory

pytestmark = pytest.mark.azure_integration


@pytest.fixture
def azure_config(request: pytest.FixtureRequest) -> AzureMLConfig:
    path = request.config.getoption("--azure-config")
    if not path:
        pytest.skip(
            "--azure-config not provided -- skipping real Azure ML integration test "
            "(pass e.g. --azure-config configs/azure_ml.yaml to run it)."
        )
    return AzureMLConfig.from_yaml(path)


def test_workspace_is_reachable(azure_config: AzureMLConfig) -> None:
    ml_client = DefaultAzureMLClientFactory().create(azure_config)
    workspace = ml_client.workspaces.get(azure_config.workspace_name)
    assert workspace.name == azure_config.workspace_name


def test_compute_target_exists(azure_config: AzureMLConfig) -> None:
    ml_client = DefaultAzureMLClientFactory().create(azure_config)
    compute = ml_client.compute.get(azure_config.compute_name)
    assert compute.name == azure_config.compute_name
