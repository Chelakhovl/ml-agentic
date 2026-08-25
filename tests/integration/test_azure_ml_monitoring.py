"""Integration test — Application Insights log ingestion (opt-in).

Requires: pip install -e ".[dev,azure]", az login (or SP env vars),
--azure-config configs/azure_ml.yaml, AND an App Insights workspace_id
set in azure_ml.yaml under the `appinsights_workspace_id` key.

Scope: queries the App Insights `traces` table for inference records emitted by
`azure_jobs/score.py`, then runs MonitoringAgent over the retrieved records.
This is a read-only API call — no compute or cost.

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
        pytest.skip("--azure-config not provided -- skipping App Insights integration test.")
    return AzureMLConfig.from_yaml(path)


def test_appinsights_log_client_connects(azure_config: AzureMLConfig) -> None:
    """ApplicationInsightsLogClient should connect and return (possibly empty) records."""
    try:
        from agentic_mlops.integrations.appinsights_log_client import (
            ApplicationInsightsLogClient,
        )
    except ImportError:
        pytest.skip("azure-monitor-query not installed")

    workspace_id = getattr(azure_config, "appinsights_workspace_id", None)
    if not workspace_id:
        pytest.skip(
            "azure_ml.yaml does not set appinsights_workspace_id — "
            "skipping App Insights connectivity test."
        )

    client = ApplicationInsightsLogClient(
        workspace_id=workspace_id,
        endpoint_name="integration-test",
        window_seconds=3600,
    )
    records = client.fetch_records()
    # Records may be empty (no recent traffic) — just assert no exception
    assert isinstance(records, list)


def test_monitoring_agent_with_appinsights(azure_config: AzureMLConfig, tmp_path) -> None:
    """MonitoringAgent should complete with ApplicationInsightsLogClient (even with 0 records)."""
    try:
        from agentic_mlops.integrations.appinsights_log_client import (
            ApplicationInsightsLogClient,
        )
    except ImportError:
        pytest.skip("azure-monitor-query not installed")

    workspace_id = getattr(azure_config, "appinsights_workspace_id", None)
    if not workspace_id:
        pytest.skip("appinsights_workspace_id not configured")

    from agentic_mlops.agents.monitoring import MonitoringAgent
    from agentic_mlops.contracts.monitoring import MonitoringInput

    client = ApplicationInsightsLogClient(
        workspace_id=workspace_id,
        endpoint_name="integration-test-endpoint",
        window_seconds=3600,
    )
    agent = MonitoringAgent(artifacts_dir=tmp_path, log_client=client)
    result = agent.run(
        MonitoringInput(
            predictions_log_path=None,  # log_client takes precedence
            endpoint_name="integration-test-endpoint",
            output_dir=tmp_path,
        )
    )
    # success=True even with 0 records (nothing to alert on)
    assert result.success
