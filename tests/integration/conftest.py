"""Shared fixtures and hooks for integration tests.

All tests here are guarded by ``pytestmark = pytest.mark.azure_integration``
and skip automatically unless ``--azure-config`` is passed on the CLI.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --azure-config CLI option (idempotent — unit conftest also registers it)."""
    try:
        parser.addoption(
            "--azure-config",
            action="store",
            default=None,
            help="Path to azure_ml.yaml for real Azure ML integration tests",
        )
    except ValueError:
        # Already registered by the top-level conftest — that's fine.
        pass


@pytest.fixture(scope="session")
def azure_config_path(request: pytest.FixtureRequest):
    """Return the --azure-config path, or skip the test if not provided."""
    path = request.config.getoption("--azure-config", default=None)
    if not path:
        pytest.skip(
            "--azure-config not provided — skipping Azure ML integration tests. "
            "Pass e.g. --azure-config configs/azure_ml.yaml to run them."
        )
    return path
