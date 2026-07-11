"""Azure ML training integration.

SDK v2 client factories used by AzureMLTrainingRunner / AzureMLEvaluationRunner /
AzureMLModelRegistryClient (see tools/training_runner.py, tools/evaluation_runner.py,
integrations/model_registry.py) — this module only builds the MLClient connection,
it does not submit jobs itself.

Real implementation requires:
  pip install agentic-mlops-yolo[azure]
  env vars: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = logging.getLogger(__name__)


# ── SDK v2 client factories ────────────────────────────────────────────────────


class AzureMLClientFactory(Protocol):
    """Protocol for factories that produce azure.ai.ml.MLClient instances."""

    def create(self, config: AzureMLConfig) -> Any:
        """Return a configured MLClient."""
        ...


class DefaultAzureMLClientFactory:
    """Creates a real MLClient using DefaultAzureCredential."""

    def create(self, config: AzureMLConfig) -> Any:
        try:
            from azure.ai.ml import MLClient  # noqa: PLC0415
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Azure SDK is not installed. Run: pip install 'agentic-mlops-yolo[azure]'"
            ) from exc
        try:
            cred = DefaultAzureCredential()
        except Exception as exc:
            raise RuntimeError(
                "Azure authentication failed. Run 'az login' or set service principal env vars."
            ) from exc
        return MLClient(
            credential=cred,
            subscription_id=config.subscription_id,
            resource_group_name=config.resource_group,
            workspace_name=config.workspace_name,
        )


class _FakeJob:
    """Minimal stub returned by FakeJobsOperations.create_or_update / get."""

    def __init__(self, name: str, status: str) -> None:
        self.name = name
        self.status = status
        self.studio_url = f"https://ml.azure.com/runs/{name}"


class FakeJobsOperations:
    """Fake azure.ai.ml MLClient.jobs for unit tests."""

    def __init__(self, job_status: str = "Completed") -> None:
        self._status = job_status
        self._counter = 0
        self.created: list[Any] = []
        self.streamed: list[str] = []
        self.cancelled: list[str] = []
        self.downloaded: list[tuple[str, str, str]] = []

    def create_or_update(self, job: Any) -> _FakeJob:
        self._counter += 1
        name = f"fake_azure_job_{self._counter:04d}"
        self.created.append(job)
        return _FakeJob(name=name, status="Running")

    def stream(self, name: str) -> None:
        self.streamed.append(name)

    def get(self, name: str) -> _FakeJob:
        return _FakeJob(name=name, status=self._status)

    def download(self, name: str, output_name: str, download_path: str) -> None:
        self.downloaded.append((name, output_name, download_path))
        # Create the expected artifact files so runner code can find them
        out_dir = Path(download_path) / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "best.pt").write_bytes(b"fake-best-pt")
        (out_dir / "last.pt").write_bytes(b"fake-last-pt")
        (out_dir / "results.csv").write_text("epoch,train/loss\n1,0.5\n", encoding="utf-8")

    def cancel(self, name: str) -> None:
        self.cancelled.append(name)


class _FakeRegisteredModel:
    """Minimal stub returned by FakeModelsOperations.create_or_update."""

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.id = (
            "azureml://subscriptions/fake/resourceGroups/fake/workspaces/fake"
            f"/models/{name}/versions/{version}"
        )


class FakeModelsOperations:
    """Fake azure.ai.ml MLClient.models for unit tests."""

    def __init__(self) -> None:
        self.created: list[Any] = []
        self._next_version: dict[str, int] = {}

    def create_or_update(self, model: Any) -> _FakeRegisteredModel:
        name = model.name
        version = self._next_version.get(name, 0) + 1
        self._next_version[name] = version
        self.created.append(model)
        return _FakeRegisteredModel(name=name, version=str(version))


class _FakePoller:
    """Mimics azure.core.polling.LROPoller — real online_endpoints/online_deployments
    calls return one of these; .result() blocks until done and returns the resource."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _FakeOnlineEndpoint:
    def __init__(self, name: str, traffic: dict[str, int] | None = None) -> None:
        self.name = name
        self.auth_mode = "key"
        self.traffic = traffic or {}
        self.scoring_uri = f"https://{name}.fake.inference.ml.azure.com/score"


class FakeOnlineEndpointsOperations:
    """Fake azure.ai.ml MLClient.online_endpoints for unit tests."""

    def __init__(self) -> None:
        self.created: list[Any] = []
        self._endpoints: dict[str, _FakeOnlineEndpoint] = {}

    def begin_create_or_update(self, endpoint: Any) -> _FakePoller:
        self.created.append(endpoint)
        existing = self._endpoints.get(endpoint.name)
        traffic = getattr(endpoint, "traffic", None) or (existing.traffic if existing else {})
        fake = _FakeOnlineEndpoint(name=endpoint.name, traffic=traffic)
        self._endpoints[endpoint.name] = fake
        return _FakePoller(fake)

    def get(self, name: str) -> _FakeOnlineEndpoint:
        return self._endpoints[name]


class _FakeOnlineDeployment:
    def __init__(self, name: str, endpoint_name: str) -> None:
        self.name = name
        self.endpoint_name = endpoint_name


class FakeOnlineDeploymentsOperations:
    """Fake azure.ai.ml MLClient.online_deployments for unit tests."""

    def __init__(self) -> None:
        self.created: list[Any] = []

    def begin_create_or_update(self, deployment: Any) -> _FakePoller:
        self.created.append(deployment)
        fake = _FakeOnlineDeployment(name=deployment.name, endpoint_name=deployment.endpoint_name)
        return _FakePoller(fake)


class FakeMLClient:
    """Minimal MLClient stub for unit tests."""

    def __init__(self, job_status: str = "Completed") -> None:
        self.jobs = FakeJobsOperations(job_status=job_status)
        self.models = FakeModelsOperations()
        self.online_endpoints = FakeOnlineEndpointsOperations()
        self.online_deployments = FakeOnlineDeploymentsOperations()


class FakeAzureMLClientFactory:
    """Injects a FakeMLClient — use in unit tests instead of DefaultAzureMLClientFactory."""

    def __init__(self, job_status: str = "Completed") -> None:
        self._status = job_status
        self._clients: list[FakeMLClient] = []

    def create(self, config: AzureMLConfig) -> FakeMLClient:
        client = FakeMLClient(job_status=self._status)
        self._clients.append(client)
        return client

    @property
    def last_client(self) -> FakeMLClient | None:
        return self._clients[-1] if self._clients else None
