"""Azure ML training integration.

Production class: AzureMLTrainingClient — raises NotImplementedError.
Test class: FakeAzureMLTrainingClient — returns deterministic fake data.

Real implementation requires:
  pip install agentic-mlops-yolo[azure]
  env vars: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agentic_mlops.contracts.training import TrainingConfig, TrainingJobStatus

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = logging.getLogger(__name__)


class AzureMLTrainingClientBase(ABC):
    """Abstract interface for Azure ML training operations."""

    @abstractmethod
    def submit_training_job(
        self,
        config: TrainingConfig,
        dataset_path: str,
        data_yaml_path: str,
    ) -> str:
        """Submit a YOLO training job. Returns the Azure ML job ID."""
        ...

    @abstractmethod
    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Return the current status of a submitted job."""
        ...

    @abstractmethod
    def get_job_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        """Download job output artifacts to output_dir. Returns local file paths."""
        ...


class AzureMLTrainingClient(AzureMLTrainingClientBase):
    """Production Azure ML client stub.

    All methods raise NotImplementedError until azure-ai-ml is wired up.
    """

    def __init__(self) -> None:
        # TODO: from azure.ai.ml import MLClient
        # TODO: from azure.identity import DefaultAzureCredential
        # TODO: self._client = MLClient(
        #     credential=DefaultAzureCredential(),
        #     subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        #     resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        #     workspace_name=os.environ["AZURE_ML_WORKSPACE"],
        # )
        logger.warning("AzureMLTrainingClient is a stub — no real Azure connection.")

    def submit_training_job(
        self,
        config: TrainingConfig,
        dataset_path: str,
        data_yaml_path: str,
    ) -> str:
        # TODO: build a CommandJob from config and submit via self._client.jobs.create_or_update
        raise NotImplementedError(
            "Azure ML training is not implemented. Set mode=local_dry_run or local_train."
        )

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        raise NotImplementedError

    def get_job_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        raise NotImplementedError


class FakeAzureMLTrainingClient(AzureMLTrainingClientBase):
    """In-memory fake for unit tests.

    Records every call so tests can assert on interactions.
    """

    def __init__(self) -> None:
        self.submitted_jobs: list[dict] = []
        self.status_map: dict[str, TrainingJobStatus] = {}
        self._next_job_id = 0

    def submit_training_job(
        self,
        config: TrainingConfig,
        dataset_path: str,
        data_yaml_path: str,
    ) -> str:
        job_id = f"fake_job_{self._next_job_id:04d}"
        self._next_job_id += 1
        self.submitted_jobs.append(
            {"job_id": job_id, "config": config, "dataset_path": dataset_path}
        )
        self.status_map[job_id] = TrainingJobStatus.COMPLETED
        return job_id

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        return self.status_map.get(job_id, TrainingJobStatus.FAILED)

    def get_job_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        return []


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


class FakeMLClient:
    """Minimal MLClient stub for unit tests."""

    def __init__(self, job_status: str = "Completed") -> None:
        self.jobs = FakeJobsOperations(job_status=job_status)
        self.models = FakeModelsOperations()


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
