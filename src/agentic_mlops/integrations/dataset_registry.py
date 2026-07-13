"""Dataset version registry — filesystem-based, mirrors integrations/model_registry.py.

LocalDatasetVersionRegistry:
  <registry_dir>/<dataset_name>/
    versions/
      <N>/
        dataset/       ← full copy of the structured YOLO dataset at registration time
        lineage.json
        dataset_version_output.json
    latest.json         ← updated only after a successful new registration

Deduplication: before creating a new version, the content hash (over every file's
relative path + SHA-256, sorted) is compared against every existing version's
lineage.json. A match returns the existing version instead of copying again.

AzureMLDatasetRegistryClient registers the dataset as a real Azure ML Data
asset instead — see its own docstring below. No local hash-dedup there;
Azure owns versioning for that backend.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_mlops.contracts.dataset_versioning import (
    DatasetLineage,
    DatasetRegistryBackend,
    DatasetVersioningInput,
    DatasetVersioningOutput,
    DatasetVersionStatus,
)
from agentic_mlops.observability.logging import get_logger

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = get_logger(__name__)


def _sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataset(dataset_path: Path) -> str:
    """Deterministic content hash: sorted (relative_path, file_sha256) pairs."""
    h = hashlib.sha256()
    files = sorted(p for p in dataset_path.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(dataset_path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(_sha256_file(f).encode("utf-8"))
    return h.hexdigest()


class DatasetVersionRegistryClientBase(ABC):
    @abstractmethod
    def register(
        self,
        inp: DatasetVersioningInput,
        classes: list[str],
        validation_status: str | None,
        label_qa_status: str | None,
        artifacts_dir: Path,
    ) -> DatasetVersioningOutput:
        ...


class LocalDatasetVersionRegistry(DatasetVersionRegistryClientBase):
    """Filesystem dataset version registry with hash-based deduplication."""

    def register(
        self,
        inp: DatasetVersioningInput,
        classes: list[str],
        validation_status: str | None,
        label_qa_status: str | None,
        artifacts_dir: Path,
    ) -> DatasetVersioningOutput:
        dataset_path = Path(inp.dataset_path).resolve()
        content_hash = hash_dataset(dataset_path)

        registry_root = Path(inp.registry_dir) / inp.dataset_name
        versions_dir = registry_root / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = sorted(
            int(p.name) for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )

        for v in existing_versions:
            lineage_path = versions_dir / str(v) / "lineage.json"
            if not lineage_path.exists():
                continue
            existing = json.loads(lineage_path.read_text(encoding="utf-8"))
            if existing.get("hash") == content_hash:
                logger.info(
                    "Dataset content unchanged — deduplicated",
                    extra={"dataset_name": inp.dataset_name, "version": v, "hash": content_hash},
                )
                return DatasetVersioningOutput(
                    success=True,
                    message=(
                        f"Dataset content matches existing version {v} — "
                        "no new version created."
                    ),
                    status=DatasetVersionStatus.DEDUPLICATED,
                    dataset_name=inp.dataset_name,
                    version=v,
                    dataset_version_path=str(versions_dir / str(v) / "dataset"),
                    dataset_version_artifact=str(lineage_path),
                    hash=content_hash,
                    lineage=DatasetLineage.model_validate(existing),
                )

        version = (existing_versions[-1] + 1) if existing_versions else 1
        version_dir = versions_dir / str(version)
        dataset_copy_dir = version_dir / "dataset"

        try:
            shutil.copytree(dataset_path, dataset_copy_dir)

            registered_at = datetime.now(tz=UTC).isoformat()
            lineage = DatasetLineage(
                dataset_name=inp.dataset_name,
                version=version,
                parent_version=inp.parent_version,
                workflow_id=inp.workflow_id,
                source_batches=inp.source_batches,
                classes=classes,
                hash=content_hash,
                approved_by=inp.approved_by,
                validation_status=validation_status,
                label_qa_status=label_qa_status,
                registered_at=registered_at,
                source_dataset_path=str(dataset_path),
            )

            lineage_path = version_dir / "lineage.json"
            lineage_path.write_text(
                json.dumps(lineage.model_dump(mode="json"), indent=2), encoding="utf-8"
            )

            output = DatasetVersioningOutput(
                success=True,
                message=f"Dataset '{inp.dataset_name}' registered as version {version}.",
                status=DatasetVersionStatus.REGISTERED,
                dataset_name=inp.dataset_name,
                version=version,
                dataset_version_path=str(dataset_copy_dir),
                dataset_version_artifact=str(lineage_path),
                hash=content_hash,
                lineage=lineage,
                artifacts=[str(lineage_path)],
            )

            out_path = version_dir / "dataset_version_output.json"
            out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            output.artifacts.append(str(out_path))

            # Also write to agent artifacts_dir for workflow tracking (mirrors ModelRegistryAgent)
            agent_out_path = artifacts_dir / "dataset_version_output.json"
            agent_out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
            )

            latest_path = registry_root / "latest.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "dataset_name": inp.dataset_name,
                        "version": version,
                        "hash": content_hash,
                        "registered_at": registered_at,
                        "dataset_path": str(dataset_copy_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "Dataset version registered",
                extra={
                    "dataset_name": inp.dataset_name,
                    "version": version,
                    "hash": content_hash,
                },
            )
            return output

        except Exception as exc:
            if version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)
            logger.error(
                "LocalDatasetVersionRegistry registration failed", extra={"error": str(exc)}
            )
            return DatasetVersioningOutput(
                success=False,
                message=f"Dataset versioning failed: {exc}",
                status=DatasetVersionStatus.FAILED,
                dataset_name=inp.dataset_name,
                errors=[str(exc)],
            )


class FakeDatasetVersionRegistry(DatasetVersionRegistryClientBase):
    """In-memory test double. Records calls; never touches disk."""

    def __init__(self) -> None:
        self.calls: list[DatasetVersioningInput] = []
        self._next_version: int = 1

    def register(
        self,
        inp: DatasetVersioningInput,
        classes: list[str],
        validation_status: str | None,
        label_qa_status: str | None,
        artifacts_dir: Path,
    ) -> DatasetVersioningOutput:
        self.calls.append(inp)
        version = self._next_version
        self._next_version += 1
        return DatasetVersioningOutput(
            success=True,
            message=f"Fake registration: '{inp.dataset_name}' v{version}.",
            status=DatasetVersionStatus.REGISTERED,
            dataset_name=inp.dataset_name,
            version=version,
            dataset_version_path=f"/fake/registry/{inp.dataset_name}/versions/{version}/dataset",
            hash="fake_hash",
        )


class AzureMLDatasetRegistryClient(DatasetVersionRegistryClientBase):
    """Registers the structured dataset directory as an Azure ML Data asset (MLClient.data).

    Unlike LocalDatasetVersionRegistry, this backend needs external connection info
    (subscription, resource group, workspace) that DatasetVersioningInput does not
    carry — so it must be constructed explicitly with an AzureMLConfig and injected
    as ``registry_client=`` (create_dataset_registry_client() cannot build one on
    its own). See CLI: ``version-dataset --backend azure_ml --azure-config
    configs/azure_ml.yaml``.

    No local content-hash dedup here (unlike the local backend) — Azure ML assigns
    and owns the version number for a given asset name, same "just register, let
    the service handle versioning" approach as AzureMLModelRegistryClient.

    Inject a client_factory (e.g. FakeAzureMLClientFactory) for tests — no real
    Azure calls made.
    """

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )
            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    def register(
        self,
        inp: DatasetVersioningInput,
        classes: list[str],
        validation_status: str | None,
        label_qa_status: str | None,
        artifacts_dir: Path,
    ) -> DatasetVersioningOutput:
        try:
            dataset_path = Path(inp.dataset_path).resolve()
            content_hash = hash_dataset(dataset_path)
            ml_client = self._factory.create(self._config)

            data_asset = self._build_data_asset(inp, classes)
            registered = ml_client.data.create_or_update(data_asset)
            version = int(registered.version)
            dataset_version_path = getattr(registered, "id", None) or (
                f"azureml:{inp.dataset_name}:{version}"
            )
            registered_at = datetime.now(tz=UTC).isoformat()

            lineage = DatasetLineage(
                dataset_name=inp.dataset_name,
                version=version,
                parent_version=inp.parent_version,
                workflow_id=inp.workflow_id,
                source_batches=inp.source_batches,
                classes=classes,
                hash=content_hash,
                approved_by=inp.approved_by,
                validation_status=validation_status,
                label_qa_status=label_qa_status,
                registered_at=registered_at,
                source_dataset_path=str(dataset_path),
            )

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            lineage_path = artifacts_dir / "lineage.json"
            lineage_path.write_text(
                json.dumps(lineage.model_dump(mode="json"), indent=2), encoding="utf-8"
            )

            output = DatasetVersioningOutput(
                success=True,
                message=(
                    f"Dataset '{inp.dataset_name}' registered as Azure ML data asset "
                    f"version {version}."
                ),
                status=DatasetVersionStatus.REGISTERED,
                dataset_name=inp.dataset_name,
                version=version,
                dataset_version_path=dataset_version_path,
                dataset_version_artifact=str(lineage_path),
                hash=content_hash,
                lineage=lineage,
                artifacts=[str(lineage_path)],
                metadata={
                    "azure_workspace": self._config.workspace_name,
                    "azure_data_asset_id": dataset_version_path,
                },
            )

            out_path = artifacts_dir / "dataset_version_output.json"
            out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            output.artifacts.append(str(out_path))

            logger.info(
                "Dataset version registered in Azure ML",
                extra={
                    "dataset_name": inp.dataset_name,
                    "version": version,
                    "id": dataset_version_path,
                },
            )
            return output

        except Exception as exc:
            logger.error(
                "AzureMLDatasetRegistryClient registration failed", extra={"error": str(exc)}
            )
            return DatasetVersioningOutput(
                success=False,
                message=f"Azure ML dataset registration failed: {exc}",
                status=DatasetVersionStatus.FAILED,
                dataset_name=inp.dataset_name,
                errors=[str(exc)],
            )

    def _build_data_asset(self, inp: DatasetVersioningInput, classes: list[str]) -> Any:
        from azure.ai.ml.constants import AssetTypes  # noqa: PLC0415
        from azure.ai.ml.entities import Data  # noqa: PLC0415

        return Data(
            path=str(Path(inp.dataset_path).resolve()),
            name=inp.dataset_name,
            type=AssetTypes.URI_FOLDER,
            description="Structured YOLO dataset registered by agentic-mlops.",
            tags={"classes": ",".join(classes), "num_classes": str(len(classes))},
        )


def create_dataset_registry_client(
    backend: DatasetRegistryBackend,
) -> DatasetVersionRegistryClientBase:
    """Resolve the dataset registry client for a given backend.

    Used by DatasetVersioningAgent when no client was explicitly injected, so
    that DatasetVersioningInput.backend actually determines where the dataset
    is registered.

    AZURE_ML cannot be resolved here: it needs an AzureMLConfig (subscription,
    resource group, workspace) that DatasetVersioningInput does not carry. Build
    AzureMLDatasetRegistryClient(config) yourself and pass it as registry_client=
    (see CLI: version-dataset --backend azure_ml --azure-config ...).
    """
    if backend == DatasetRegistryBackend.LOCAL:
        return LocalDatasetVersionRegistry()
    if backend == DatasetRegistryBackend.AZURE_ML:
        raise ValueError(
            "Azure ML Dataset Registry backend requires an AzureMLConfig. Construct "
            "AzureMLDatasetRegistryClient(config) explicitly and pass it as "
            "registry_client= (see CLI: version-dataset --backend azure_ml "
            "--azure-config ...)."
        )
    raise ValueError(f"Unknown dataset registry backend: {backend}")
