"""Artifact store abstraction for uploading workflow artifacts to Azure Blob Storage.

Usage pattern — upload is additive: local artifacts are always written first by the
runners/agents, then the workflow layer calls upload_directory / upload_file to mirror
them to blob.  blob is never the primary artifact source; local disk remains
authoritative for resume.

Three implementations:
  NoOpArtifactStore   — default, safe for local-only runs (no azure-storage-blob needed)
  AzureBlobArtifactStore — uploads to Azure Blob; resolves account/container from the
                            workspace's default datastore (auto mode) or from explicit
                            account_url / container_name in AzureMLStorageConfig
  FakeArtifactStore   — in-memory test double; records all calls in .uploaded / .downloaded
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = logging.getLogger(__name__)

_SKIP_DIRS = {"__pycache__", "_pipeline_outputs"}
_SKIP_SUFFIXES = {".pyc"}


# ── Protocol ──────────────────────────────────────────────────────────────────


class ArtifactStore:
    """Minimal interface all store implementations satisfy."""

    def upload_file(self, local_path: Path, blob_key: str) -> str:
        """Upload one file; returns blob URI (empty string on NoOp)."""
        raise NotImplementedError

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> list[str]:
        """Upload every file under local_dir; returns list of blob URIs."""
        raise NotImplementedError

    def download_file(self, blob_key: str, local_path: Path) -> None:
        """Download one blob to local_path."""
        raise NotImplementedError

    def exists(self, blob_key: str) -> bool:
        """Return True if the blob key exists in the container."""
        raise NotImplementedError


# ── NoOp ──────────────────────────────────────────────────────────────────────


class NoOpArtifactStore(ArtifactStore):
    """Default store — all operations are no-ops.  Zero dependencies."""

    def upload_file(self, local_path: Path, blob_key: str) -> str:
        return ""

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> list[str]:
        return []

    def download_file(self, blob_key: str, local_path: Path) -> None:
        pass

    def exists(self, blob_key: str) -> bool:
        return False


# ── Azure Blob ────────────────────────────────────────────────────────────────


class AzureBlobArtifactStore(ArtifactStore):
    """Uploads artifacts to Azure Blob Storage.

    Requires ``pip install 'agentic-mlops-yolo[azure]'`` (pulls in azure-storage-blob).

    Auth: DefaultAzureCredential — same as all other Azure integrations in this
    codebase (``az login`` or service-principal env vars).

    Initialisation logic:
      1. If ``config.storage.account_url`` is set → explicit mode: uses that URL +
         ``config.storage.container_name`` directly; no MLClient call.
      2. Otherwise → auto-resolve mode: calls ``ml_client.datastores.get(datastore_name)``
         to read ``.account_name`` / ``.container_name`` from the workspace datastore.

    All uploads are best-effort: a single file failure is logged as a warning and
    skipped; the workflow is never aborted due to a blob upload error.
    """

    def __init__(
        self,
        config: AzureMLConfig,
        client_factory: Any | None = None,
    ) -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "azure-storage-blob is not installed. "
                "Run: pip install 'agentic-mlops-yolo[azure]'"
            ) from exc

        from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
            DefaultAzureMLClientFactory,
        )

        cfg = config.storage
        self._prefix = cfg.artifact_prefix.rstrip("/")

        if cfg.account_url:
            account_url = cfg.account_url
            container = cfg.container_name
            if not container:
                raise ValueError(
                    "AzureMLStorageConfig.container_name is required when "
                    "account_url is set explicitly."
                )
        else:
            factory = client_factory or DefaultAzureMLClientFactory()
            ml_client = factory.create(config)
            ds = ml_client.datastores.get(cfg.datastore_name)
            account_url = f"https://{ds.account_name}.blob.core.windows.net"
            container = cfg.container_name or ds.container_name

        try:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            credential = DefaultAzureCredential()
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is not installed. " "Run: pip install 'agentic-mlops-yolo[azure]'"
            ) from exc

        self._service = BlobServiceClient(account_url=account_url, credential=credential)
        self._container = container
        self._account_url = account_url.rstrip("/")

    # ── public API ────────────────────────────────────────────────────────────

    def upload_file(self, local_path: Path, blob_key: str) -> str:
        full_key = f"{self._prefix}/{blob_key}" if self._prefix else blob_key
        try:
            blob_client = self._service.get_blob_client(container=self._container, blob=full_key)
            with open(local_path, "rb") as fh:
                blob_client.upload_blob(fh, overwrite=True)
            uri = f"{self._account_url}/{self._container}/{full_key}"
            logger.debug("uploaded artifact %s → %s", local_path, uri)
            return uri
        except Exception as exc:
            logger.warning("blob upload failed for %s: %s", local_path, exc)
            return ""

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> list[str]:
        if not local_dir.exists():
            return []
        uris: list[str] = []
        for file_path in sorted(local_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix in _SKIP_SUFFIXES:
                continue
            if any(part in _SKIP_DIRS for part in file_path.parts):
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{blob_prefix}/{relative}".replace("\\", "/")
            uri = self.upload_file(file_path, key)
            if uri:
                uris.append(uri)
        return uris

    def download_file(self, blob_key: str, local_path: Path) -> None:
        full_key = f"{self._prefix}/{blob_key}" if self._prefix else blob_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = self._service.get_blob_client(container=self._container, blob=full_key)
        with open(local_path, "wb") as fh:
            fh.write(blob_client.download_blob().readall())

    def exists(self, blob_key: str) -> bool:
        full_key = f"{self._prefix}/{blob_key}" if self._prefix else blob_key
        blob_client = self._service.get_blob_client(container=self._container, blob=full_key)
        return blob_client.exists()


# ── Fake (test double) ────────────────────────────────────────────────────────


class FakeArtifactStore(ArtifactStore):
    """In-memory test double.  Records all calls for assertion in unit tests."""

    def __init__(self) -> None:
        self.uploaded: list[tuple[Path | str, str]] = []
        self.downloaded: list[tuple[str, Path]] = []

    def upload_file(self, local_path: Path, blob_key: str) -> str:
        self.uploaded.append((local_path, blob_key))
        return f"https://fake.blob.core.windows.net/fake/{blob_key}"

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> list[str]:
        self.uploaded.append((local_dir, blob_prefix))
        return []

    def download_file(self, blob_key: str, local_path: Path) -> None:
        self.downloaded.append((blob_key, local_path))

    def exists(self, blob_key: str) -> bool:
        return False
