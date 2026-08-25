"""Annotation client connector — CVAT, Label Studio, and Fake implementations.

All implementations use only stdlib ``urllib.request`` for HTTP (same pattern
as ``WebhookNotificationClient``).  Never raises on transport errors — all
failures are returned as ``success=False`` outputs so the calling pipeline is
never blocked.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    pass

from agentic_mlops.contracts.annotation_client import (
    AnnotationBackend,
    AnnotationClientConfig,
    AnnotationExportFormat,
    AnnotationTask,
    ExportAnnotationsInput,
    ExportAnnotationsOutput,
    ListTasksOutput,
    UploadTaskInput,
    UploadTaskOutput,
)

logger = logging.getLogger(__name__)

# ── Format strings sent to each backend ───────────────────────────────────────

_CVAT_FORMAT: dict[AnnotationExportFormat, str] = {
    AnnotationExportFormat.YOLO_DETECTION: "YOLO 1.1",
    AnnotationExportFormat.COCO: "COCO 1.0",
    AnnotationExportFormat.VOC: "Pascal VOC 1.1",
}

_LABEL_STUDIO_FORMAT: dict[AnnotationExportFormat, str] = {
    AnnotationExportFormat.YOLO_DETECTION: "YOLO",
    AnnotationExportFormat.COCO: "COCO_JSON",
    AnnotationExportFormat.VOC: "VOC",
}


# ── Abstract base ─────────────────────────────────────────────────────────────


class AnnotationClientBase(ABC):
    """Common interface every annotation backend must implement."""

    @abstractmethod
    def upload_task(self, inp: UploadTaskInput) -> UploadTaskOutput:
        """Create a task in the annotation tool and upload images."""

    @abstractmethod
    def export_annotations(self, inp: ExportAnnotationsInput) -> ExportAnnotationsOutput:
        """Download completed annotations from a task to ``inp.output_dir``."""

    @abstractmethod
    def list_tasks(self) -> ListTasksOutput:
        """Return all tasks visible to the configured credentials."""

    @abstractmethod
    def get_task(self, task_id: str) -> AnnotationTask | None:
        """Return a single task by id, or ``None`` if not found."""


# ── Fake (in-memory test double) ──────────────────────────────────────────────


class FakeAnnotationClient(AnnotationClientBase):
    """In-memory test double.  All operations are deterministic and local."""

    def __init__(self) -> None:
        self.tasks: dict[str, AnnotationTask] = {}

    def upload_task(self, inp: UploadTaskInput) -> UploadTaskOutput:
        task_id = str(uuid.uuid4())[:8]
        task = AnnotationTask(
            task_id=task_id,
            name=inp.task_name,
            status="new",
            image_count=len(inp.image_paths),
        )
        self.tasks[task_id] = task
        return UploadTaskOutput(
            success=True,
            message=f"Task '{inp.task_name}' created (id={task_id})",
            task=task,
        )

    def export_annotations(self, inp: ExportAnnotationsInput) -> ExportAnnotationsOutput:
        if inp.task_id not in self.tasks:
            return ExportAnnotationsOutput(
                success=False,
                message=f"Task '{inp.task_id}' not found",
            )
        task = self.tasks[inp.task_id]
        inp.output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[Path] = []
        # Write one empty YOLO-format .txt per image slot
        for i in range(task.image_count):
            label_file = inp.output_dir / f"image_{i:04d}.txt"
            label_file.write_text("", encoding="utf-8")
            exported.append(label_file)
        return ExportAnnotationsOutput(
            success=True,
            message=f"Exported {len(exported)} label files",
            exported_files=exported,
            annotation_count=len(exported),
        )

    def list_tasks(self) -> ListTasksOutput:
        return ListTasksOutput(
            success=True,
            message=f"Found {len(self.tasks)} tasks",
            tasks=list(self.tasks.values()),
        )

    def get_task(self, task_id: str) -> AnnotationTask | None:
        return self.tasks.get(task_id)


# ── CVAT REST API v2 ──────────────────────────────────────────────────────────


class CVATAnnotationClient(AnnotationClientBase):
    """Thin client wrapping the CVAT REST API v2.

    Requires ``config.base_url`` pointing at the CVAT server root
    (e.g. ``http://localhost:8080``).  Authenticates via
    ``Authorization: Token <api_key>`` when ``api_key`` is set, otherwise
    falls back to HTTP Basic auth using ``username``/``password``.
    """

    def __init__(self, config: AnnotationClientConfig) -> None:
        self._cfg = config
        self._base = config.base_url.rstrip("/")
        self._timeout = config.timeout_seconds

    # ── internal helpers ──────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self._cfg.api_key:
            return {"Authorization": f"Token {self._cfg.api_key}"}
        if self._cfg.username and self._cfg.password:
            import base64

            creds = base64.b64encode(f"{self._cfg.username}:{self._cfg.password}".encode()).decode()
            return {"Authorization": f"Basic {creds}"}
        return {}

    def _get(self, path: str, params: dict | None = None) -> tuple[int, bytes]:
        url = f"{self._base}{path}"
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"
        req = Request(url, headers={**self._auth_headers(), "Accept": "application/json"})
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            logger.warning("CVAT GET %s failed: %s", path, exc)
            return 0, b""

    def _post_json(self, path: str, body: dict) -> tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self._base}{path}",
            data=data,
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            logger.warning("CVAT POST %s failed: %s", path, exc)
            return 0, b""

    def _post_multipart(self, path: str, files: list[tuple[str, bytes, str]]) -> tuple[int, bytes]:
        """POST multipart/form-data.  ``files`` is a list of (field, data, filename)."""
        boundary = uuid.uuid4().hex
        body_parts: list[bytes] = []
        for field, data, filename in files:
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            )
            body_parts.append(header.encode("utf-8") + data + b"\r\n")
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        req = Request(
            f"{self._base}{path}",
            data=body,
            headers={
                **self._auth_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            logger.warning("CVAT multipart POST %s failed: %s", path, exc)
            return 0, b""

    # ── public interface ──────────────────────────────────────────────────

    def upload_task(self, inp: UploadTaskInput) -> UploadTaskOutput:
        # 1. Create the task
        labels_payload = [{"name": lbl} for lbl in inp.labels]
        status, body = self._post_json(
            "/api/tasks",
            {"name": inp.task_name, "labels": labels_payload},
        )
        if status not in (200, 201):
            return UploadTaskOutput(
                success=False,
                message=f"Failed to create CVAT task (HTTP {status}): {body[:200]}",
            )
        try:
            task_data = json.loads(body)
        except json.JSONDecodeError as exc:
            return UploadTaskOutput(success=False, message=f"Invalid JSON from CVAT: {exc}")

        task_id = str(task_data.get("id", ""))
        task = AnnotationTask(
            task_id=task_id,
            name=inp.task_name,
            status="new",
            image_count=len(inp.image_paths),
            url=task_data.get("url"),
        )

        # 2. Upload images — best-effort; don't fail the task creation if upload fails
        files: list[tuple[str, bytes, str]] = []
        for img_path in inp.image_paths:
            try:
                files.append(("client_files", img_path.read_bytes(), img_path.name))
            except OSError as exc:
                logger.warning("Could not read image %s for CVAT upload: %s", img_path, exc)

        if files:
            up_status, up_body = self._post_multipart(f"/api/tasks/{task_id}/data", files)
            if up_status not in (200, 201, 202):
                logger.warning(
                    "CVAT image upload to task %s returned HTTP %s: %s",
                    task_id,
                    up_status,
                    up_body[:200],
                )

        return UploadTaskOutput(
            success=True,
            message=f"CVAT task '{inp.task_name}' created (id={task_id})",
            task=task,
        )

    def list_tasks(self) -> ListTasksOutput:
        status, body = self._get("/api/tasks", {"page_size": 100})
        if status != 200:
            return ListTasksOutput(
                success=False,
                message=f"Failed to list CVAT tasks (HTTP {status})",
            )
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            return ListTasksOutput(success=False, message=f"Invalid JSON from CVAT: {exc}")

        results = data.get("results", data) if isinstance(data, dict) else data
        tasks = [
            AnnotationTask(
                task_id=str(t.get("id", "")),
                name=t.get("name", ""),
                status=t.get("status", "new"),
                image_count=t.get("size", 0),
                annotation_count=t.get("jobs", {}).get("completed", 0)
                if isinstance(t.get("jobs"), dict)
                else 0,
                url=t.get("url"),
            )
            for t in results
        ]
        return ListTasksOutput(success=True, message=f"Found {len(tasks)} tasks", tasks=tasks)

    def get_task(self, task_id: str) -> AnnotationTask | None:
        status, body = self._get(f"/api/tasks/{task_id}")
        if status != 200:
            return None
        try:
            t = json.loads(body)
        except json.JSONDecodeError:
            return None
        return AnnotationTask(
            task_id=str(t.get("id", task_id)),
            name=t.get("name", ""),
            status=t.get("status", "new"),
            image_count=t.get("size", 0),
            url=t.get("url"),
        )

    def export_annotations(self, inp: ExportAnnotationsInput) -> ExportAnnotationsOutput:
        fmt = _CVAT_FORMAT.get(inp.format, "YOLO 1.1")
        from urllib.parse import urlencode

        params = urlencode({"format": fmt})
        url = f"{self._base}/api/tasks/{inp.task_id}/annotations?{params}"

        # CVAT may return 202 (export in progress) — retry up to 5 times
        zip_data: bytes | None = None
        for attempt in range(5):
            req = Request(url, headers={**self._auth_headers(), "Accept": "application/zip"})
            try:
                with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                    if resp.status == 200:
                        zip_data = resp.read()
                        break
                    if resp.status == 202:
                        logger.debug(
                            "CVAT export not ready (attempt %d/5), waiting 2s …", attempt + 1
                        )
                        time.sleep(2)
                        continue
                    return ExportAnnotationsOutput(
                        success=False,
                        message=f"CVAT export returned HTTP {resp.status}",
                    )
            except HTTPError as exc:
                if exc.code == 202:
                    time.sleep(2)
                    continue
                return ExportAnnotationsOutput(
                    success=False,
                    message=f"CVAT export HTTP error {exc.code}",
                )
            except URLError as exc:
                return ExportAnnotationsOutput(success=False, message=f"CVAT export failed: {exc}")

        if zip_data is None:
            return ExportAnnotationsOutput(
                success=False,
                message="CVAT export timed out after 5 retries",
            )

        inp.output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[Path] = []
        try:
            import io

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                for member in zf.namelist():
                    dest = inp.output_dir / Path(member).name
                    dest.write_bytes(zf.read(member))
                    exported.append(dest)
        except zipfile.BadZipFile as exc:
            return ExportAnnotationsOutput(
                success=False, message=f"CVAT returned invalid zip: {exc}"
            )

        return ExportAnnotationsOutput(
            success=True,
            message=f"Exported {len(exported)} files",
            exported_files=exported,
            annotation_count=len(exported),
        )


# ── Label Studio REST API ─────────────────────────────────────────────────────


class LabelStudioAnnotationClient(AnnotationClientBase):
    """Thin client wrapping the Label Studio REST API.

    Requires ``config.base_url`` pointing at the Label Studio server
    (e.g. ``http://localhost:8080``) and ``config.api_key`` for Bearer-token
    auth (``Authorization: Token <api_key>``).
    """

    def __init__(self, config: AnnotationClientConfig) -> None:
        self._cfg = config
        self._base = config.base_url.rstrip("/")
        self._timeout = config.timeout_seconds

    # ── internal helpers ──────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self._cfg.api_key:
            return {"Authorization": f"Token {self._cfg.api_key}"}
        return {}

    def _get(self, path: str, params: dict | None = None) -> tuple[int, bytes]:
        url = f"{self._base}{path}"
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"
        req = Request(url, headers={**self._auth_headers(), "Accept": "application/json"})
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            logger.warning("Label Studio GET %s failed: %s", path, exc)
            return 0, b""

    def _post_json(self, path: str, body: dict) -> tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self._base}{path}",
            data=data,
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            logger.warning("Label Studio POST %s failed: %s", path, exc)
            return 0, b""

    def _find_or_create_project(self, name: str) -> tuple[str | None, str]:
        """Return (project_id, error_message).  Empty error means success."""
        status, body = self._get("/api/projects", {"title": name})
        if status == 200:
            try:
                data = json.loads(body)
                results = data.get("results", data) if isinstance(data, dict) else data
                for proj in results:
                    if proj.get("title") == name:
                        return str(proj["id"]), ""
            except (json.JSONDecodeError, KeyError):
                pass  # fall through to create

        # Create project
        status, body = self._post_json(
            "/api/projects", {"title": name, "label_config": "<View></View>"}
        )
        if status not in (200, 201):
            return None, f"Failed to create Label Studio project (HTTP {status}): {body[:200]}"
        try:
            proj = json.loads(body)
            return str(proj["id"]), ""
        except (json.JSONDecodeError, KeyError) as exc:
            return None, f"Invalid JSON from Label Studio: {exc}"

    # ── public interface ──────────────────────────────────────────────────

    def upload_task(self, inp: UploadTaskInput) -> UploadTaskOutput:
        project_id, err = self._find_or_create_project(self._cfg.project_name or inp.task_name)
        if not project_id:
            return UploadTaskOutput(success=False, message=err)

        # Build JSON tasks list — Label Studio expects {"data": {"image": "<path>"}}
        ls_tasks = [{"data": {"image": str(p)}} for p in inp.image_paths]
        status, body = self._post_json(f"/api/projects/{project_id}/import", ls_tasks)
        if status not in (200, 201):
            return UploadTaskOutput(
                success=False,
                message=f"Label Studio import failed (HTTP {status}): {body[:200]}",
            )

        task = AnnotationTask(
            task_id=project_id,
            name=inp.task_name,
            status="new",
            image_count=len(inp.image_paths),
        )
        return UploadTaskOutput(
            success=True,
            message=(
                f"Imported {len(inp.image_paths)} images into " f"Label Studio project {project_id}"
            ),
            task=task,
        )

    def list_tasks(self) -> ListTasksOutput:
        status, body = self._get("/api/projects")
        if status != 200:
            return ListTasksOutput(
                success=False,
                message=f"Failed to list Label Studio projects (HTTP {status})",
            )
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            return ListTasksOutput(success=False, message=f"Invalid JSON from Label Studio: {exc}")

        results = data.get("results", data) if isinstance(data, dict) else data
        tasks = [
            AnnotationTask(
                task_id=str(p.get("id", "")),
                name=p.get("title", ""),
                status="in_progress" if p.get("num_tasks_with_annotations", 0) > 0 else "new",
                image_count=p.get("task_number", 0),
                annotation_count=p.get("num_tasks_with_annotations", 0),
            )
            for p in results
        ]
        return ListTasksOutput(success=True, message=f"Found {len(tasks)} projects", tasks=tasks)

    def get_task(self, task_id: str) -> AnnotationTask | None:
        status, body = self._get(f"/api/projects/{task_id}")
        if status != 200:
            return None
        try:
            p = json.loads(body)
        except json.JSONDecodeError:
            return None
        return AnnotationTask(
            task_id=str(p.get("id", task_id)),
            name=p.get("title", ""),
            status="in_progress" if p.get("num_tasks_with_annotations", 0) > 0 else "new",
            image_count=p.get("task_number", 0),
            annotation_count=p.get("num_tasks_with_annotations", 0),
        )

    def export_annotations(self, inp: ExportAnnotationsInput) -> ExportAnnotationsOutput:
        fmt = _LABEL_STUDIO_FORMAT.get(inp.format, "YOLO")
        from urllib.parse import urlencode

        params = urlencode({"exportType": fmt})
        url = f"{self._base}/api/projects/{inp.task_id}/export?{params}"

        req = Request(url, headers={**self._auth_headers()})
        try:
            with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                if resp.status != 200:
                    return ExportAnnotationsOutput(
                        success=False,
                        message=f"Label Studio export returned HTTP {resp.status}",
                    )
                zip_data = resp.read()
        except HTTPError as exc:
            return ExportAnnotationsOutput(
                success=False, message=f"Label Studio export HTTP error {exc.code}"
            )
        except URLError as exc:
            return ExportAnnotationsOutput(
                success=False, message=f"Label Studio export failed: {exc}"
            )

        inp.output_dir.mkdir(parents=True, exist_ok=True)
        exported: list[Path] = []
        try:
            import io

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                for member in zf.namelist():
                    dest = inp.output_dir / Path(member).name
                    dest.write_bytes(zf.read(member))
                    exported.append(dest)
        except zipfile.BadZipFile as exc:
            return ExportAnnotationsOutput(
                success=False, message=f"Label Studio returned invalid zip: {exc}"
            )

        return ExportAnnotationsOutput(
            success=True,
            message=f"Exported {len(exported)} files",
            exported_files=exported,
            annotation_count=len(exported),
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_annotation_client(config: AnnotationClientConfig) -> AnnotationClientBase:
    """Return the appropriate annotation client for *config.backend*."""
    if config.backend == AnnotationBackend.FAKE:
        return FakeAnnotationClient()
    if config.backend == AnnotationBackend.CVAT:
        return CVATAnnotationClient(config)
    if config.backend == AnnotationBackend.LABEL_STUDIO:
        return LabelStudioAnnotationClient(config)
    raise ValueError(f"Unknown annotation backend: {config.backend!r}")
