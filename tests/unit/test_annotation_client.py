"""Tests for the annotation client connector (contracts + integrations)."""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.contracts.annotation_client import (
    AnnotationBackend,
    AnnotationClientConfig,
    AnnotationExportFormat,
    ExportAnnotationsInput,
    UploadTaskInput,
)
from agentic_mlops.integrations.annotation_client import (
    CVATAnnotationClient,
    FakeAnnotationClient,
    LabelStudioAnnotationClient,
    create_annotation_client,
)


def _make_images(tmp_path: Path, count: int = 3) -> list[Path]:
    """Write *count* minimal dummy image files and return their paths."""
    images = []
    for i in range(count):
        p = tmp_path / f"image_{i:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG stub
        images.append(p)
    return images


# ── FakeAnnotationClient ──────────────────────────────────────────────────────


class TestFakeAnnotationClient:
    def test_list_tasks_empty_initially(self):
        client = FakeAnnotationClient()
        out = client.list_tasks()
        assert out.success
        assert out.tasks == []

    def test_upload_task_creates_task(self, tmp_path: Path):
        client = FakeAnnotationClient()
        images = _make_images(tmp_path, 3)
        inp = UploadTaskInput(
            task_name="review_batch_1",
            image_paths=images,
            labels=["scratch", "dent"],
        )
        out = client.upload_task(inp)
        assert out.success
        assert out.task is not None
        assert out.task.name == "review_batch_1"
        assert out.task.image_count == 3

        # Task now appears in list_tasks
        list_out = client.list_tasks()
        assert len(list_out.tasks) == 1
        assert list_out.tasks[0].task_id == out.task.task_id

    def test_export_creates_label_files(self, tmp_path: Path):
        client = FakeAnnotationClient()
        images = _make_images(tmp_path, 3)
        upload_out = client.upload_task(
            UploadTaskInput(task_name="batch", image_paths=images, labels=["crack"])
        )
        assert upload_out.task is not None

        export_dir = tmp_path / "exports"
        export_out = client.export_annotations(
            ExportAnnotationsInput(
                task_id=upload_out.task.task_id,
                output_dir=export_dir,
                format=AnnotationExportFormat.YOLO_DETECTION,
            )
        )
        assert export_out.success
        assert len(export_out.exported_files) == 3
        for f in export_out.exported_files:
            assert f.suffix == ".txt"
            assert f.exists()

    def test_get_task_returns_none_for_unknown(self):
        client = FakeAnnotationClient()
        assert client.get_task("nonexistent") is None

    def test_get_task_returns_task_after_upload(self, tmp_path: Path):
        client = FakeAnnotationClient()
        images = _make_images(tmp_path, 1)
        out = client.upload_task(
            UploadTaskInput(task_name="t1", image_paths=images, labels=["dent"])
        )
        assert out.task is not None
        found = client.get_task(out.task.task_id)
        assert found is not None
        assert found.task_id == out.task.task_id

    def test_export_returns_failure_for_unknown_task(self, tmp_path: Path):
        client = FakeAnnotationClient()
        out = client.export_annotations(
            ExportAnnotationsInput(task_id="bad_id", output_dir=tmp_path / "out")
        )
        assert not out.success

    def test_upload_multiple_tasks(self, tmp_path: Path):
        client = FakeAnnotationClient()
        images = _make_images(tmp_path, 2)
        client.upload_task(UploadTaskInput(task_name="a", image_paths=images, labels=["x"]))
        client.upload_task(UploadTaskInput(task_name="b", image_paths=images, labels=["y"]))
        list_out = client.list_tasks()
        assert len(list_out.tasks) == 2
        names = {t.name for t in list_out.tasks}
        assert names == {"a", "b"}


# ── Factory ───────────────────────────────────────────────────────────────────


class TestCreateAnnotationClient:
    def test_create_annotation_client_fake(self):
        cfg = AnnotationClientConfig(backend=AnnotationBackend.FAKE)
        client = create_annotation_client(cfg)
        assert isinstance(client, FakeAnnotationClient)

    def test_create_annotation_client_cvat(self):
        cfg = AnnotationClientConfig(
            backend=AnnotationBackend.CVAT,
            base_url="http://localhost:8080",
            api_key="testkey",
        )
        client = create_annotation_client(cfg)
        assert isinstance(client, CVATAnnotationClient)

    def test_create_annotation_client_label_studio(self):
        cfg = AnnotationClientConfig(
            backend=AnnotationBackend.LABEL_STUDIO,
            base_url="http://localhost:8081",
            api_key="ls_key",
        )
        client = create_annotation_client(cfg)
        assert isinstance(client, LabelStudioAnnotationClient)

    def test_unknown_backend_raises(self):
        import pytest

        cfg = AnnotationClientConfig.__new__(AnnotationClientConfig)
        # Bypass Pydantic validation to inject a bad backend string
        object.__setattr__(cfg, "backend", "unknown_backend")
        object.__setattr__(cfg, "base_url", "")
        object.__setattr__(cfg, "api_key", None)
        object.__setattr__(cfg, "username", None)
        object.__setattr__(cfg, "password", None)
        object.__setattr__(cfg, "project_name", "x")
        object.__setattr__(cfg, "timeout_seconds", 30)
        with pytest.raises(ValueError, match="Unknown annotation backend"):
            create_annotation_client(cfg)
