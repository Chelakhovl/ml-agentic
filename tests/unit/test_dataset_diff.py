"""Unit tests for dataset diff reader and /datasets/{name}/diff route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentic_mlops.web import reader
from agentic_mlops.web.app import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_version(
    registry_dir: Path,
    dataset_name: str,
    version: int,
    num_images: int = 100,
    num_labels: int = 90,
    classes: list[str] | None = None,
    validation_status: str = "passed",
    blocking_issues: int = 0,
    label_issues: int = 0,
    source_batches: list[str] | None = None,
    content_hash: str | None = None,
) -> None:
    ver_dir = registry_dir / dataset_name / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "registered_at": f"2026-08-{20 + version:02d}T10:00:00",
        "validation_status": validation_status,
        "classes": classes or ["scratch", "dent"],
        "content_hash": content_hash or f"hash_{version}",
        "source_batches": source_batches or [f"batch_{version}"],
        "quality_summary": {
            "num_images": num_images,
            "num_labels": num_labels,
            "blocking_issues_count": blocking_issues,
            "label_issues_count": label_issues,
            "class_distribution": {
                c: num_labels // len(classes or ["scratch", "dent"])
                for c in (classes or ["scratch", "dent"])
            },
        },
    }
    (ver_dir / "lineage.json").write_text(json.dumps(lineage))
    latest = {
        "dataset_name": dataset_name,
        "version": version,
        "registered_at": lineage["registered_at"],
    }
    (registry_dir / dataset_name / "latest.json").write_text(json.dumps(latest))


@pytest.fixture
def reg(tmp_path):
    _write_version(tmp_path, "ds", 1, num_images=100, classes=["scratch", "dent"])
    _write_version(tmp_path, "ds", 2, num_images=150, classes=["scratch", "dent", "crack"])
    return tmp_path


@pytest.fixture
def client(tmp_path, reg):
    app = create_app(
        runs_dir=str(tmp_path / "runs"),
        registry_dir=str(tmp_path / "registry"),
        datasets_dir=str(reg),
    )
    return TestClient(app)


# ── Reader unit tests ─────────────────────────────────────────────────────────


class TestGetDatasetDiff:
    def test_returns_none_for_missing_version(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 99)
        assert diff is None

    def test_returns_none_for_unknown_dataset(self, reg):
        diff = reader.get_dataset_diff(reg, "nope", 1, 2)
        assert diff is None

    def test_diff_has_correct_keys(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        assert diff is not None
        assert diff["v1"] == 1
        assert diff["v2"] == 2
        assert "fields" in diff
        assert "classes_added" in diff
        assert "classes_removed" in diff

    def test_numeric_delta_computed(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        ni = diff["fields"]["num_images"]
        assert ni["a"] == 100
        assert ni["b"] == 150
        assert ni["delta"] == 50
        assert ni["changed"] is True

    def test_unchanged_field_not_flagged(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        vs = diff["fields"]["validation_status"]
        assert vs["changed"] is False

    def test_classes_added_detected(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        assert "crack" in diff["classes_added"]
        assert diff["classes_removed"] == []

    def test_classes_removed_detected(self, tmp_path):
        _write_version(tmp_path, "ex", 1, classes=["scratch", "dent", "crack"])
        _write_version(tmp_path, "ex", 2, classes=["scratch", "dent"])
        diff = reader.get_dataset_diff(tmp_path, "ex", 1, 2)
        assert "crack" in diff["classes_removed"]
        assert diff["classes_added"] == []

    def test_same_version_shows_no_changes(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 1)
        assert diff is not None
        for key, f in diff["fields"].items():
            assert f["changed"] is False, f"{key} unexpectedly changed"

    def test_class_distributions_present(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        assert isinstance(diff["class_dist_a"], dict)
        assert isinstance(diff["class_dist_b"], dict)

    def test_source_batches_present(self, reg):
        diff = reader.get_dataset_diff(reg, "ds", 1, 2)
        assert diff["v1_source_batches"] == ["batch_1"]
        assert diff["v2_source_batches"] == ["batch_2"]


# ── Route tests ───────────────────────────────────────────────────────────────


class TestDatasetDiffRoute:
    def test_returns_200_with_valid_versions(self, client):
        r = client.get("/datasets/ds/diff?v1=1&v2=2")
        assert r.status_code == 200

    def test_page_contains_version_numbers(self, client):
        r = client.get("/datasets/ds/diff?v1=1&v2=2")
        assert "v1" in r.text or "Version 1" in r.text
        assert "v2" in r.text or "Version 2" in r.text

    def test_404_for_unknown_dataset(self, client):
        r = client.get("/datasets/nope/diff?v1=1&v2=2")
        assert r.status_code == 404

    def test_404_for_missing_version(self, client):
        r = client.get("/datasets/ds/diff?v1=1&v2=99")
        assert r.status_code == 404

    def test_default_versions_chosen_when_omitted(self, client):
        r = client.get("/datasets/ds/diff")
        assert r.status_code == 200

    def test_class_change_shown(self, client):
        r = client.get("/datasets/ds/diff?v1=1&v2=2")
        assert "crack" in r.text

    def test_selector_form_present(self, client):
        r = client.get("/datasets/ds/diff?v1=1&v2=2")
        assert "<select" in r.text
        assert "Compare" in r.text

    def test_quality_page_has_diff_link(self, client):
        r = client.get("/datasets/ds/quality")
        assert r.status_code == 200
        assert "/diff?" in r.text
