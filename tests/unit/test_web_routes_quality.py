"""Tests for the dataset quality dashboard route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from agentic_mlops.web.app import create_app  # noqa: E402


def _write_lineage(
    reg_dir: Path,
    dataset_name: str,
    version: int,
    quality_summary: dict | None = None,
    validation_status: str = "passed",
    classes: list | None = None,
) -> None:
    version_dir = reg_dir / dataset_name / "versions" / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    lineage: dict = {
        "dataset_name": dataset_name,
        "version": version,
        "hash": "abc123",
        "registered_at": "2026-01-01T10:00:00+00:00",
        "validation_status": validation_status,
        "label_qa_status": "passed",
        "classes": classes or ["scratch", "dent"],
    }
    if quality_summary is not None:
        lineage["quality_summary"] = quality_summary
    (version_dir / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")


@pytest.fixture()
def client(tmp_path):
    runs = tmp_path / "runs"
    registry = tmp_path / "models"
    datasets = tmp_path / "datasets"
    for d in (runs, registry, datasets):
        d.mkdir(parents=True, exist_ok=True)
    app = create_app(
        runs_dir=str(runs),
        registry_dir=str(registry),
        datasets_dir=str(datasets),
    )
    return TestClient(app), datasets


class TestDatasetQualityRoute:
    def test_404_for_unknown_dataset(self, client):
        tc, _reg = client
        resp = tc.get("/datasets/no_such_dataset/quality")
        assert resp.status_code == 404

    def test_returns_200_for_known_dataset(self, client):
        tc, reg = client
        _write_lineage(reg, "my_ds", 1)
        resp = tc.get("/datasets/my_ds/quality")
        assert resp.status_code == 200

    def test_page_contains_dataset_name(self, client):
        tc, reg = client
        _write_lineage(reg, "factory_defects", 1)
        resp = tc.get("/datasets/factory_defects/quality")
        assert "factory_defects" in resp.text

    def test_stat_cards_shown_with_quality_summary(self, client):
        tc, reg = client
        qs = {
            "num_images": 200,
            "num_labels": 195,
            "class_distribution": {"crack": 100, "dent": 95},
            "blocking_issues_count": 0,
            "label_issues_count": 2,
        }
        _write_lineage(reg, "ds", 1, quality_summary=qs)
        resp = tc.get("/datasets/ds/quality")
        assert resp.status_code == 200
        assert "200" in resp.text  # num_images
        assert "195" in resp.text  # num_labels
        assert "crack" in resp.text

    def test_version_history_table_present(self, client):
        tc, reg = client
        _write_lineage(reg, "ds", 1)
        _write_lineage(reg, "ds", 2)
        resp = tc.get("/datasets/ds/quality")
        assert resp.status_code == 200
        assert "v1" in resp.text
        assert "v2" in resp.text

    def test_class_distribution_bar_chart_present(self, client):
        tc, reg = client
        qs = {
            "num_images": 50,
            "num_labels": 50,
            "class_distribution": {"scratch": 30, "dent": 20},
            "blocking_issues_count": 0,
            "label_issues_count": 0,
        }
        _write_lineage(reg, "ds", 1, quality_summary=qs, classes=["scratch", "dent"])
        resp = tc.get("/datasets/ds/quality")
        assert "Class distribution" in resp.text
        assert "scratch" in resp.text
        assert "dent" in resp.text

    def test_back_link_present(self, client):
        tc, reg = client
        _write_lineage(reg, "my_dataset", 1)
        resp = tc.get("/datasets/my_dataset/quality")
        assert "/datasets/my_dataset" in resp.text

    def test_validation_status_badge_displayed(self, client):
        tc, reg = client
        _write_lineage(reg, "ds", 1, validation_status="passed")
        resp = tc.get("/datasets/ds/quality")
        assert "passed" in resp.text

    def test_missing_quality_summary_shows_dash(self, client):
        tc, reg = client
        _write_lineage(reg, "ds", 1)  # no quality_summary
        resp = tc.get("/datasets/ds/quality")
        assert resp.status_code == 200
        assert "—" in resp.text
