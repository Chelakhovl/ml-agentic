"""Unit tests for BaselineResolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.tools.baseline_resolver import BaselineResolver, FakeBaselineResolver


def _make_version(registry_dir: Path, model: str, version: int, map50: float, **kwargs) -> None:
    ver_dir = registry_dir / model / "versions" / str(version)
    ver_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "map50": map50,
        "map50_95": kwargs.get("map50_95", 0.6),
        "precision": kwargs.get("precision", 0.8),
        "recall": kwargs.get("recall", 0.75),
    }
    (ver_dir / "lineage.json").write_text(json.dumps(lineage))


class TestBaselineResolver:
    def test_returns_none_for_missing_dir(self, tmp_path):
        resolver = BaselineResolver()
        assert resolver.resolve(tmp_path / "nonexistent") is None

    def test_returns_none_for_empty_registry(self, tmp_path):
        (tmp_path / "registry").mkdir()
        assert BaselineResolver().resolve(tmp_path / "registry") is None

    def test_picks_highest_map50(self, tmp_path):
        reg = tmp_path / "registry"
        _make_version(reg, "model-a", 1, map50=0.80)
        _make_version(reg, "model-a", 2, map50=0.91)
        _make_version(reg, "model-b", 1, map50=0.75)

        result = BaselineResolver().resolve(reg)
        assert result is not None
        assert result.model_name == "model-a"
        assert result.version == 2
        assert result.map50 == pytest.approx(0.91)

    def test_exclude_model_name(self, tmp_path):
        reg = tmp_path / "registry"
        _make_version(reg, "current-model", 1, map50=0.95)
        _make_version(reg, "old-model", 1, map50=0.80)

        result = BaselineResolver().resolve(reg, exclude_model_name="current-model")
        assert result is not None
        assert result.model_name == "old-model"

    def test_exclude_specific_version_keeps_others(self, tmp_path):
        reg = tmp_path / "registry"
        _make_version(reg, "model-a", 1, map50=0.90)
        _make_version(reg, "model-a", 2, map50=0.95)

        result = BaselineResolver().resolve(reg, exclude_model_name="model-a", exclude_version=2)
        assert result is not None
        assert result.version == 1

    def test_skips_versions_without_map50(self, tmp_path):
        reg = tmp_path / "registry"
        ver_dir = reg / "no-metrics-model" / "versions" / "1"
        ver_dir.mkdir(parents=True)
        (ver_dir / "lineage.json").write_text(json.dumps({"some": "field"}))

        assert BaselineResolver().resolve(reg) is None

    def test_write_synthetic_report(self, tmp_path):
        reg = tmp_path / "registry"
        _make_version(reg, "model-a", 1, map50=0.85, map50_95=0.65, precision=0.88, recall=0.83)

        resolver = BaselineResolver()
        result = resolver.resolve(reg)
        out = resolver.write_synthetic_report(result, tmp_path / "out")

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["auto_resolved_baseline"] is True
        assert data["baseline_model_name"] == "model-a"
        assert data["metrics"]["map50"] == pytest.approx(0.85)
        assert data["metrics"]["precision"] == pytest.approx(0.88)

    def test_fake_resolver_records_calls(self, tmp_path):
        fake = FakeBaselineResolver()
        result = fake.resolve(tmp_path / "registry", exclude_model_name="x")
        assert result is not None
        assert result.model_name == "fake-model"
        assert len(fake.resolve_calls) == 1
        assert fake.resolve_calls[0]["exclude_model_name"] == "x"
