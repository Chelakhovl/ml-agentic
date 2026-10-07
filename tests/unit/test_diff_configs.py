"""Unit tests for the diff-configs CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_mlops.cli.main import app

runner = CliRunner()


def _write_yaml(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestDiffConfigsIdentical:
    def test_identical_files_exits_0(self, tmp_path: Path) -> None:
        cfg = _write_yaml(tmp_path / "a.yaml", "epochs: 5\nbatch: 16\n")
        result = runner.invoke(app, ["diff-configs", cfg, cfg])
        assert result.exit_code == 0, result.output

    def test_identical_files_text_says_identical(self, tmp_path: Path) -> None:
        cfg = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        result = runner.invoke(app, ["diff-configs", cfg, cfg])
        assert "identical" in result.output.lower()

    def test_identical_files_json_has_identical_true(self, tmp_path: Path) -> None:
        cfg = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        result = runner.invoke(app, ["diff-configs", cfg, cfg, "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["identical"] is True
        assert data["added"] == {}
        assert data["removed"] == {}
        assert data["changed"] == {}


class TestDiffConfigsDifferences:
    def test_changed_key_exits_1(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        b = _write_yaml(tmp_path / "b.yaml", "epochs: 10\n")
        result = runner.invoke(app, ["diff-configs", a, b])
        assert result.exit_code == 1

    def test_added_key_shown(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        b = _write_yaml(tmp_path / "b.yaml", "epochs: 5\nbatch: 16\n")
        result = runner.invoke(app, ["diff-configs", a, b])
        assert result.exit_code == 1
        assert "batch" in result.output

    def test_removed_key_shown(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\nbatch: 16\n")
        b = _write_yaml(tmp_path / "b.yaml", "epochs: 5\n")
        result = runner.invoke(app, ["diff-configs", a, b])
        assert result.exit_code == 1
        assert "batch" in result.output

    def test_json_format_diff(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\nbatch: 16\n")
        b = _write_yaml(tmp_path / "b.yaml", "epochs: 10\nbatch: 16\n")
        result = runner.invoke(app, ["diff-configs", a, b, "--format", "json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["identical"] is False
        assert "epochs" in data["changed"]
        assert data["changed"]["epochs"]["from"] == 5
        assert data["changed"]["epochs"]["to"] == 10

    def test_nested_key_diff_shown_as_dotted_path(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "training:\n  epochs: 5\n")
        b = _write_yaml(tmp_path / "b.yaml", "training:\n  epochs: 20\n")
        result = runner.invoke(app, ["diff-configs", a, b, "--format", "json"])
        data = json.loads(result.output)
        assert "training.epochs" in data["changed"]

    def test_list_index_diff_shown_in_json(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "steps:\n  - training\n  - evaluation\n")
        b = _write_yaml(tmp_path / "b.yaml", "steps:\n  - training\n  - approval\n")
        result = runner.invoke(app, ["diff-configs", a, b, "--format", "json"])
        data = json.loads(result.output)
        assert any("[1]" in k for k in data["changed"])


class TestDiffConfigsErrors:
    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        cfg = str(tmp_path / "nonexistent.yaml")
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        result = runner.invoke(app, ["diff-configs", a, cfg])
        assert result.exit_code == 2

    def test_empty_yaml_treated_as_empty_dict(self, tmp_path: Path) -> None:
        a = _write_yaml(tmp_path / "a.yaml", "epochs: 5\n")
        b = _write_yaml(tmp_path / "b.yaml", "")
        result = runner.invoke(app, ["diff-configs", a, b, "--format", "json"])
        data = json.loads(result.output)
        assert "epochs" in data["removed"]
