"""Tests for `agentic-mlops init` scaffolding command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentic_mlops.cli.main import app

runner = CliRunner()

EXPECTED_FILES = [
    "training.yaml",
    "promotion_policy.yaml",
    "orchestrator.yaml",
    "mlflow.yaml",
    "azure_ml.yaml",
]


class TestInitCommand:
    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "new_configs"
        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0
        assert out.is_dir()

    def test_writes_all_five_configs(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0
        for name in EXPECTED_FILES:
            assert (out / name).exists(), f"expected {name} to be written"

    def test_files_are_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        runner.invoke(app, ["init", str(out)])
        for name in EXPECTED_FILES:
            f = out / name
            if f.exists():
                assert f.stat().st_size > 0, f"{name} must not be empty"

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        out.mkdir()
        sentinel = out / "training.yaml"
        sentinel.write_text("original", encoding="utf-8")

        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0
        # existing file not overwritten
        assert sentinel.read_text(encoding="utf-8") == "original"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        out.mkdir()
        sentinel = out / "training.yaml"
        sentinel.write_text("original", encoding="utf-8")

        result = runner.invoke(app, ["init", str(out), "--force"])
        assert result.exit_code == 0
        assert sentinel.read_text(encoding="utf-8") != "original"

    def test_output_mentions_written_count(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0
        assert "Scaffolded" in result.output

    def test_default_output_dir_is_configs(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / "configs").is_dir()

    def test_creates_standard_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "configs"
        runner.invoke(app, ["init", str(out)])
        assert (tmp_path / "runs").is_dir()
        assert (tmp_path / "outputs" / "model_registry").is_dir()
        assert (tmp_path / "outputs" / "dataset_registry").is_dir()

    def test_standard_dirs_not_recreated_when_exist(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        runs.mkdir()
        sentinel = runs / "do_not_delete.txt"
        sentinel.write_text("keep", encoding="utf-8")

        runner.invoke(app, ["init", str(tmp_path / "configs")])
        assert sentinel.read_text(encoding="utf-8") == "keep"
