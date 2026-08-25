"""Unit tests for HardSampleIngestionAgent, HardSampleIngester, and CLI command.

Coverage:
  1.  Missing manifest path -> failed output
  2.  Missing images_source_dir -> failed output
  3.  All hard samples not found -> failed output + warnings
  4.  Happy path: found images staged + DataIntakeAgent runs successfully
  5.  Partial match: some found, some not found -> success with warnings
  6.  image_id as absolute path -> resolved directly
  7.  image_id as stem only -> resolved via extension fallback
  8.  CLI: ingest-hard-samples exits 0 on success
  9.  CLI: ingest-hard-samples exits 1 when manifest missing
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agentic_mlops.agents.hard_sample_ingestion import HardSampleIngestionAgent
from agentic_mlops.contracts.hard_sample_ingestion import HardSampleIngestionInput
from agentic_mlops.tools.hard_sample_ingester import HardSampleIngester, _resolve_image

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_manifest(path: Path, image_ids: list[str]) -> Path:
    manifest = [{"image_id": iid, "reason": "low_confidence"} for iid in image_ids]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _make_image(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p)


# ── HardSampleIngester unit tests ─────────────────────────────────────────────


def test_resolve_image_by_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "frame001.jpg")
    result = _resolve_image(src, "frame001.jpg")
    assert result is not None and result.name == "frame001.jpg"


def test_resolve_image_by_stem_fallback(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "frame002.png")
    result = _resolve_image(src, "frame002")  # no extension in image_id
    assert result is not None and result.name == "frame002.png"


def test_resolve_image_absolute_path(tmp_path: Path) -> None:
    img = tmp_path / "absolute.jpg"
    _make_image(img)
    result = _resolve_image(tmp_path / "other", str(img))
    assert result == img


def test_resolve_image_not_found(tmp_path: Path) -> None:
    assert _resolve_image(tmp_path, "ghost.jpg") is None


def test_ingester_stage_copies_found_skips_missing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "good.jpg")
    manifest = _make_manifest(tmp_path / "manifest.json", ["good.jpg", "missing.jpg"])
    staging = tmp_path / "staging"

    ingester = HardSampleIngester()
    total, found, not_found, warnings = ingester.stage(manifest, src, staging)

    assert total == 2
    assert found == 1
    assert not_found == 1
    assert (staging / "good.jpg").exists()
    assert any("missing.jpg" in w for w in warnings)


# ── Agent tests ───────────────────────────────────────────────────────────────


def test_missing_manifest_returns_failed(tmp_path: Path) -> None:
    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(tmp_path / "no_such_manifest.json"),
            images_source_dir=str(tmp_path),
            dataset_name="test",
        )
    )
    assert result.success is False
    assert "not found" in result.message.lower()


def test_missing_source_dir_returns_failed(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path / "manifest.json", ["img.jpg"])
    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(manifest),
            images_source_dir=str(tmp_path / "no_such_dir"),
            dataset_name="test",
        )
    )
    assert result.success is False
    assert "not found" in result.message.lower()


def test_no_images_found_returns_failed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    manifest = _make_manifest(tmp_path / "manifest.json", ["ghost1.jpg", "ghost2.jpg"])
    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(manifest),
            images_source_dir=str(src),
            dataset_name="test",
        )
    )
    assert result.success is False
    assert result.num_images_found == 0
    assert result.num_images_not_found == 2


def test_happy_path_stages_and_runs_intake(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "hard1.jpg")
    _make_image(src / "hard2.jpg")
    manifest = _make_manifest(tmp_path / "manifest.json", ["hard1.jpg", "hard2.jpg"])

    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(manifest),
            images_source_dir=str(src),
            dataset_name="hard_retraining",
        )
    )

    assert result.num_hard_samples_in_manifest == 2
    assert result.num_images_found == 2
    assert result.num_images_not_found == 0
    assert result.staging_dir is not None
    assert Path(result.staging_dir).is_dir()
    # DataIntake manifest should have been written
    assert result.intake_manifest_path is not None


def test_partial_match_success_with_warnings(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "found.jpg")
    manifest = _make_manifest(tmp_path / "manifest.json", ["found.jpg", "missing.jpg"])

    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(manifest),
            images_source_dir=str(src),
            dataset_name="partial",
        )
    )

    assert result.num_images_found == 1
    assert result.num_images_not_found == 1
    assert any("missing.jpg" in w for w in result.warnings)


def test_default_source_label_is_hard_sample_mining(tmp_path: Path) -> None:
    """When source is None, DataIntakeAgent should receive 'hard_sample_mining'."""
    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "img.jpg")
    manifest = _make_manifest(tmp_path / "manifest.json", ["img.jpg"])

    agent = HardSampleIngestionAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(manifest),
            images_source_dir=str(src),
            dataset_name="test",
            source=None,
        )
    )

    # The agent succeeds; the source label is "hard_sample_mining" — verified
    # indirectly by checking the intake manifest contains the source field.
    assert result.intake_manifest_path is not None
    intake_data = json.loads(Path(result.intake_manifest_path).read_text())
    assert intake_data.get("source") == "hard_sample_mining"


# ── CLI tests ─────────────────────────────────────────────────────────────────


def test_cli_ingest_hard_samples_exits_0(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    _make_image(src / "img.jpg")
    manifest = _make_manifest(tmp_path / "manifest.json", ["img.jpg"])

    result = CliRunner().invoke(
        app,
        [
            "ingest-hard-samples",
            str(manifest),
            "--images-source-dir",
            str(src),
            "--dataset-name",
            "cli_test",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_ingest_hard_samples_exits_1_on_missing_manifest(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "ingest-hard-samples",
            str(tmp_path / "no_manifest.json"),
            "--images-source-dir",
            str(tmp_path),
            "--dataset-name",
            "test",
        ],
    )
    assert result.exit_code == 1
