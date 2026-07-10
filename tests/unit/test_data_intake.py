"""Unit tests for the Data Intake Agent, scanner tool, and CLI command.

Coverage matrix:
    1.  raw_data_path not found -> status failed
    2.  Empty directory -> status failed
    3.  Fewer files than min_files -> status failed
    4.  Clean dataset + source given -> status passed
    5.  Missing source -> status needs_human_source_approval
    6.  Corrupted image detected and listed
    7.  Corrupted ratio over threshold -> status failed
    8.  Duplicate image detected (same content, different filename)
    9.  Duplicate ratio over threshold -> status needs_human_source_approval
   10.  Unexpected format file flagged, triggers needs_human_source_approval
   11.  Pillow unavailable -> pillow_available False, falls back to size-only check
   12.  DataIntakeAgent writes dataset_manifest.json / .md
   13.  DataIntakeAgent logs to MLflow when enabled
   14.  CLI: data-intake command exists and passes on a clean dataset (exit 0)
   15.  CLI: data-intake exits 1 on missing raw_data_path
   16.  CLI: data-intake exits 0 (not 1) when only needs_human_source_approval
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agentic_mlops.agents.data_intake import DataIntakeAgent
from agentic_mlops.contracts.data_intake import DataIntakeInput, DataIntakeStatus
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.data_intake_scanner import DataIntakeScanner

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_jpeg(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


# ── 1-3. Structural failures ──────────────────────────────────────────────────


def test_raw_path_not_found_fails(tmp_path: Path) -> None:
    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(tmp_path / "nonexistent"), dataset_name="ds")
    )

    assert result.success is False
    assert result.status == DataIntakeStatus.FAILED
    assert "not found" in result.message.lower()


def test_empty_directory_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    result = DataIntakeScanner().scan(DataIntakeInput(raw_data_path=str(raw), dataset_name="ds"))

    assert result.success is False
    assert result.status == DataIntakeStatus.FAILED


def test_min_files_threshold_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")

    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", min_files=5, source="src")
    )

    assert result.success is False
    assert result.status == DataIntakeStatus.FAILED
    assert any("expected at least" in e for e in result.errors)


# ── 4-5. Status determination ─────────────────────────────────────────────────


def test_clean_dataset_with_source_passes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _make_jpeg(raw / f"img_{i}.jpg", color=(i * 10, 0, 0))

    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="camera_batch_1")
    )

    assert result.success is True
    assert result.status == DataIntakeStatus.PASSED
    assert result.num_files == 3
    assert result.valid_images == 3
    assert result.corrupted_images == []
    assert result.duplicate_groups == []


def test_missing_source_needs_approval(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")

    result = DataIntakeScanner().scan(DataIntakeInput(raw_data_path=str(raw), dataset_name="ds"))

    assert result.success is True
    assert result.status == DataIntakeStatus.NEEDS_HUMAN_SOURCE_APPROVAL
    assert any("source" in w.lower() for w in result.warnings)


# ── 6-7. Corruption ────────────────────────────────────────────────────────────


def test_corrupted_image_detected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "good.jpg")
    (raw / "bad.jpg").write_bytes(b"this is not a valid jpeg")

    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s")
    )

    # 1/2 corrupted = 50% > default 5% threshold -> failed, but still reported
    assert "bad.jpg" in result.corrupted_images
    assert result.valid_images == 1


def test_corrupted_ratio_exceeds_threshold_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "good.jpg")
    (raw / "bad.jpg").write_bytes(b"garbage")

    result = DataIntakeScanner().scan(
        DataIntakeInput(
            raw_data_path=str(raw), dataset_name="ds", source="s",
            corrupted_ratio_threshold=0.1,
        )
    )

    assert result.success is False
    assert result.status == DataIntakeStatus.FAILED
    assert any("corrupted" in e.lower() for e in result.errors)


def test_corrupted_ratio_within_threshold_does_not_fail(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    # Widely-spread colors: small JPEG-quantized deltas can otherwise compress
    # distinct near-identical colors down to byte-identical files (false duplicates).
    for i in range(20):
        color = (i * 12 % 256, i * 47 % 256, i * 91 % 256)
        _make_jpeg(raw / f"good_{i}.jpg", color=color)
    (raw / "bad.jpg").write_bytes(b"garbage")

    result = DataIntakeScanner().scan(
        DataIntakeInput(
            raw_data_path=str(raw), dataset_name="ds", source="s",
            corrupted_ratio_threshold=0.1,  # 1/21 ~= 4.8% < 10%
        )
    )

    assert result.success is True
    assert result.status == DataIntakeStatus.PASSED


# ── 8-9. Duplicates ────────────────────────────────────────────────────────────


def test_duplicate_detected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_a.jpg", color=(1, 2, 3))
    # byte-identical copy
    (raw / "img_a_copy.jpg").write_bytes((raw / "img_a.jpg").read_bytes())
    _make_jpeg(raw / "img_b.jpg", color=(9, 9, 9))

    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s")
    )

    assert len(result.duplicate_groups) == 1
    group = result.duplicate_groups[0]
    assert "img_a.jpg" in group
    assert "img_a_copy.jpg" in group


def test_duplicate_ratio_exceeds_threshold_needs_approval(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_a.jpg", color=(1, 2, 3))
    (raw / "img_a_copy1.jpg").write_bytes((raw / "img_a.jpg").read_bytes())
    (raw / "img_a_copy2.jpg").write_bytes((raw / "img_a.jpg").read_bytes())
    _make_jpeg(raw / "img_b.jpg", color=(9, 9, 9))

    result = DataIntakeScanner().scan(
        DataIntakeInput(
            raw_data_path=str(raw), dataset_name="ds", source="s",
            duplicate_ratio_threshold=0.1,  # 2 extra copies / 4 files = 50% > 10%
        )
    )

    assert result.status == DataIntakeStatus.NEEDS_HUMAN_SOURCE_APPROVAL
    assert any("duplicate" in w.lower() for w in result.warnings)


# ── 10. Unexpected format ─────────────────────────────────────────────────────


def test_unexpected_format_file_flagged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")
    (raw / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (raw / "notes.txt").write_text("not an image", encoding="utf-8")

    result = DataIntakeScanner().scan(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s")
    )

    assert "notes.txt" in result.unexpected_format_files
    assert result.status == DataIntakeStatus.NEEDS_HUMAN_SOURCE_APPROVAL


# ── 11. Pillow unavailable fallback ───────────────────────────────────────────


def test_pillow_unavailable_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")

    real_import = builtins.__import__

    def _block_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("PIL not installed")
        return real_import(name, *args, **kwargs)

    cached = {k: v for k, v in sys.modules.items() if k == "PIL" or k.startswith("PIL.")}
    for k in cached:
        del sys.modules[k]
    try:
        with patch("builtins.__import__", side_effect=_block_pil):
            result = DataIntakeScanner().scan(
                DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s")
            )
    finally:
        sys.modules.update(cached)

    assert result.pillow_available is False
    assert result.valid_images == 1  # size-only fallback still accepts the file
    assert result.corrupted_images == []


# ── 12-13. DataIntakeAgent ─────────────────────────────────────────────────────


def test_agent_writes_manifest_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")
    artifacts_dir = tmp_path / "artifacts"

    agent = DataIntakeAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s")
    )

    assert (artifacts_dir / "dataset_manifest.json").exists()
    assert (artifacts_dir / "dataset_manifest.md").exists()
    assert result.manifest_path == str(artifacts_dir / "dataset_manifest.json")

    data = json.loads((artifacts_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert len(data["images"]) == 1


def test_agent_logs_to_mlflow_when_enabled(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = DataIntakeAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(DataIntakeInput(raw_data_path=str(raw), dataset_name="ds", source="s"))

    metrics = client.runs[run_id]["metrics"]
    assert "data_intake.num_files" in metrics
    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "data_intake"
    assert len(client.runs[run_id]["artifacts"]) > 0


# ── 14-16. CLI ─────────────────────────────────────────────────────────────────


def test_cli_data_intake_passes_on_clean_dataset(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")

    result = CliRunner().invoke(
        app, ["data-intake", str(raw), "--dataset-name", "ds", "--source", "camera1"]
    )

    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_cli_data_intake_exits_1_on_missing_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app, ["data-intake", str(tmp_path / "nonexistent"), "--dataset-name", "ds"]
    )

    assert result.exit_code == 1


def test_cli_data_intake_needs_approval_exits_0(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    raw = tmp_path / "raw"
    _make_jpeg(raw / "img_0.jpg")

    # no --source given -> needs_human_source_approval, but success=True (soft gate)
    result = CliRunner().invoke(app, ["data-intake", str(raw), "--dataset-name", "ds"])

    assert result.exit_code == 0, result.output
    assert "NEEDS_HUMAN_SOURCE_APPROVAL" in result.output
