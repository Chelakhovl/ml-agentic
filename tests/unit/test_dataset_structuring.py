"""Unit tests for the Dataset Structuring Agent, structurer tool, and CLI command.

Coverage matrix:
    1.  raw_data_path not found -> failed
    2.  No images found -> failed
    3.  YOLO source: images without a label file are treated as background (+ warning)
    4.  Random split respects approximate ratios, no cross-split overlap
    5.  data.yaml only lists splits with ratio > 0
    6.  grouped_by_source + regex keeps whole groups in a single split (no leakage)
    7.  grouped_by_source without a regex degrades to per-file grouping (+ warning)
    8.  group_by_regex not matching a filename -> warning, file becomes its own group
    9.  COCO source: bbox correctly converted to normalised YOLO coordinates
   10.  COCO source: unknown category skipped with a warning
   11.  COCO source: image file missing on disk skipped with a warning
   12.  COCO format requires coco_annotations_path
   13.  classes must be non-empty (pydantic)
   14.  train/val/test ratios must sum to 1.0 (pydantic)
   15.  DatasetStructuringAgent writes split_report.json / .md
   16.  DatasetStructuringAgent logs to MLflow when enabled
   17.  CLI: structure-dataset command exists and succeeds on a clean source (exit 0)
   18.  CLI: structure-dataset exits 1 on missing raw_data_path
   19.  CLI: structure-dataset exits 1 on invalid --label-format
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from agentic_mlops.agents.dataset_structuring import DatasetStructuringAgent
from agentic_mlops.contracts.dataset_structuring import (
    DatasetStructuringInput,
    LabelFormat,
    SplitStrategy,
)
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.dataset_structurer import DatasetStructurer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_yolo_source(raw: Path, n: int, with_labels: bool = True) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        color = (i * 23 % 256, i * 61 % 256, i * 97 % 256)
        Image.new("RGB", (16, 16), color=color).save(raw / f"img_{i:03d}.jpg")
        if with_labels:
            (raw / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _all_assigned_filenames(assignments) -> dict[str, list[str]]:
    by_split: dict[str, list[str]] = {}
    for a in assignments:
        by_split.setdefault(a.split, []).append(a.filename)
    return by_split


# ── 1-2. Structural failures ──────────────────────────────────────────────────


def test_raw_path_not_found_fails(tmp_path: Path) -> None:
    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(tmp_path / "nonexistent"),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
        )
    )

    assert result.success is False
    assert "not found" in result.message.lower()


def test_no_images_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw), output_dataset_path=str(tmp_path / "out"), classes=["a"]
        )
    )

    assert result.success is False
    assert "no images" in result.message.lower()


# ── 3. Missing label = background ─────────────────────────────────────────────


def test_missing_label_treated_as_background(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 3, with_labels=False)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw), output_dataset_path=str(tmp_path / "out"), classes=["a"]
        )
    )

    assert result.success is True
    assert result.num_images == 3
    assert result.num_labels == 0
    assert any("background" in w.lower() for w in result.warnings)


# ── 4-5. Random split + data.yaml ─────────────────────────────────────────────


def test_random_split_ratios_and_no_overlap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 20)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
        )
    )

    assert result.success is True
    assert sum(result.split_counts.values()) == 20
    assert result.split_counts["train"] == pytest.approx(14, abs=1)

    by_split = _all_assigned_filenames(result.assignments)
    all_files = [f for files in by_split.values() for f in files]
    assert len(all_files) == len(set(all_files))  # no file assigned twice


def test_data_yaml_only_lists_active_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 5)
    out_dir = tmp_path / "out"

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(out_dir),
            classes=["a", "b"],
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
        )
    )

    assert result.success is True
    data_yaml = (out_dir / "data.yaml").read_text(encoding="utf-8")
    assert "train:" in data_yaml
    assert "val:" in data_yaml
    assert "test:" not in data_yaml
    assert not (out_dir / "images" / "test").exists()


# ── 6-8. Grouped split ─────────────────────────────────────────────────────────


def test_grouped_by_source_keeps_groups_together(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for v in range(4):
        for f in range(5):
            color = (v * 50 % 256, f * 30 % 256, (v + f) * 11 % 256)
            Image.new("RGB", (16, 16), color=color).save(raw / f"video{v}_frame{f}.jpg")

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
            split_strategy=SplitStrategy.GROUPED_BY_SOURCE,
            group_by_regex=r"^(video\d+)_",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
    )

    assert result.success is True
    groups_per_split: dict[str, set[str]] = {}
    for a in result.assignments:
        groups_per_split.setdefault(a.split, set()).add(a.group_key)

    seen_groups: dict[str, str] = {}
    for split, groups in groups_per_split.items():
        for g in groups:
            assert g not in seen_groups, f"group {g} leaked into multiple splits"
            seen_groups[g] = split
    assert len(seen_groups) == 4  # video0..video3


def test_grouped_by_source_without_regex_warns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 5)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
            split_strategy=SplitStrategy.GROUPED_BY_SOURCE,
        )
    )

    assert result.success is True
    assert any("no group_by_regex" in w.lower() for w in result.warnings)


def test_group_by_regex_not_matching_warns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 3)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
            split_strategy=SplitStrategy.GROUPED_BY_SOURCE,
            group_by_regex=r"^(nomatch_\d+)_",
        )
    )

    assert result.success is True
    assert any("did not match" in w.lower() for w in result.warnings)


# ── 9-12. COCO source ──────────────────────────────────────────────────────────


def _write_coco(
    tmp_path: Path,
    raw: Path,
    bbox: tuple[int, int, int, int] = (10, 20, 30, 40),
    category_id: int = 5,
    category_name: str = "dent",
    file_name: str = "img1.jpg",
    width: int = 100,
    height: int = 200,
    write_image: bool = True,
) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    if write_image:
        Image.new("RGB", (width, height)).save(raw / file_name)
    coco = {
        "images": [{"id": 1, "file_name": file_name, "width": width, "height": height}],
        "annotations": [{"image_id": 1, "category_id": category_id, "bbox": list(bbox)}],
        "categories": [{"id": category_id, "name": category_name}],
    }
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path


def test_coco_bbox_converted_correctly(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    coco_path = _write_coco(tmp_path, raw, bbox=(10, 20, 30, 40), width=100, height=200)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["scratch", "dent", "crack"],
            label_format=LabelFormat.COCO,
            coco_annotations_path=str(coco_path),
            train_ratio=1e-9,
            val_ratio=1.0 - 2e-9,
            test_ratio=1e-9,
        )
    )

    assert result.success is True
    label_files = list((tmp_path / "out" / "labels").rglob("*.txt"))
    assert len(label_files) == 1
    parts = label_files[0].read_text(encoding="utf-8").strip().split()
    assert parts[0] == "1"  # index of 'dent' in classes
    assert float(parts[1]) == pytest.approx(0.25)   # xc
    assert float(parts[2]) == pytest.approx(0.20)   # yc
    assert float(parts[3]) == pytest.approx(0.30)   # w
    assert float(parts[4]) == pytest.approx(0.20)   # h


def test_coco_unknown_category_skipped_with_warning(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    coco_path = _write_coco(tmp_path, raw, category_id=99, category_name="unknown_class")

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["scratch", "dent"],
            label_format=LabelFormat.COCO,
            coco_annotations_path=str(coco_path),
        )
    )

    assert result.success is True
    assert any("unknown category" in w.lower() for w in result.warnings)
    label_files = list((tmp_path / "out" / "labels").rglob("*.txt"))
    assert label_files[0].read_text(encoding="utf-8").strip() == ""


def test_coco_missing_image_file_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    coco_path = _write_coco(tmp_path, raw, write_image=False)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["scratch", "dent"],
            label_format=LabelFormat.COCO,
            coco_annotations_path=str(coco_path),
        )
    )

    assert result.success is False  # no images survived -> "No images found"
    assert any("not found" in w.lower() for w in [*result.warnings, result.message])


def test_coco_requires_annotations_path(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 1)

    result = DatasetStructurer().structure(
        DatasetStructuringInput(
            raw_data_path=str(raw),
            output_dataset_path=str(tmp_path / "out"),
            classes=["a"],
            label_format=LabelFormat.COCO,
        )
    )

    assert result.success is False
    assert "coco_annotations_path" in result.message


# ── 13-14. Contract validation ────────────────────────────────────────────────


def test_empty_classes_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        DatasetStructuringInput(raw_data_path="/a", output_dataset_path="/b", classes=[])


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        DatasetStructuringInput(
            raw_data_path="/a",
            output_dataset_path="/b",
            classes=["x"],
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.2,
        )


# ── 15-16. DatasetStructuringAgent ────────────────────────────────────────────


def test_agent_writes_split_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 5)
    artifacts_dir = tmp_path / "artifacts"

    agent = DatasetStructuringAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DatasetStructuringInput(
            raw_data_path=str(raw), output_dataset_path=str(tmp_path / "out"), classes=["a"]
        )
    )

    assert (artifacts_dir / "split_report.json").exists()
    assert (artifacts_dir / "split_report.md").exists()
    assert result.split_report_path == str(artifacts_dir / "split_report.json")

    data = json.loads((artifacts_dir / "split_report.json").read_text(encoding="utf-8"))
    assert data["success"] is True
    assert data["num_images"] == 5


def test_agent_logs_to_mlflow_when_enabled(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_yolo_source(raw, 5)
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = DatasetStructuringAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(
        DatasetStructuringInput(
            raw_data_path=str(raw), output_dataset_path=str(tmp_path / "out"), classes=["a"]
        )
    )

    metrics = client.runs[run_id]["metrics"]
    assert "dataset_structuring.num_images" in metrics
    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "dataset_structuring"
    assert len(client.runs[run_id]["artifacts"]) > 0


# ── 17-19. CLI ─────────────────────────────────────────────────────────────────


def test_cli_structure_dataset_succeeds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    raw = tmp_path / "raw"
    _make_yolo_source(raw, 5)

    result = CliRunner().invoke(
        app,
        [
            "structure-dataset",
            str(raw),
            "--output-dataset-path", str(tmp_path / "out"),
            "--classes", "scratch,dent",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "SUCCESS" in result.output


def test_cli_structure_dataset_exits_1_on_missing_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "structure-dataset",
            str(tmp_path / "nonexistent"),
            "--output-dataset-path", str(tmp_path / "out"),
            "--classes", "a",
        ],
    )

    assert result.exit_code == 1


def test_cli_structure_dataset_exits_1_on_invalid_label_format(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    raw = tmp_path / "raw"
    _make_yolo_source(raw, 1)

    result = CliRunner().invoke(
        app,
        [
            "structure-dataset",
            str(raw),
            "--output-dataset-path", str(tmp_path / "out"),
            "--classes", "a",
            "--label-format", "bogus",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid label-format" in result.output
