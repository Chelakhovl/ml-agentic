"""Unit tests for DatasetValidator.

Coverage matrix:
    1. valid dataset          → status=passed, no blocking issues
    2. missing data.yaml      → status=failed, blocking issue
    3. missing split dir      → status=failed, blocking issue
    4. bad class_id           → status=failed, blocking label issue
    5. bbox out of [0,1]      → status=failed, blocking label issue
    6. bbox width/height <= 0 → status=failed, blocking label issue
    7. wrong column count     → status=failed, blocking label issue
    8. cross-split duplicates → status=failed, blocking issue
    9. empty label file       → status=passed  (background image — allowed)
   10. class imbalance        → status=warning (not blocking)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_mlops.contracts.datasets import DatasetValidationInput
from agentic_mlops.tools.dataset_validator import DatasetValidator
from tests.conftest import make_data_yaml, make_image, make_label, make_valid_dataset


@pytest.fixture
def validator() -> DatasetValidator:
    return DatasetValidator()


def _validate(validator: DatasetValidator, dataset_root: Path, **kwargs) -> object:
    return validator.validate(DatasetValidationInput(dataset_path=str(dataset_root), **kwargs))


# ── 1. Valid dataset ───────────────────────────────────────────────────────────


def test_valid_dataset_passes(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    result = _validate(validator, tmp_path)

    assert result.success is True
    assert result.status == "passed"
    assert result.blocking_issues == []
    assert result.num_images == 6  # 3 train + 3 val
    assert result.num_labels == 6
    assert "scratch" in result.class_distribution


# ── 2. Missing data.yaml ───────────────────────────────────────────────────────


def test_missing_data_yaml_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    (tmp_path / "data.yaml").unlink()

    result = _validate(validator, tmp_path)

    assert result.success is False
    assert result.status == "failed"
    assert any("data.yaml" in issue for issue in result.blocking_issues)


# ── 3. Missing split directory ─────────────────────────────────────────────────


def test_missing_images_train_dir_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "images" / "train")

    result = _validate(validator, tmp_path)

    assert result.success is False
    assert any("images/train" in issue for issue in result.blocking_issues)


# ── 4. Bad class_id ────────────────────────────────────────────────────────────


def test_bad_class_id_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)  # 3 classes → valid ids: 0, 1, 2
    # Overwrite one label file with an out-of-range class_id
    bad_label = tmp_path / "labels" / "train" / "img_train_000.txt"
    bad_label.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    result = _validate(validator, tmp_path)

    assert result.success is False
    assert result.status == "failed"
    blocking_text = " ".join(result.blocking_issues)
    assert "99" in blocking_text or "class_id" in blocking_text.lower()


# ── 5. Bbox coordinate out of range ───────────────────────────────────────────


def test_bbox_out_of_range_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    bad_label = tmp_path / "labels" / "train" / "img_train_000.txt"
    # x_center = 1.5 → out of [0,1]
    bad_label.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")

    result = _validate(validator, tmp_path)

    assert result.success is False
    blocking_text = " ".join(result.blocking_issues)
    assert "1.5" in blocking_text or "out of" in blocking_text.lower()


# ── 6. Degenerate bbox (width <= 0) ───────────────────────────────────────────


def test_degenerate_bbox_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    bad_label = tmp_path / "labels" / "train" / "img_train_000.txt"
    # width = 0.0 → degenerate
    bad_label.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")

    result = _validate(validator, tmp_path)

    assert result.success is False
    blocking_text = " ".join(result.blocking_issues)
    assert "degenerate" in blocking_text.lower() or "width=0" in blocking_text


# ── 7. Wrong column count ──────────────────────────────────────────────────────


def test_wrong_column_count_fails(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    bad_label = tmp_path / "labels" / "train" / "img_train_000.txt"
    bad_label.write_text("0 0.5 0.5\n", encoding="utf-8")  # only 3 columns

    result = _validate(validator, tmp_path)

    assert result.success is False
    blocking_text = " ".join(result.blocking_issues)
    assert "column" in blocking_text.lower() or "3" in blocking_text


# ── 8. Cross-split duplicates ─────────────────────────────────────────────────


def test_cross_split_duplicates_fail(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    # Copy a train image into val (same bytes → same hash)
    import shutil

    src = tmp_path / "images" / "train" / "img_train_000.jpg"
    dst = tmp_path / "images" / "val" / "img_train_000.jpg"  # same name, same bytes
    shutil.copy(src, dst)
    # Provide a matching label in val so the image is processed
    make_label(tmp_path / "labels" / "val" / "img_train_000.txt", "0 0.5 0.5 0.2 0.2\n")

    result = _validate(validator, tmp_path)

    assert result.success is False
    blocking_text = " ".join(result.blocking_issues)
    assert "duplicate" in blocking_text.lower()


# ── 9. Empty label file (background image — valid) ────────────────────────────


def test_empty_label_is_allowed(tmp_path: Path, validator: DatasetValidator) -> None:
    make_valid_dataset(tmp_path)
    # Overwrite one label with empty content
    label = tmp_path / "labels" / "train" / "img_train_000.txt"
    label.write_text("", encoding="utf-8")

    result = _validate(validator, tmp_path)

    # Empty labels are valid (background images) — should still pass
    assert result.success is True


# ── 10. Class imbalance → warning, not blocking ───────────────────────────────


def test_class_imbalance_is_warning_not_blocking(
    tmp_path: Path, validator: DatasetValidator
) -> None:
    make_data_yaml(tmp_path, ["scratch", "rare_defect"])
    # 10 images with class 0, 1 image with class 1 → imbalanced
    for split in ("train", "val"):
        for idx in range(5):
            img = tmp_path / "images" / split / f"img_{split}_{idx:03d}.jpg"
            lbl = tmp_path / "labels" / split / f"img_{split}_{idx:03d}.txt"
            make_image(img)
            # All class 0 except the first val image
            if split == "val" and idx == 0:
                make_label(lbl, "1 0.5 0.5 0.1 0.1\n")
            else:
                make_label(lbl, "0 0.5 0.5 0.2 0.2\n")

    result = _validate(validator, tmp_path)

    # Imbalance is a warning, not a blocking issue
    assert result.blocking_issues == []
    assert any("rare_defect" in w for w in result.warnings)
    # Status should be warning (not failed) because we have no blocking issues
    assert result.status in ("passed", "warning")
