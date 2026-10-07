"""Shared pytest fixtures for building minimal YOLO dataset trees in tmp dirs."""

from __future__ import annotations

from pathlib import Path

# ── pytest CLI options ──────────────────────────────────────────────────────────
# --azure-config backs the opt-in `azure_integration`-marked tests under
# tests/integration/ (see pyproject.toml's markers list). Registered here
# rather than in a package-level conftest since testpaths=["tests"] makes
# this file the collection root for both tests/unit and tests/integration.


def pytest_addoption(parser) -> None:  # noqa: ANN001
    parser.addoption(
        "--azure-config",
        action="store",
        default=None,
        help="Path to a real azure_ml.yaml -- enables tests/integration/*azure_integration* tests.",
    )


# ── Fixture builders ───────────────────────────────────────────────────────────


def make_data_yaml(root: Path, class_names: list[str] | None = None) -> Path:
    if class_names is None:
        class_names = ["scratch", "dent", "crack"]
    content = "path: .\ntrain: images/train\nval: images/val\nnames:\n"
    for i, name in enumerate(class_names):
        content += f"  {i}: {name}\n"
    yaml_path = root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def make_image(path: Path) -> None:
    """Write a minimal 1×1 JPEG-like stub (just needs to be a non-zero file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid JPEG header — enough for hash-based duplicate detection
    path.write_bytes(b"\xff\xd8\xff\xe0" + path.name.encode() + b"\xff\xd9")


def make_label(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_valid_dataset(root: Path, num_classes: int = 3) -> None:
    """Create a minimal valid YOLO dataset."""
    class_names = ["scratch", "dent", "crack"][:num_classes]
    make_data_yaml(root, class_names)

    for split in ("train", "val"):
        for idx in range(3):
            img = root / "images" / split / f"img_{split}_{idx:03d}.jpg"
            lbl = root / "labels" / split / f"img_{split}_{idx:03d}.txt"
            make_image(img)
            # Valid label: class 0, centred bbox
            make_label(lbl, "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
