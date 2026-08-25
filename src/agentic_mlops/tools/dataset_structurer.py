"""Dataset structurer — the core tool used by Dataset Structuring Agent.

Takes raw images (+ YOLO or COCO labels) and produces a YOLO-compatible
images/{train,val,test} + labels/{train,val,test} + data.yaml layout, with a
grouped or random train/val/test split.

"grouped_by_source" keeps whole groups (e.g. all frames of one video) together
in a single split — required to avoid leaking near-duplicate frames across
train/val/test. Groups are assigned to splits via a greedy largest-deficit
algorithm so ratios are matched as closely as group sizes allow.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agentic_mlops.contracts.dataset_structuring import (
    DatasetStructuringInput,
    DatasetStructuringOutput,
    LabelFormat,
    SplitAssignment,
    SplitStrategy,
)
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SPLITS = ("train", "val", "test")


@dataclass
class _Record:
    image_path: Path
    label_lines: list[str] = field(default_factory=list)


class DatasetStructurer:
    """Converts raw images (+ YOLO/COCO labels) into a split YOLO dataset."""

    def structure(self, inp: DatasetStructuringInput) -> DatasetStructuringOutput:
        raw_dir = Path(inp.raw_data_path).resolve()
        if not raw_dir.is_dir():
            return _failed(f"raw_data_path not found or not a directory: {raw_dir}")

        if inp.label_format == LabelFormat.YOLO:
            records, warnings = _load_yolo_source(raw_dir)
        elif inp.label_format == LabelFormat.COCO:
            if not inp.coco_annotations_path:
                return _failed("coco_annotations_path is required when label_format='coco'")
            coco_path = Path(inp.coco_annotations_path).resolve()
            if not coco_path.exists():
                return _failed(f"coco_annotations_path not found: {coco_path}")
            try:
                records, warnings = _load_coco_source(raw_dir, coco_path, inp.classes)
            except Exception as exc:
                return _failed(f"Failed to parse COCO annotations: {exc}")
        elif inp.label_format == LabelFormat.VOC:
            if inp.voc_annotations_dir:
                voc_dir = Path(inp.voc_annotations_dir).resolve()
                if not voc_dir.is_dir():
                    return _failed(f"voc_annotations_dir not found: {voc_dir}")
            else:
                # Convention: Annotations/ subdir first, then root of raw_data_path.
                voc_dir = raw_dir / "Annotations" if (raw_dir / "Annotations").is_dir() else raw_dir
            try:
                records, warnings = _load_voc_source(raw_dir, voc_dir, inp.classes)
            except Exception as exc:
                return _failed(f"Failed to parse VOC annotations: {exc}")
        else:
            return _failed(f"Unsupported label_format: {inp.label_format}")

        if not records:
            return _failed(f"No images found under {raw_dir}", warnings=warnings)

        assignments, split_warnings = _split_records(records, inp)
        warnings += split_warnings

        overlap = _detect_overlap(assignments)
        if overlap:
            return _failed(
                f"Internal error: {len(overlap)} file(s) assigned to more than one split: "
                f"{sorted(overlap)[:10]}"
            )

        out_dir = Path(inp.output_dataset_path).resolve()
        active_splits = {s for s in _SPLITS if _split_ratio(inp, s) > 0}
        _write_structured_dataset(out_dir, records, assignments, active_splits)
        data_yaml_path = _write_data_yaml(out_dir, inp.classes, active_splits)

        split_counts = dict(Counter(a.split for a in assignments))
        num_labels = sum(1 for r in records if r.label_lines)

        logger.info(
            "Dataset structuring complete",
            extra={"num_images": len(records), "split_counts": split_counts},
        )

        return DatasetStructuringOutput(
            success=True,
            message=(
                f"Structured {len(records)} image(s) into {split_counts} " f"under {out_dir}."
            ),
            structured_dataset_path=str(out_dir),
            data_yaml_path=str(data_yaml_path),
            num_images=len(records),
            num_labels=num_labels,
            split_counts=split_counts,
            classes=inp.classes,
            assignments=assignments,
            warnings=warnings,
        )


# ── YOLO source loading ───────────────────────────────────────────────────────


def _load_yolo_source(raw_dir: Path) -> tuple[list[_Record], list[str]]:
    img_dir = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
    lbl_dir = raw_dir / "labels" if (raw_dir / "labels").is_dir() else raw_dir

    images = sorted(
        p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    warnings: list[str] = []
    records: list[_Record] = []

    for img in images:
        label_path = lbl_dir / (img.stem + ".txt")
        lines: list[str] = []
        if label_path.exists():
            lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            warnings.append(f"No label file for image '{img.name}' — treated as background.")
        records.append(_Record(image_path=img, label_lines=lines))

    return records, warnings


# ── COCO source loading ────────────────────────────────────────────────────────


def _load_coco_source(
    raw_dir: Path, coco_path: Path, classes: list[str]
) -> tuple[list[_Record], list[str]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}
    categories_by_id: dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}
    class_index = {name: i for i, name in enumerate(classes)}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    img_dir = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
    warnings: list[str] = []
    records: list[_Record] = []

    for image_id, img_meta in images_by_id.items():
        file_name = img_meta["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            warnings.append(f"COCO image '{file_name}' not found under {img_dir} — skipped.")
            continue

        width, height = img_meta.get("width"), img_meta.get("height")
        if not width or not height:
            warnings.append(f"COCO image '{file_name}' missing width/height — skipped.")
            continue

        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            cat_name = categories_by_id.get(ann["category_id"])
            if cat_name is None or cat_name not in class_index:
                warnings.append(
                    f"COCO annotation on '{file_name}' references unknown category "
                    f"'{cat_name}' — skipped."
                )
                continue
            x, y, w, h = ann["bbox"]
            xc, yc = (x + w / 2) / width, (y + h / 2) / height
            wn, hn = w / width, h / height
            lines.append(f"{class_index[cat_name]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        records.append(_Record(image_path=img_path, label_lines=lines))

    return records, warnings


def _load_voc_source(
    raw_dir: Path, voc_dir: Path, classes: list[str]
) -> tuple[list[_Record], list[str]]:
    """Parse Pascal VOC XML annotations and convert bboxes to normalised YOLO format.

    xmin/ymin/xmax/ymax (absolute pixel) → (class_id, xc, yc, w, h) normalised.
    Unknown classes and missing images are skipped with a warning, not a hard failure.
    """
    img_dir = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
    class_index = {name: i for i, name in enumerate(classes)}
    warnings: list[str] = []
    records: list[_Record] = []

    xml_files = sorted(voc_dir.rglob("*.xml"))
    if not xml_files:
        warnings.append(f"No VOC XML annotation files found under {voc_dir}.")
        return records, warnings

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            warnings.append(f"Failed to parse VOC XML '{xml_path.name}': {exc} — skipped.")
            continue

        # Resolve the image file.  <filename> tag is preferred; fall back to same stem.
        file_name_el = root.find("filename")
        file_name = (
            file_name_el.text.strip() if file_name_el is not None and file_name_el.text else None
        )
        img_path: Path | None = None
        if file_name:
            candidate = img_dir / file_name
            if candidate.exists():
                img_path = candidate
        if img_path is None:
            img_path = next(
                (
                    img_dir / (xml_path.stem + ext)
                    for ext in _IMAGE_EXTS
                    if (img_dir / (xml_path.stem + ext)).exists()
                ),
                None,
            )
        if img_path is None:
            label = file_name or xml_path.stem
            warnings.append(
                f"VOC XML '{xml_path.name}': image '{label}' not found under {img_dir} — skipped."
            )
            continue

        size_el = root.find("size")
        if size_el is None:
            warnings.append(f"VOC XML '{xml_path.name}': missing <size> element — skipped.")
            continue
        try:
            width = int(size_el.findtext("width", "0") or "0")
            height = int(size_el.findtext("height", "0") or "0")
        except ValueError:
            warnings.append(f"VOC XML '{xml_path.name}': non-integer width/height — skipped.")
            continue
        if not width or not height:
            warnings.append(f"VOC XML '{xml_path.name}': zero width or height — skipped.")
            continue

        lines: list[str] = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or not name_el.text:
                warnings.append(f"VOC XML '{xml_path.name}': <object> missing <name> — skipped.")
                continue
            class_name = name_el.text.strip()
            if class_name not in class_index:
                warnings.append(
                    f"VOC XML '{xml_path.name}': unknown class '{class_name}' — skipped."
                )
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                warnings.append(f"VOC XML '{xml_path.name}': <object> missing <bndbox> — skipped.")
                continue
            try:
                xmin = float(bndbox.findtext("xmin", "0") or "0")
                ymin = float(bndbox.findtext("ymin", "0") or "0")
                xmax = float(bndbox.findtext("xmax", "0") or "0")
                ymax = float(bndbox.findtext("ymax", "0") or "0")
            except ValueError:
                warnings.append(
                    f"VOC XML '{xml_path.name}': non-numeric bndbox coordinates — skipped."
                )
                continue
            xc = (xmin + xmax) / 2 / width
            yc = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            lines.append(f"{class_index[class_name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        records.append(_Record(image_path=img_path, label_lines=lines))

    return records, warnings


# ── Split assignment ───────────────────────────────────────────────────────────


def _split_ratio(inp: DatasetStructuringInput, split: str) -> float:
    return {"train": inp.train_ratio, "val": inp.val_ratio, "test": inp.test_ratio}[split]


def _split_records(
    records: list[_Record], inp: DatasetStructuringInput
) -> tuple[list[SplitAssignment], list[str]]:
    warnings: list[str] = []
    groups: dict[str, list[_Record]] = defaultdict(list)

    if inp.split_strategy == SplitStrategy.GROUPED_BY_SOURCE and inp.group_by_regex:
        pattern = re.compile(inp.group_by_regex)
        for r in records:
            m = pattern.match(r.image_path.name)
            if m and m.groups():
                groups[m.group(1)].append(r)
            else:
                warnings.append(
                    f"group_by_regex did not match '{r.image_path.name}' — "
                    "using filename as its own group."
                )
                groups[r.image_path.name].append(r)
    else:
        if inp.split_strategy == SplitStrategy.GROUPED_BY_SOURCE:
            warnings.append(
                "split_strategy='grouped_by_source' but no group_by_regex given — "
                "each file treated as its own group (no leakage protection applied)."
            )
        for r in records:
            groups[r.image_path.name].append(r)

    group_items = list(groups.items())
    random.Random(inp.seed).shuffle(group_items)

    total = len(records)
    targets = {s: _split_ratio(inp, s) * total for s in _SPLITS}
    eligible_splits = [s for s in _SPLITS if targets[s] > 0] or ["train"]
    counts = dict.fromkeys(_SPLITS, 0)

    assignments: list[SplitAssignment] = []
    for key, recs in group_items:
        best_split = max(eligible_splits, key=lambda s: targets[s] - counts[s])
        counts[best_split] += len(recs)
        for r in recs:
            assignments.append(
                SplitAssignment(filename=r.image_path.name, split=best_split, group_key=key)
            )

    return assignments, warnings


def _detect_overlap(assignments: list[SplitAssignment]) -> set[str]:
    seen: dict[str, str] = {}
    overlap: set[str] = set()
    for a in assignments:
        if a.filename in seen and seen[a.filename] != a.split:
            overlap.add(a.filename)
        seen[a.filename] = a.split
    return overlap


# ── Output writing ─────────────────────────────────────────────────────────────


def _write_structured_dataset(
    out_dir: Path,
    records: list[_Record],
    assignments: list[SplitAssignment],
    active_splits: set[str],
) -> None:
    split_by_filename = {a.filename: a.split for a in assignments}

    for split in active_splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for r in records:
        split = split_by_filename[r.image_path.name]
        dst_img = out_dir / "images" / split / r.image_path.name
        shutil.copy2(r.image_path, dst_img)

        dst_lbl = out_dir / "labels" / split / (r.image_path.stem + ".txt")
        content = "\n".join(r.label_lines)
        dst_lbl.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _write_data_yaml(out_dir: Path, classes: list[str], active_splits: set[str]) -> Path:
    data: dict = {"path": "."}
    for split in _SPLITS:
        if split in active_splits:
            data[split] = f"images/{split}"
    data["names"] = dict(enumerate(classes))

    path = out_dir / "data.yaml"
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _failed(message: str, warnings: list[str] | None = None) -> DatasetStructuringOutput:
    return DatasetStructuringOutput(
        success=False, message=message, errors=[message], warnings=warnings or []
    )
