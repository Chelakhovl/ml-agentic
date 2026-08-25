"""Azure ML job entry script — runs YOLO evaluation on Azure compute.

Invoked by AzureMLEvaluationRunner via CommandJob:
  python eval_yolo.py
    --weights      ${{inputs.weights}}
    --dataset-path ${{inputs.dataset}}
    --data-yaml    ${{inputs.data_yaml}}
    --imgsz        640
    --batch        8
    --device       cpu
    --output-dir   ${{outputs.eval_output}}

The dataset-path comes from the Azure-mounted input URI, not a Windows local path.
A runtime data.yaml is generated pointing at that mount path so no local paths leak in.
Writes metrics.json (global + per-class metrics) and any confusion-matrix / PR-curve
plots ultralytics produces into --output-dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def _generate_runtime_data_yaml(dataset_path: str, source_data_yaml: str, output_dir: Path) -> Path:
    """Generate a data.yaml at output_dir/runtime_data.yaml referencing the Azure mount path."""
    with open(source_data_yaml, encoding="utf-8") as fh:
        src = yaml.safe_load(fh) or {}

    runtime = {
        "path": dataset_path,
        "train": src.get("train", "images/train"),
        "val": src.get("val", "images/val"),
    }
    if "nc" in src:
        runtime["nc"] = src["nc"]
    if "names" in src:
        runtime["names"] = src["names"]

    runtime_path = output_dir / "runtime_data.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(yaml.dump(runtime, allow_unicode=True), encoding="utf-8")
    return runtime_path


def _extract_metrics(results, class_names: list[str]) -> dict:
    """Parse an Ultralytics val() results object into a plain JSON-able dict."""
    box = results.box
    if hasattr(box, "map50"):
        map50 = float(box.map50)
    elif hasattr(box, "map") and len(box.map) > 0:
        map50 = float(box.map[0])
    else:
        map50 = 0.0
    map50_95 = float(box.map)
    precision = float(box.mp) if hasattr(box, "mp") else 0.0
    recall = float(box.mr) if hasattr(box, "mr") else 0.0

    per_class: dict[str, dict] = {}
    ap_class_index = getattr(box, "ap_class_index", None)
    ap50_list = getattr(box, "ap50", None)
    ap_list = getattr(box, "ap", None)
    p_list = getattr(box, "p", None)
    r_list = getattr(box, "r", None)

    if ap_class_index is not None:
        names = getattr(results, "names", {}) or {}
        for i, cls_idx in enumerate(ap_class_index):
            idx = int(cls_idx)
            cls_name = names.get(idx) or (class_names[idx] if idx < len(class_names) else str(idx))
            per_class[cls_name] = {
                "precision": float(p_list[i]) if p_list is not None and i < len(p_list) else 0.0,
                "recall": float(r_list[i]) if r_list is not None and i < len(r_list) else 0.0,
                "map50": (
                    float(ap50_list[i]) if ap50_list is not None and i < len(ap50_list) else 0.0
                ),
                "map50_95": float(ap_list[i]) if ap_list is not None and i < len(ap_list) else 0.0,
            }

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "per_class_metrics": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO evaluation entry script for Azure ML")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_yaml = _generate_runtime_data_yaml(args.dataset_path, args.data_yaml, output_dir)

    with open(args.data_yaml, encoding="utf-8") as fh:
        src = yaml.safe_load(fh) or {}
    names = src.get("names", {})
    class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    try:
        from ultralytics import YOLO, settings  # noqa: PLC0415

        settings.update({"mlflow": False})

        model = YOLO(args.weights)
        results = model.val(
            data=str(runtime_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(output_dir),
            name="run",
        )
        save_dir = Path(results.save_dir)
    except Exception as exc:
        print(f"ERROR: YOLO evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    metrics = _extract_metrics(results, class_names)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for plot in save_dir.glob("*.png"):
        shutil.copy2(plot, output_dir / plot.name)

    print(f"Evaluation complete. Metrics written to {output_dir / 'metrics.json'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
