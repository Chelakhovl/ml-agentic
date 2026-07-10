"""Azure ML job entry script — runs on Azure compute.

Invoked by AzureMLTrainingRunner via CommandJob:
  python train_yolo.py
    --dataset-path ${{inputs.dataset}}
    --data-yaml    ${{inputs.data_yaml}}
    --model        yolo11n.pt
    --epochs       1
    --imgsz        320
    --batch        4
    --device       cpu
    --patience     20
    --output-dir   ${{outputs.model_output}}

The dataset-path comes from the Azure-mounted input URI, not a Windows local path.
A runtime data.yaml is generated pointing at that mount path so no local paths leak in.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO training entry script for Azure ML")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_yaml = _generate_runtime_data_yaml(args.dataset_path, args.data_yaml, output_dir)

    try:
        from ultralytics import YOLO, settings  # noqa: PLC0415

        settings.update({"mlflow": False})

        model = YOLO(args.model)
        model.train(
            data=str(runtime_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            project=str(output_dir),
            name="run",
        )
        save_dir = Path(model.trainer.save_dir)
    except Exception as exc:
        print(f"ERROR: YOLO training failed: {exc}", file=sys.stderr)
        sys.exit(1)

    for weight_name in ("best.pt", "last.pt"):
        src = save_dir / "weights" / weight_name
        if src.exists():
            shutil.copy2(src, output_dir / weight_name)

    for fname in ("results.csv", "args.yaml"):
        src = save_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    for plot in save_dir.glob("*.png"):
        shutil.copy2(plot, output_dir / plot.name)

    print(f"Training complete. Artifacts written to {output_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
