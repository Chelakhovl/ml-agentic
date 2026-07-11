"""Azure ML Managed Online Endpoint scoring script — runs inside the serving container.

Loaded by the Azure ML inference server, which calls init() once at container
startup and run(raw_data) once per scoring request. Deployed by
AzureMLOnlineEndpointDeployer via CodeConfiguration(code=<this directory>,
scoring_script="score.py"). Self-contained: no `agentic_mlops` package import,
since only azure_jobs/ is uploaded as the deployment's code snapshot (same
constraint as train_yolo.py / eval_yolo.py).

Request body (JSON): {"image_path": "<path the container can read>", "conf": 0.25}
Response body (JSON): {"detections": [{"class_id": 0, "confidence": 0.91,
                        "xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}, ...]}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

model = None

_WEIGHT_EXTS = (".pt",)


def _find_weights(model_dir: str) -> str:
    root = Path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"AZUREML_MODEL_DIR not found: {model_dir}")
    candidates = [p for p in root.rglob("*") if p.suffix in _WEIGHT_EXTS]
    if not candidates:
        raise FileNotFoundError(f"No .pt weights found under {model_dir}")
    # Prefer a file literally named best.pt if present (matches training/registry convention).
    best = next((p for p in candidates if p.name == "best.pt"), None)
    return str(best or candidates[0])


def init() -> None:
    """Called once when the container starts."""
    global model
    from ultralytics import YOLO  # noqa: PLC0415

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    weights_path = _find_weights(model_dir)
    model = YOLO(weights_path)


def run(raw_data: str) -> str:
    """Called once per scoring request. raw_data is the request body as a string."""
    if model is None:
        return json.dumps({"error": "model not initialized"})

    try:
        payload = json.loads(raw_data)
        image_path = payload["image_path"]
        conf = float(payload.get("conf", 0.25))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid request: {exc}"})

    results = model.predict(source=image_path, conf=conf, verbose=False)
    detections = _extract_detections(results[0])
    return json.dumps({"detections": detections})


def _extract_detections(result: object) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn
    cls = boxes.cls
    conf = boxes.conf
    detections = []
    for i in range(len(boxes)):
        xc, yc, w, h = (float(v) for v in xywhn[i])
        detections.append(
            {
                "class_id": int(cls[i]),
                "confidence": float(conf[i]),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            }
        )
    return detections
