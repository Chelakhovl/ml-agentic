"""Azure ML Managed Online Endpoint scoring script — runs inside the serving container.

Loaded by the Azure ML inference server, which calls init() once at container
startup and run(raw_data) once per scoring request. Deployed by
AzureMLOnlineEndpointDeployer via CodeConfiguration(code=<this directory>,
scoring_script="score.py"). Self-contained: no `agentic_mlops` package import,
since only azure_jobs/ is uploaded as the deployment's code snapshot (same
constraint as train_yolo.py / eval_yolo.py).

Request body (JSON): {"image_path": "<path the container can read>",
                      "image_id": "<optional id>", "conf": 0.25}
Response body (JSON): {"detections": [{"class_id": 0, "confidence": 0.91,
                        "xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}, ...]}

App Insights telemetry (soft dependency on opencensus-ext-azure):
  Enabled when APPLICATIONINSIGHTS_CONNECTION_STRING env var is set.
  Emits one "inference_record" trace per request with custom_dimensions:
    endpoint_name, image_id, latency_ms, error, detections (JSON string).
  These traces are queryable by ApplicationInsightsLogClient / MonitoringAgent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

model = None
_endpoint_name: str = "unknown"
_ai_logger: logging.Logger | None = None

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


def _setup_ai_telemetry() -> None:
    """Attach AzureEventHandler when APPLICATIONINSIGHTS_CONNECTION_STRING is set.

    Silently no-ops when the env var is absent or opencensus-ext-azure is not
    installed — inference always continues regardless of telemetry availability.
    """
    global _ai_logger
    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn_str:
        return
    try:
        from opencensus.ext.azure.log_exporter import AzureEventHandler  # noqa: PLC0415

        handler = AzureEventHandler(connection_string=conn_str)
        handler.setLevel(logging.INFO)
        lgr = logging.getLogger("agentic_mlops.score")
        lgr.setLevel(logging.INFO)
        if not any(isinstance(h, AzureEventHandler) for h in lgr.handlers):
            lgr.addHandler(handler)
        _ai_logger = lgr
    except ImportError:
        pass  # opencensus-ext-azure not installed — telemetry disabled silently


def _emit_inference_record(
    image_id: str,
    latency_ms: float,
    is_error: bool,
    detections: list[dict],
    endpoint_name: str,
) -> None:
    """Emit one structured trace to App Insights. Never raises — telemetry is best-effort."""
    if _ai_logger is None:
        return
    try:
        _ai_logger.info(
            "inference_record",
            extra={
                "custom_dimensions": {
                    "endpoint_name": endpoint_name,
                    "image_id": image_id,
                    "latency_ms": latency_ms,
                    "error": is_error,
                    "detections": json.dumps(detections),
                }
            },
        )
    except Exception:  # noqa: BLE001
        pass


def init() -> None:
    """Called once when the container starts."""
    global model, _endpoint_name
    from ultralytics import YOLO  # noqa: PLC0415

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    weights_path = _find_weights(model_dir)
    model = YOLO(weights_path)
    _endpoint_name = os.getenv("AZUREML_ENDPOINT_NAME", "unknown")
    _setup_ai_telemetry()


def run(raw_data: str) -> str:
    """Called once per scoring request. raw_data is the request body as a string."""
    t0 = time.perf_counter()
    image_id = "unknown"

    if model is None:
        return json.dumps({"error": "model not initialized"})

    try:
        payload = json.loads(raw_data)
        image_path = payload["image_path"]
        image_id = str(payload.get("image_id") or Path(image_path).name)
        conf = float(payload.get("conf", 0.25))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        _emit_inference_record(image_id, latency_ms, True, [], _endpoint_name)
        return json.dumps({"error": f"invalid request: {exc}"})

    try:
        results = model.predict(source=image_path, conf=conf, verbose=False)
        detections = _extract_detections(results[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        _emit_inference_record(image_id, latency_ms, False, detections, _endpoint_name)
        return json.dumps({"detections": detections})
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        _emit_inference_record(image_id, latency_ms, True, [], _endpoint_name)
        return json.dumps({"error": f"prediction failed: {exc}"})


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
