"""Inference log clients for the Monitoring Agent.

Three implementations, following the same injection pattern as
AzureMLTrainingRunner / AzureMLEvaluationRunner:

  LocalFileLogClient          — reads a local JSON-Lines predictions log
                                 (the existing default behaviour)
  ApplicationInsightsLogClient — queries the App Insights `traces` table
                                 via azure-monitor-query; requires records
                                 to have been emitted by azure_jobs/score.py
  FakeInferenceLogClient       — in-memory test double

All three return (records: list[dict], warnings: list[str]) where each
record is a dict matching the MonitoringInput log format:
    {"timestamp": ..., "image_id": ..., "latency_ms": ...,
     "error": ..., "detections": [{"class": ..., "confidence": ...}, ...]}
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# KQL emitted by score.py: message="inference_record" + custom_dimensions
_KQL_TEMPLATE = """\
traces
| where message == "inference_record"
| where customDimensions.endpoint_name == "{endpoint_name}"
| where timestamp >= ago({window_seconds}s)
| project
    log_timestamp = tostring(timestamp),
    image_id      = tostring(customDimensions.image_id),
    latency_ms    = todouble(customDimensions.latency_ms),
    is_error      = tobool(customDimensions.error),
    detections_json = tostring(customDimensions.detections)
| order by log_timestamp asc
"""


# ── Base ───────────────────────────────────────────────────────────────────────


class InferenceLogClient:
    """Base class — all log sources implement this interface."""

    def fetch_records(self) -> tuple[list[dict], list[str]]:
        """Return (records, warnings).

        Each record is a dict with keys:
            timestamp   str | None
            image_id    str
            latency_ms  float | None
            error       bool
            detections  list[dict]  — each {class: str, confidence: float}
        Warnings are non-fatal advisory messages.
        """
        raise NotImplementedError


# ── Local file ────────────────────────────────────────────────────────────────


class LocalFileLogClient(InferenceLogClient):
    """Reads inference records from a local JSON-Lines predictions file.

    Wraps the same loading logic previously inlined in tools/monitor.py so
    ModelMonitor can use it via the common InferenceLogClient interface.
    """

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)

    def fetch_records(self) -> tuple[list[dict], list[str]]:
        records: list[dict] = []
        warnings: list[str] = []
        try:
            with self._path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        warnings.append(
                            f"Skipping malformed JSON on line {lineno}: {exc}"
                        )
                        continue
                    if not isinstance(record, dict):
                        warnings.append(
                            f"Skipping non-object record on line {lineno}."
                        )
                        continue
                    records.append(record)
        except OSError as exc:
            warnings.append(f"Cannot read log file: {exc}")
        return records, warnings


# ── Application Insights ──────────────────────────────────────────────────────


class ApplicationInsightsLogClient(InferenceLogClient):
    """Queries the Application Insights `traces` table for endpoint inference records.

    Records must have been emitted by azure_jobs/score.py (or any scoring script
    that calls the same opencensus logger with ``message="inference_record"`` and
    the expected ``custom_dimensions``).

    Auth: DefaultAzureCredential — ``az login`` or service-principal env vars.

    Args:
        workspace_id:   Application Insights Application ID (the GUID shown in
                        the portal under Overview → Application ID).  This is
                        passed as the ``workspace_id`` parameter to
                        ``LogsQueryClient.query_workspace()``.
        endpoint_name:  The ``endpoint_name`` value stored in
                        ``customDimensions`` by score.py.  Used in a KQL WHERE
                        clause so only records for this endpoint are returned.
        window_seconds: How far back to query (e.g. 86400 for 24 h).  Baked
                        into the KQL as ``ago(<N>s)`` so App Insights does the
                        time-range filtering server-side.
        credential:     Optional Azure credential.  Defaults to
                        ``DefaultAzureCredential()``.
    """

    def __init__(
        self,
        workspace_id: str,
        endpoint_name: str,
        window_seconds: int,
        credential: Any | None = None,
    ) -> None:
        try:
            from azure.monitor.query import LogsQueryClient  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "azure-monitor-query is not installed. "
                "Run: pip install 'agentic-mlops-yolo[azure]'"
            ) from exc
        try:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            cred = credential or DefaultAzureCredential()
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is not installed. "
                "Run: pip install 'agentic-mlops-yolo[azure]'"
            ) from exc

        self._client = LogsQueryClient(cred)
        self._workspace_id = workspace_id
        self._endpoint_name = endpoint_name
        self._window_seconds = window_seconds

    def fetch_records(self) -> tuple[list[dict], list[str]]:
        from azure.monitor.query import LogsQueryStatus  # noqa: PLC0415

        kql = _KQL_TEMPLATE.format(
            endpoint_name=self._endpoint_name.replace("'", "\\'"),
            window_seconds=self._window_seconds,
        )
        warnings: list[str] = []
        try:
            response = self._client.query_workspace(
                workspace_id=self._workspace_id,
                query=kql,
                timespan=timedelta(seconds=self._window_seconds),
            )
        except Exception as exc:
            warnings.append(f"App Insights query failed: {exc}")
            return [], warnings

        if response.status == LogsQueryStatus.FAILURE:
            err = getattr(response, "partial_error", None) or "unknown error"
            warnings.append(f"App Insights query returned FAILURE: {err}")
            return [], warnings

        if response.status == LogsQueryStatus.PARTIAL:
            warnings.append(
                "App Insights query returned PARTIAL results — some data may be missing."
            )
            tables = response.partial_data
        else:
            tables = response.tables

        if not tables:
            return [], warnings

        records = _rows_to_records(tables[0])
        logger.debug(
            "Fetched %d inference records from App Insights workspace %s",
            len(records),
            self._workspace_id,
        )
        return records, warnings


def _rows_to_records(table: Any) -> list[dict]:
    """Convert a LogsTable to the standard predictions-log dict format."""
    records: list[dict] = []
    col_names = [col.name for col in table.columns]
    for row in table.rows:
        row_dict = dict(zip(col_names, row))
        try:
            detections = json.loads(row_dict.get("detections_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            detections = []
        records.append(
            {
                "timestamp": row_dict.get("log_timestamp"),
                "image_id": row_dict.get("image_id") or "unknown",
                "latency_ms": _safe_float(row_dict.get("latency_ms")),
                "error": bool(row_dict.get("is_error") or False),
                "detections": [
                    _normalise_detection(d) for d in detections if isinstance(d, dict)
                ],
            }
        )
    return records


def _normalise_detection(d: dict) -> dict:
    """Map score.py's {class_id, confidence} to the monitor's {class, confidence}."""
    return {
        "class": str(d.get("class") or d.get("class_id", "unknown")),
        "confidence": float(d.get("confidence", 0.0)),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Fake (test double) ────────────────────────────────────────────────────────


class FakeInferenceLogClient(InferenceLogClient):
    """In-memory test double.  Returns pre-set records and warnings."""

    def __init__(
        self,
        records: list[dict],
        warnings: list[str] | None = None,
    ) -> None:
        self._records = records
        self._warnings = warnings or []

    def fetch_records(self) -> tuple[list[dict], list[str]]:
        return list(self._records), list(self._warnings)
