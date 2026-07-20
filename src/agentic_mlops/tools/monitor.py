"""Model monitor — the core tool used by the Monitoring Agent.

Supports two log sources (MonitoringInput.source):
  "local"         — reads a local JSON-Lines predictions log (original behaviour)
  "azure_monitor" — queries Application Insights via ApplicationInsightsLogClient

Legacy docstring note: a local release
directory, not a live serving target). Monitoring here means reading a local
JSON-Lines predictions log — the kind of export a real serving stack would
eventually produce from Azure Monitor/App Insights — and computing
latency/error/confidence/drift signals from it deterministically.

Each log line is one inference record:
    {"timestamp": "2026-07-10T12:00:00Z", "image_id": "img_0001.jpg",
     "latency_ms": 42.0, "error": false,
     "detections": [{"class": "scratch", "confidence": 0.91}, ...]}

An image counts as a "hard sample" (mined for review, same idea as the
Annotation Agent's low/medium buckets) if it has zero detections, or its
weakest detection's confidence is below `low_confidence_threshold` — the
same "weakest detection decides" rule ConfidenceThresholds uses.

Drift is measured as total variation distance between the current window's
class distribution and a baseline distribution (e.g. the training dataset's
class_distribution) — a simple, dependency-free metric in [0, 1].

Never triggers retraining itself — only recommends an action. Matches the
"no auto-promote" rule every other agent in this codebase already follows.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.contracts.monitoring import (
    HardSample,
    MonitoringInput,
    MonitoringOutput,
    MonitoringStatus,
    RecommendedAction,
)
from agentic_mlops.observability.logging import get_logger

if TYPE_CHECKING:
    from agentic_mlops.integrations.appinsights_log_client import InferenceLogClient

logger = get_logger(__name__)

_WINDOW_RE = re.compile(r"^(\d+)(s|m|h|d)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ModelMonitor:
    """Reads a predictions log and produces a monitoring report + hard sample manifest.

    Accepts an optional ``log_client`` injection for testing or for non-local
    sources (e.g. ApplicationInsightsLogClient).  When not injected, the client
    is constructed automatically based on ``MonitoringInput.source``.
    """

    def __init__(
        self,
        log_client: InferenceLogClient | None = None,
    ) -> None:
        self._log_client = log_client

    def _resolve_client(
        self, inp: MonitoringInput, window_seconds: int
    ) -> tuple[InferenceLogClient | None, MonitoringOutput | None]:
        """Return (client, None) on success or (None, error_output) on failure."""
        from agentic_mlops.integrations.appinsights_log_client import (  # noqa: PLC0415
            ApplicationInsightsLogClient,
            LocalFileLogClient,
        )

        if self._log_client is not None:
            return self._log_client, None

        if inp.source == "azure_monitor":
            if not inp.app_insights_workspace_id:
                return None, _failed(
                    "app_insights_workspace_id is required when source='azure_monitor'."
                )
            try:
                client: InferenceLogClient = ApplicationInsightsLogClient(
                    workspace_id=inp.app_insights_workspace_id,
                    endpoint_name=inp.endpoint_name,
                    window_seconds=window_seconds,
                )
            except RuntimeError as exc:
                return None, _failed(str(exc))
            return client, None

        # source == "local" (default)
        log_path = Path(inp.predictions_log_path)
        if not log_path.is_file():
            return None, _failed(f"predictions_log_path not found: {log_path}")
        return LocalFileLogClient(log_path), None

    def run(self, inp: MonitoringInput, artifacts_dir: Path) -> MonitoringOutput:
        window_seconds = _parse_window(inp.monitoring_window)
        if window_seconds is None:
            return _failed(
                f"Invalid monitoring_window '{inp.monitoring_window}'. "
                "Expected e.g. '24h', '7d', '30m', '60s'."
            )

        log_client, err = self._resolve_client(inp, window_seconds)
        if err is not None:
            return err

        assert log_client is not None
        records, warnings = log_client.fetch_records()
        if not records:
            source_desc = (
                inp.app_insights_workspace_id or inp.predictions_log_path or inp.source
            )
            return _failed(f"No prediction records found in {source_desc}")

        filtered, window_start, window_end = _filter_window(records, window_seconds)
        if not filtered:
            return _failed(
                f"No prediction records fall within the {inp.monitoring_window} window "
                f"ending {window_end}."
            )

        baseline: dict[str, float] | None = None
        if inp.baseline_class_distribution_path:
            baseline, err = _load_baseline(inp.baseline_class_distribution_path)
            if err:
                return _failed(err)

        total = len(filtered)
        error_count = sum(1 for r in filtered if r.get("error") is True)
        error_rate = error_count / total

        latencies = [
            float(r["latency_ms"])
            for r in filtered
            if isinstance(r.get("latency_ms"), int | float)
        ]
        if not latencies:
            warnings.append("No latency_ms values found in the windowed log.")
        p95_latency_ms = _percentile(latencies, 95.0)

        hard_samples: list[HardSample] = []
        class_counts: dict[str, int] = {}
        for r in filtered:
            dets = [
                d
                for d in r.get("detections", []) or []
                if isinstance(d, dict)
                and "class" in d
                and isinstance(d.get("confidence"), int | float)
            ]
            for d in dets:
                class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

            if not dets:
                hard_samples.append(
                    HardSample(
                        image_id=str(r.get("image_id", "unknown")),
                        timestamp=r.get("timestamp"),
                        num_detections=0,
                        min_confidence=None,
                        reason="no detections",
                    )
                )
                continue

            min_conf = min(float(d["confidence"]) for d in dets)
            if min_conf < inp.low_confidence_threshold:
                hard_samples.append(
                    HardSample(
                        image_id=str(r.get("image_id", "unknown")),
                        timestamp=r.get("timestamp"),
                        num_detections=len(dets),
                        min_confidence=min_conf,
                        reason=(
                            f"weakest detection confidence {min_conf:.2f} "
                            f"< threshold {inp.low_confidence_threshold:.2f}"
                        ),
                    )
                )

        low_confidence_ratio = len(hard_samples) / total
        total_detections = sum(class_counts.values())
        class_distribution = (
            {cls: count / total_detections for cls, count in class_counts.items()}
            if total_detections
            else {}
        )

        drift_score = 0.0
        critical_class_drop_max = 0.0
        new_classes_detected: list[str] = []
        if baseline is not None:
            drift_score = _drift_score(class_distribution, baseline)
            new_classes_detected = sorted(set(class_distribution) - set(baseline))
            for cls in inp.critical_classes:
                drop = baseline.get(cls, 0.0) - class_distribution.get(cls, 0.0)
                critical_class_drop_max = max(critical_class_drop_max, drop)
        elif inp.critical_classes:
            warnings.append(
                "critical_classes given but no baseline_class_distribution_path — "
                "critical_class_drop cannot be computed."
            )

        metrics = {
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency_ms,
            "low_confidence_ratio": low_confidence_ratio,
            "drift_score": drift_score,
            "critical_class_drop_max": critical_class_drop_max,
            "total_detections": float(total_detections),
        }

        triggered_alerts: list[str] = []
        requires_human_review = False
        fired_actions: list[RecommendedAction] = []

        th = inp.thresholds
        if baseline is not None and critical_class_drop_max >= th.critical_class_drop:
            triggered_alerts.append(
                f"critical_class_drop {critical_class_drop_max:.2f} >= "
                f"threshold {th.critical_class_drop:.2f}"
            )
            fired_actions.append(RecommendedAction.MODEL_REVIEW)
            requires_human_review = True
        if baseline is not None and drift_score >= th.drift_score:
            triggered_alerts.append(
                f"drift_score {drift_score:.2f} >= threshold {th.drift_score:.2f}"
            )
            fired_actions.append(RecommendedAction.CREATE_RETRAINING_REQUEST)
            requires_human_review = True
        if low_confidence_ratio >= th.low_confidence_ratio:
            triggered_alerts.append(
                f"low_confidence_ratio {low_confidence_ratio:.2f} >= "
                f"threshold {th.low_confidence_ratio:.2f}"
            )
            fired_actions.append(RecommendedAction.NEED_MORE_DATA)
            requires_human_review = True
        if p95_latency_ms >= th.p95_latency_ms:
            triggered_alerts.append(
                f"p95_latency_ms {p95_latency_ms:.1f} >= threshold {th.p95_latency_ms:.1f}"
            )
            fired_actions.append(RecommendedAction.NOTIFY_OPS)
        if error_rate >= th.error_rate:
            triggered_alerts.append(
                f"error_rate {error_rate:.2f} >= threshold {th.error_rate:.2f}"
            )
            fired_actions.append(RecommendedAction.NOTIFY_OPS)
            requires_human_review = True
        if new_classes_detected:
            triggered_alerts.append(f"new classes detected: {', '.join(new_classes_detected)}")
            requires_human_review = True

        recommended_action = fired_actions[0] if fired_actions else RecommendedAction.NO_ACTION
        drift_detected = RecommendedAction.CREATE_RETRAINING_REQUEST in fired_actions

        hard_samples_path = artifacts_dir / "hard_samples_manifest.json"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        hard_samples_path.write_text(
            json.dumps([h.model_dump() for h in hard_samples], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        status = (
            MonitoringStatus.ALERTS_TRIGGERED if triggered_alerts else MonitoringStatus.COMPLETED
        )
        logger.info(
            "Monitoring run complete",
            extra={
                "endpoint_name": inp.endpoint_name,
                "total_predictions": total,
                "triggered_alerts": len(triggered_alerts),
                "recommended_action": str(recommended_action),
            },
        )

        return MonitoringOutput(
            success=True,
            message=(
                f"Monitored {total} prediction(s) for '{inp.endpoint_name}': "
                f"{len(triggered_alerts)} alert(s) triggered."
                if triggered_alerts
                else f"Monitored {total} prediction(s) for '{inp.endpoint_name}': no issues."
            ),
            status=status,
            endpoint_name=inp.endpoint_name,
            model_version=inp.model_version or None,
            window_start=window_start,
            window_end=window_end,
            total_predictions=total,
            metrics=metrics,
            class_distribution=class_distribution,
            drift_detected=drift_detected,
            new_classes_detected=new_classes_detected,
            triggered_alerts=triggered_alerts,
            recommended_action=recommended_action,
            requires_human_review=requires_human_review,
            hard_samples=hard_samples,
            hard_samples_manifest_path=str(hard_samples_path),
            warnings=warnings,
            artifacts=[str(hard_samples_path)],
        )


def _parse_window(window: str) -> int | None:
    m = _WINDOW_RE.match(window.strip().lower())
    if not m:
        return None
    value, unit = m.groups()
    return int(value) * _UNIT_SECONDS[unit]


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_records(log_path: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    warnings: list[str] = []
    with log_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipping malformed JSON on line {lineno}: {exc}")
                continue
            if not isinstance(record, dict):
                warnings.append(f"Skipping non-object record on line {lineno}.")
                continue
            records.append(record)
    return records, warnings


def _filter_window(
    records: list[dict], window_seconds: int
) -> tuple[list[dict], str | None, str | None]:
    timestamped = [(r, _parse_timestamp(r.get("timestamp"))) for r in records]
    valid = [(r, ts) for r, ts in timestamped if ts is not None]
    if not valid:
        return records, None, None

    window_end = max(ts for _, ts in valid)
    window_start = window_end.fromtimestamp(
        window_end.timestamp() - window_seconds, tz=window_end.tzinfo
    )
    filtered = [r for r, ts in valid if ts >= window_start]
    return filtered, window_start.isoformat(), window_end.isoformat()


def _load_baseline(path: str) -> tuple[dict[str, float] | None, str | None]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Cannot read baseline_class_distribution_path: {exc}"
    if not isinstance(raw, dict) or not raw:
        return None, f"baseline_class_distribution_path is not a non-empty JSON object: {path}"
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        return None, f"baseline_class_distribution_path values sum to <= 0: {path}"
    return {k: float(v) / total for k, v in raw.items()}, None


def _drift_score(current: dict[str, float], baseline: dict[str, float]) -> float:
    classes = set(current) | set(baseline)
    if not classes:
        return 0.0
    return 0.5 * sum(abs(current.get(c, 0.0) - baseline.get(c, 0.0)) for c in classes)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(pct / 100 * len(s)) - 1))
    return s[idx]


def _failed(message: str) -> MonitoringOutput:
    return MonitoringOutput(
        success=False, message=message, status=MonitoringStatus.FAILED, errors=[message]
    )
