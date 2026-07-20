"""Pydantic contracts for the Monitoring Agent.

Two log sources are supported:
  "local"         — reads a local JSON-Lines predictions log (original behaviour)
  "azure_monitor" — queries the Application Insights traces table via
                    ApplicationInsightsLogClient (azure-monitor-query SDK)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .common import ToolResult


class RecommendedAction(StrEnum):
    NO_ACTION = "no_action"
    NOTIFY_OPS = "notify_ops"
    NEED_MORE_DATA = "need_more_data"
    MODEL_REVIEW = "model_review"
    CREATE_RETRAINING_REQUEST = "create_retraining_request"


class MonitoringStatus(StrEnum):
    COMPLETED = "completed"          # no thresholds breached
    ALERTS_TRIGGERED = "alerts_triggered"  # ran fine, but one or more triggers fired
    FAILED = "failed"                # structural problem (bad path, bad window, ...)


class MonitoringThresholds(BaseModel):
    """Mirrors the `triggers:` block in 12_monitoring_agent.md."""

    low_confidence_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    p95_latency_ms: float = Field(default=200.0, gt=0.0)
    critical_class_drop: float = Field(default=0.15, ge=0.0, le=1.0)
    drift_score: float = Field(default=0.30, ge=0.0, le=1.0)
    error_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class MonitoringInput(BaseModel):
    endpoint_name: str
    model_version: str = ""

    # Which source to read inference records from.
    source: Literal["local", "azure_monitor"] = "local"

    # Required when source="local": path to a JSON-Lines predictions log.
    #   {"timestamp": "...", "image_id": "...", "latency_ms": 42.0, "error": false,
    #    "detections": [{"class": "scratch", "confidence": 0.91}, ...]}
    predictions_log_path: str = ""

    # Required when source="azure_monitor": App Insights Application ID (GUID).
    app_insights_workspace_id: str | None = None

    # e.g. "24h", "7d", "30m" — window ends at the latest timestamp in the log
    # (not wall-clock now), so results stay deterministic for a fixed log file.
    monitoring_window: str = "24h"

    # Path to a JSON {"class_name": proportion_or_count, ...} snapshot of the
    # class distribution the model was trained/evaluated against. Optional —
    # drift/critical-class-drop checks are skipped without it.
    baseline_class_distribution_path: str | None = None
    critical_classes: list[str] = Field(default_factory=list)

    # Per-image "weakest detection" confidence floor used for hard-sample mining
    # (same "weakest detection decides the bucket" rule as ConfidenceThresholds
    # in the Annotation Agent). Images with zero detections also count as hard.
    low_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    thresholds: MonitoringThresholds = Field(default_factory=MonitoringThresholds)


class HardSample(BaseModel):
    image_id: str
    timestamp: str | None = None
    num_detections: int = 0
    min_confidence: float | None = None
    reason: str


class MonitoringOutput(ToolResult):
    status: MonitoringStatus = MonitoringStatus.FAILED
    endpoint_name: str | None = None
    model_version: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    total_predictions: int = 0

    metrics: dict[str, float] = Field(default_factory=dict)
    class_distribution: dict[str, float] = Field(default_factory=dict)

    drift_detected: bool = False
    new_classes_detected: list[str] = Field(default_factory=list)
    triggered_alerts: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.NO_ACTION
    requires_human_review: bool = False

    hard_samples: list[HardSample] = Field(default_factory=list)
    hard_samples_manifest_path: str | None = None
    monitoring_report_path: str | None = None
