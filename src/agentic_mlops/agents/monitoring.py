"""Monitoring Agent — watches a deployed model's predictions log for drift/quality issues.

Supports two inference log sources (set via MonitoringInput.source):
  "local"         — reads a local JSON-Lines predictions log (default)
  "azure_monitor" — queries Application Insights via ApplicationInsightsLogClient

Computes latency/error/confidence/drift signals, mines hard samples for human
review, and recommends an action. Never triggers retraining itself —
only recommends — matching the "no auto-promote" rule every other agent in
this codebase already follows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.monitoring import MonitoringInput, MonitoringOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.monitor import ModelMonitor
from agentic_mlops.tools.report_writer import ReportWriter

if TYPE_CHECKING:
    from agentic_mlops.integrations.appinsights_log_client import InferenceLogClient


class MonitoringAgent(BaseAgent):
    """Analyzes a predictions log and produces a monitoring_report + hard_samples_manifest.

    Pass ``log_client`` to inject a custom InferenceLogClient (e.g.
    ApplicationInsightsLogClient or FakeInferenceLogClient for tests).
    When None, the client is resolved from MonitoringInput.source at run time.

    Output artifacts:
        artifacts_dir/hard_samples_manifest.json
        artifacts_dir/monitoring_report.json
        artifacts_dir/monitoring_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        monitor: ModelMonitor | None = None,
        log_client: InferenceLogClient | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._monitor = monitor or ModelMonitor(log_client=log_client)
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: MonitoringInput) -> MonitoringOutput:
        self._log_start(endpoint_name=input.endpoint_name, window=input.monitoring_window)

        output = self._monitor.run(input, self.artifacts_dir)

        json_path, md_path = self._report_writer.write_monitoring_report(output, self.artifacts_dir)
        output.monitoring_report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id and output.success:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=str(output.status),
            recommended_action=str(output.recommended_action),
            triggered_alerts=str(len(output.triggered_alerts)),
        )
        return output

    def _log_to_mlflow(self, inp: MonitoringInput, output: MonitoringOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        client.log_params(
            rid,
            {
                "monitoring.endpoint_name": inp.endpoint_name,
                "monitoring.window": inp.monitoring_window,
                "monitoring.recommended_action": str(output.recommended_action),
            },
        )
        client.log_metrics(rid, {f"monitoring.{k}": v for k, v in output.metrics.items()})
        client.log_tags(
            rid,
            {
                "workflow_step": "monitoring",
                "monitoring_status": str(output.status),
                "drift_detected": str(output.drift_detected),
            },
        )
        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
