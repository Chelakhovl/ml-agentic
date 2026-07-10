"""Model Decision Agent — turns an evaluation report into an explainable recommendation.

Sits conceptually between EvaluationAgent and HumanApprovalAgent: reads the
evaluation report EvaluationAgent already produced, adds baseline comparison and
runtime (model size / latency) budget checks that no other code path performs, and
writes a decision_report. Never promotes anything itself — approval remains
HumanApprovalAgent's job (H5 gate), matching the spec's "no auto-production-promote"
rule.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.model_decision import ModelDecisionInput, ModelDecisionOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.model_decider import ModelDecider
from agentic_mlops.tools.report_writer import ReportWriter


class ModelDecisionAgent(BaseAgent):
    """Produces a promote/reject/retrain/need_more_data/need_label_review decision.

    Output artifacts:
        artifacts_dir/decision_report.json
        artifacts_dir/decision_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._decider = ModelDecider()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: ModelDecisionInput) -> ModelDecisionOutput:
        self._log_start(evaluation_report_path=input.evaluation_report_path)

        output = self._decider.decide(input)

        json_path, md_path = self._report_writer.write_model_decision_report(
            output, self.artifacts_dir
        )
        output.decision_report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(status=str(output.decision), reasons=len(output.reasons))
        return output

    def _log_to_mlflow(self, inp: ModelDecisionInput, output: ModelDecisionOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        client.log_params(rid, {
            "model_decision.evaluation_report_path": inp.evaluation_report_path,
        })

        metrics: dict[str, float] = {}
        if output.map50_improvement is not None:
            metrics["model_decision.map50_improvement"] = output.map50_improvement
        if output.model_size_mb is not None:
            metrics["model_decision.model_size_mb"] = output.model_size_mb
        if metrics:
            client.log_metrics(rid, metrics)

        client.log_tags(rid, {
            "workflow_step": "model_decision",
            "model_decision": str(output.decision),
        })

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
