"""Evaluation Agent — MVP Agent 3.

Runs YOLO evaluation (or a dry-run) and applies the promotion policy.
Refuses to evaluate when the upstream training status is failed or cancelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.evaluation import (
    EvaluationConfig,
    EvaluationInput,
    EvaluationMetrics,
    EvaluationOutput,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.report_writer import ReportWriter
from agentic_mlops.tools.yolo_evaluator import YoloEvaluator

if TYPE_CHECKING:
    from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner


class EvaluationAgent(BaseAgent):
    """Orchestrates YOLO model evaluation and promotion-policy gating.

    Output artifacts:
        artifacts_dir/evaluation_request.json  (dry-run)
        artifacts_dir/evaluation_report.json
        artifacts_dir/evaluation_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        dry_run_override_metrics: EvaluationMetrics | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
        azure_runner: AzureMLEvaluationRunner | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._evaluator = YoloEvaluator(
            dry_run_override_metrics=dry_run_override_metrics, azure_runner=azure_runner
        )
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: EvaluationInput) -> EvaluationOutput:
        self._log_start(workflow_id=input.workflow_id, mode=input.mode)

        if input.training_status in {"failed", "cancelled"}:
            msg = (
                f"Evaluation blocked: upstream training status is '{input.training_status}'. "
                "Fix training before evaluating."
            )
            self.logger.warning(msg)
            output = EvaluationOutput(
                success=False,
                message=msg,
                mode=input.mode,
                errors=[msg],
            )
            self._report_writer.write_evaluation_report(output, self.artifacts_dir)
            return output

        output = self._evaluator.run(input, self.artifacts_dir)

        json_path, md_path = self._report_writer.write_evaluation_report(output, self.artifacts_dir)
        output.evaluation_report_path = str(md_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(status=output.recommendation, mode=output.mode)
        return output

    def _log_to_mlflow(self, inp: EvaluationInput, output: EvaluationOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "evaluation.runner": str(output.runner or inp.mode),
            "evaluation.model_path": inp.weights_path,
            "evaluation.data_yaml": inp.data_yaml_path,
        }
        if inp.evaluation_config_path:
            try:
                eval_cfg = EvaluationConfig.from_yaml(inp.evaluation_config_path)
                params["evaluation.imgsz"] = str(eval_cfg.imgsz)
                params["evaluation.batch"] = str(eval_cfg.batch)
                params["evaluation.device"] = eval_cfg.device
            except Exception:
                pass
        client.log_params(rid, params)

        m = output.metrics
        metrics: dict[str, float] = {
            "evaluation.map50": m.map50,
            "evaluation.map50_95": m.map50_95,
            "evaluation.precision": m.precision,
            "evaluation.recall": m.recall,
        }
        for cls_name, pcm in m.per_class_metrics.items():
            safe = cls_name.replace(" ", "_")
            metrics[f"evaluation.class.{safe}.precision"] = pcm.precision
            metrics[f"evaluation.class.{safe}.recall"] = pcm.recall
            metrics[f"evaluation.class.{safe}.map50"] = pcm.map50
        client.log_metrics(rid, metrics)

        client.log_tags(
            rid,
            {
                "workflow_step": "evaluation",
                "evaluation_runner": str(output.runner or inp.mode),
                "recommendation": str(output.recommendation or "none"),
            },
        )

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
