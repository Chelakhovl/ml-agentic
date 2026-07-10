"""Label QA Agent — post-labeling quality gate.

Checks label quality after manual labeling or pseudo-labeling: suspicious bbox
geometry, class imbalance, missing label files, and (optionally) disagreement
with a reference YOLO model. Never modifies labels — only reports.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.label_qa import LabelQAInput, LabelQAOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.label_qa_checker import LabelQAChecker
from agentic_mlops.tools.report_writer import ReportWriter


class LabelQAAgent(BaseAgent):
    """Reviews label quality and produces a label_quality_report.

    Output artifacts:
        artifacts_dir/label_quality_report.json
        artifacts_dir/label_quality_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._checker = LabelQAChecker()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: LabelQAInput) -> LabelQAOutput:
        self._log_start(dataset_path=input.dataset_path)

        output = self._checker.check(input)

        json_path, md_path = self._report_writer.write_label_qa_report(
            output=output,
            artifacts_dir=self.artifacts_dir,
        )
        output.report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=str(output.status),
            suspicious_samples=len(output.suspicious_samples),
            label_quality_score=output.label_quality_score,
            report_path=output.report_path,
        )
        return output

    def _log_to_mlflow(self, inp: LabelQAInput, output: LabelQAOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {"label_qa.dataset_path": inp.dataset_path}
        if inp.reference_model_path:
            params["label_qa.reference_model_path"] = inp.reference_model_path
        client.log_params(rid, params)

        metrics: dict[str, float] = {
            "label_qa.quality_score": output.label_quality_score,
            "label_qa.num_images_checked": float(output.num_images_checked),
            "label_qa.num_labels_checked": float(output.num_labels_checked),
            "label_qa.num_suspicious": float(len(output.suspicious_samples)),
        }
        client.log_metrics(rid, metrics)

        client.log_tags(rid, {
            "workflow_step": "label_qa",
            "label_qa_status": str(output.status),
        })

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
