"""Dataset Validation Agent — MVP Agent 1.

Validates a YOLO dataset, writes quality reports, and determines
whether training can proceed.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.datasets import DatasetValidationInput, DatasetValidationOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.dataset_validator import DatasetValidator
from agentic_mlops.tools.report_writer import ReportWriter


class DatasetValidationAgent(BaseAgent):
    """Validates a YOLO dataset and produces a quality report.

    Output artifacts:
        artifacts_dir/dataset_quality_report.json
        artifacts_dir/dataset_quality_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._validator = DatasetValidator()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: DatasetValidationInput) -> DatasetValidationOutput:
        self._log_start(dataset_path=input.dataset_path)

        output = self._validator.validate(input)

        json_path, md_path = self._report_writer.write_dataset_validation_report(
            output=output,
            artifacts_dir=self.artifacts_dir,
        )

        output.report_path = str(json_path)
        output.artifacts = [str(json_path), str(md_path)]

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=output.status,
            blocking_issues=len(output.blocking_issues),
            warnings=len(output.warnings),
            report_path=output.report_path,
        )
        return output

    def _log_to_mlflow(
        self, inp: DatasetValidationInput, output: DatasetValidationOutput
    ) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {"validation.dataset_path": inp.dataset_path}
        if inp.data_yaml_path:
            params["validation.data_yaml"] = inp.data_yaml_path
        num_classes = len(output.class_distribution)
        if num_classes:
            params["validation.num_classes"] = str(num_classes)
        client.log_params(rid, params)

        metrics: dict[str, float] = {
            "validation.total_images": float(output.num_images),
            "validation.total_label_files": float(output.num_labels),
            "validation.blocking_issue_count": float(len(output.blocking_issues)),
            "validation.warning_count": float(len(output.warnings)),
        }
        client.log_metrics(rid, metrics)

        client.log_tags(rid, {
            "workflow_step": "dataset_validation",
            "validation_status": output.status,
        })

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
