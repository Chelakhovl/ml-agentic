"""Data Intake Agent — accepts new raw images and produces a dataset_manifest.

Runs when a new raw dataset arrives, monitoring finds hard samples, or a
model-decision step requests more data. Checks basic file quality (format,
corruption, duplicates) and gates on a human source approval when the data
source/provenance is unclear.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.data_intake import DataIntakeInput, DataIntakeOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.data_intake_scanner import DataIntakeScanner
from agentic_mlops.tools.report_writer import ReportWriter


class DataIntakeAgent(BaseAgent):
    """Scans raw data and produces dataset_manifest.json.

    Output artifacts:
        artifacts_dir/dataset_manifest.json
        artifacts_dir/dataset_manifest.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._scanner = DataIntakeScanner()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: DataIntakeInput) -> DataIntakeOutput:
        self._log_start(raw_data_path=input.raw_data_path, dataset_name=input.dataset_name)

        output = self._scanner.scan(input)

        json_path, md_path = self._report_writer.write_data_intake_report(
            output=output,
            artifacts_dir=self.artifacts_dir,
        )
        output.manifest_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=str(output.status),
            num_files=output.num_files,
            valid_images=output.valid_images,
            corrupted=len(output.corrupted_images),
            duplicate_groups=len(output.duplicate_groups),
        )
        return output

    def _log_to_mlflow(self, inp: DataIntakeInput, output: DataIntakeOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "data_intake.dataset_name": inp.dataset_name,
            "data_intake.raw_data_path": inp.raw_data_path,
        }
        if inp.source:
            params["data_intake.source"] = inp.source
        client.log_params(rid, params)

        metrics: dict[str, float] = {
            "data_intake.num_files": float(output.num_files),
            "data_intake.valid_images": float(output.valid_images),
            "data_intake.corrupted_images": float(len(output.corrupted_images)),
            "data_intake.duplicate_groups": float(len(output.duplicate_groups)),
        }
        client.log_metrics(rid, metrics)

        client.log_tags(
            rid,
            {
                "workflow_step": "data_intake",
                "data_intake_status": str(output.status),
            },
        )

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
