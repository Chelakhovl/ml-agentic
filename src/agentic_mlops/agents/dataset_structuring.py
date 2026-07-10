"""Dataset Structuring Agent — converts raw/curated data into a YOLO layout.

Creates images/{train,val,test} + labels/{train,val,test}, a valid data.yaml,
and a split_report. Converts COCO labels to YOLO when needed. Supports a
grouped split strategy so frames from the same video/source never leak
across train/val/test.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.dataset_structuring import (
    DatasetStructuringInput,
    DatasetStructuringOutput,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.dataset_structurer import DatasetStructurer
from agentic_mlops.tools.report_writer import ReportWriter


class DatasetStructuringAgent(BaseAgent):
    """Structures raw data into a YOLO-compatible dataset layout.

    Output artifacts:
        <output_dataset_path>/data.yaml
        artifacts_dir/split_report.json
        artifacts_dir/split_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._structurer = DatasetStructurer()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: DatasetStructuringInput) -> DatasetStructuringOutput:
        self._log_start(
            raw_data_path=input.raw_data_path,
            output_dataset_path=input.output_dataset_path,
            split_strategy=input.split_strategy,
        )

        output = self._structurer.structure(input)

        json_path, md_path = self._report_writer.write_dataset_structuring_report(
            output=output,
            artifacts_dir=self.artifacts_dir,
        )
        output.split_report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status="success" if output.success else "failed",
            num_images=output.num_images,
            split_counts=output.split_counts,
        )
        return output

    def _log_to_mlflow(
        self, inp: DatasetStructuringInput, output: DatasetStructuringOutput
    ) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "dataset_structuring.raw_data_path": inp.raw_data_path,
            "dataset_structuring.label_format": str(inp.label_format),
            "dataset_structuring.split_strategy": str(inp.split_strategy),
        }
        client.log_params(rid, params)

        metrics: dict[str, float] = {"dataset_structuring.num_images": float(output.num_images)}
        for split, count in output.split_counts.items():
            metrics[f"dataset_structuring.split.{split}"] = float(count)
        client.log_metrics(rid, metrics)

        client.log_tags(rid, {"workflow_step": "dataset_structuring"})

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
