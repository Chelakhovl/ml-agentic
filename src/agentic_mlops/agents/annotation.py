"""Annotation / Pseudo-label Agent — speeds up labeling with an approved YOLO model.

Runs YOLO predict over unlabeled/partially-labeled images, writes candidate
pseudo-labels, and routes each image into a confidence bucket so uncertain and
hard samples land in a review queue for a human. Never treats pseudo-labels
as final, and never overwrites existing human labels.
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.annotation import AnnotationInput, AnnotationOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.pseudo_labeler import PseudoLabeler
from agentic_mlops.tools.report_writer import ReportWriter


class AnnotationAgent(BaseAgent):
    """Pre-labels images with a YOLO model and produces a pseudo_label_report.

    Output artifacts:
        artifacts_dir/pseudo_labels/*.txt
        artifacts_dir/review_queue.json
        artifacts_dir/pseudo_label_report.json
        artifacts_dir/pseudo_label_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._labeler = PseudoLabeler()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: AnnotationInput) -> AnnotationOutput:
        self._log_start(images_path=input.images_path, model_path=input.model_path)

        output = self._labeler.run(input, self.artifacts_dir)

        json_path, md_path = self._report_writer.write_annotation_report(
            output=output,
            artifacts_dir=self.artifacts_dir,
        )
        output.report_path = str(json_path)
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)
        if output.review_queue_path and output.review_queue_path not in output.artifacts:
            output.artifacts.append(output.review_queue_path)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status="success" if output.success else "failed",
            high=output.high_confidence_count,
            medium=output.medium_confidence_count,
            low=output.low_confidence_count,
        )
        return output

    def _log_to_mlflow(self, inp: AnnotationInput, output: AnnotationOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "annotation.images_path": inp.images_path,
            "annotation.model_path": inp.model_path,
            "annotation.auto_candidate_threshold": str(inp.confidence_thresholds.auto_candidate),
            "annotation.human_review_threshold": str(inp.confidence_thresholds.human_review),
        }
        client.log_params(rid, params)

        metrics: dict[str, float] = {
            "annotation.high_confidence_count": float(output.high_confidence_count),
            "annotation.medium_confidence_count": float(output.medium_confidence_count),
            "annotation.low_confidence_count": float(output.low_confidence_count),
            "annotation.num_images_processed": float(output.num_images_processed),
            "annotation.num_images_skipped_existing": float(output.num_images_skipped_existing),
        }
        client.log_metrics(rid, metrics)

        client.log_tags(rid, {"workflow_step": "annotation"})

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
