"""Annotation / Pseudo-label Agent — speeds up labeling with an approved YOLO model.

Runs YOLO predict over unlabeled/partially-labeled images, writes candidate
pseudo-labels, and routes each image into a confidence bucket so uncertain and
hard samples land in a review queue for a human. Never treats pseudo-labels
as final, and never overwrites existing human labels.

When an ``annotation_client`` is injected the agent also stages medium/low
review-queue images as a new annotation task in CVAT or Label Studio after a
successful pseudo-label run.  Upload failures are logged as warnings and never
abort the agent run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.annotation import AnnotationInput, AnnotationOutput
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.pseudo_labeler import PseudoLabeler
from agentic_mlops.tools.report_writer import ReportWriter

if TYPE_CHECKING:
    from agentic_mlops.integrations.annotation_client import AnnotationClientBase

logger = logging.getLogger(__name__)


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
        annotation_client: AnnotationClientBase | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._labeler = PseudoLabeler()
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id
        self._annotation_client = annotation_client

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

        # Stage review-queue images in the external annotation tool (best-effort)
        if output.success and self._annotation_client is not None:
            self._stage_review_queue(input, output)

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

    def _stage_review_queue(self, inp: AnnotationInput, output: AnnotationOutput) -> None:
        """Upload medium/low review-queue images to the external annotation tool.

        Best-effort: any failure is logged as a warning, never re-raised.
        On success, adds ``annotation_task_url`` to ``output.artifacts``.
        """
        from agentic_mlops.contracts.annotation_client import UploadTaskInput

        review_images: list[Path] = []

        # Collect review-queue paths from the records (medium + low buckets)
        for record in output.records:
            if record.bucket in ("medium", "low") and record.image:
                img = Path(record.image)
                if img.exists():
                    review_images.append(img)

        # Fall back to parsing review_queue.json when records are not populated
        if not review_images and output.review_queue_path:
            rq = Path(output.review_queue_path)
            if rq.exists():
                try:
                    queue_data = json.loads(rq.read_text(encoding="utf-8"))
                    for entry in queue_data if isinstance(queue_data, list) else []:
                        img = Path(entry.get("image", ""))
                        if img.exists():
                            review_images.append(img)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not parse review_queue.json: %s", exc)

        if not review_images:
            return

        task_name = f"review_{Path(inp.images_path).name}"
        upload_inp = UploadTaskInput(
            task_name=task_name,
            image_paths=review_images,
            labels=[],  # labels are configured inside the annotation tool project
        )
        try:
            upload_out = self._annotation_client.upload_task(upload_inp)  # type: ignore[union-attr]
            if upload_out.success and upload_out.task is not None:
                url = upload_out.task.url
                if url and url not in output.artifacts:
                    output.artifacts.append(url)
                    output.metadata["annotation_task_url"] = url
                logger.info(
                    "Staged %d review-queue images in annotation tool (task_id=%s)",
                    len(review_images),
                    upload_out.task.task_id,
                )
            else:
                logger.warning("Annotation tool upload returned failure: %s", upload_out.message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Annotation tool upload failed: %s", exc)
