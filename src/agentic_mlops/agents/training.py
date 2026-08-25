"""Training Agent — MVP Agent 2.

Validates dataset status, builds a training request, and delegates to YoloTrainer.
In local_dry_run mode no YOLO or Azure call is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.training import (
    TrainingInput,
    TrainingJobStatus,
    TrainingOutput,
    training_mode_to_runner,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.report_writer import ReportWriter
from agentic_mlops.tools.yolo_trainer import YoloTrainer

if TYPE_CHECKING:
    from agentic_mlops.tools.training_runner import AzureMLTrainingRunner


class TrainingAgent(BaseAgent):
    """Orchestrates a YOLO training job.

    Refuses to start training when the upstream dataset validation failed.
    Supports dry-run, local, and Azure ML training modes.

    Output artifacts:
        artifacts_dir/training_request.json  (dry-run)
        artifacts_dir/training_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
        azure_runner: AzureMLTrainingRunner | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        # mlflow_client/mlflow_run_id are the parent workflow run for step-level logging,
        # used by _log_to_mlflow() below — YoloTrainer itself does no MLflow logging.
        self._trainer = YoloTrainer(azure_runner=azure_runner)
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: TrainingInput) -> TrainingOutput:
        self._log_start(
            workflow_id=input.workflow_id,
            mode=input.training_config.mode,
            dataset_validation_status=input.dataset_validation_status,
        )

        # Gate: refuse to train on a failed dataset
        if input.dataset_validation_status == "failed":
            msg = (
                "Training blocked: dataset validation status is 'failed'. "
                "Fix all blocking issues before training."
            )
            self.logger.warning(msg)
            output = TrainingOutput(
                success=False,
                message=msg,
                job_status=TrainingJobStatus.CANCELLED,
                mode=input.training_config.mode,
                errors=[msg],
            )
            self._report_writer.write_training_report(output, self.artifacts_dir)
            return output

        output = self._trainer.run(input, self.artifacts_dir)

        report_path = self._report_writer.write_training_report(output, self.artifacts_dir)
        output.report_path = str(report_path)
        if str(report_path) not in output.artifacts:
            output.artifacts.append(str(report_path))

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=output.job_status,
            job_id=output.job_id,
            mode=output.mode,
        )
        return output

    def _log_to_mlflow(self, inp: TrainingInput, output: TrainingOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        cfg = inp.training_config
        runner_name = training_mode_to_runner(cfg.mode)
        params: dict[str, str] = {
            "training.runner": runner_name,
            "training.model": cfg.model,
            "training.epochs": str(cfg.epochs),
            "training.imgsz": str(cfg.imgsz),
            "training.batch": str(cfg.batch),
            "training.project": cfg.project,
            "training.name": cfg.name,
            "training.data_yaml": inp.data_yaml_path,
        }
        client.log_params(rid, params)

        client.log_tags(
            rid,
            {
                "workflow_step": "training",
                "training_runner": runner_name,
                "training_status": str(output.job_status),
            },
        )

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
