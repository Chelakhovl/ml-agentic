"""YOLO trainer tool — handles local dry-run, local train, and Azure ML train modes.

local_dry_run : writes training_request.json, no YOLO or Azure call.
local_train   : delegates to LocalYOLOTrainingRunner (requires ultralytics).
azure_train   : delegates to AzureMLTrainingRunner when injected; else falls back to stub.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.contracts.training import (
    TrainingArtifact,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
    TrainingOutput,
)
from agentic_mlops.integrations.azure_ml_client import AzureMLTrainingClientBase
from agentic_mlops.integrations.mlflow_client import MLflowClientBase
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.tools.training_runner import LocalYOLOTrainingRunner

if TYPE_CHECKING:
    from agentic_mlops.tools.training_runner import AzureMLTrainingRunner

logger = get_logger(__name__)


class YoloTrainer:
    """Executes YOLO training in the requested mode.

    Inject FakeAzureMLTrainingClient / FakeMLflowClient for tests.
    Inject AzureMLTrainingRunner via azure_runner to enable SDK v2 Azure jobs.
    """

    def __init__(
        self,
        azure_client: AzureMLTrainingClientBase,
        mlflow_client: MLflowClientBase,
        azure_runner: AzureMLTrainingRunner | None = None,
    ) -> None:
        self._azure = azure_client
        self._mlflow = mlflow_client
        self._azure_runner = azure_runner

    def run(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mode = inp.training_config.mode

        logger.info("YoloTrainer starting", extra={"mode": mode, "workflow_id": inp.workflow_id})

        if mode == TrainingMode.LOCAL_DRY_RUN:
            return self._dry_run(inp, artifacts_dir)
        if mode == TrainingMode.LOCAL_TRAIN:
            return self._local_train(inp, artifacts_dir)
        if mode == TrainingMode.AZURE_TRAIN:
            return self._azure_train(inp, artifacts_dir)

        raise ValueError(f"Unknown training mode: {mode}")

    # ── Local train ───────────────────────────────────────────────────────────

    def _local_train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        return LocalYOLOTrainingRunner().run(inp, artifacts_dir)

    # ── Dry-run ────────────────────────────────────────────────────────────────

    def _dry_run(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        job_id = f"dry_run_{uuid.uuid4().hex[:8]}"
        cfg = inp.training_config

        plan: dict = {
            "job_id": job_id,
            "workflow_id": inp.workflow_id,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "mode": TrainingMode.LOCAL_DRY_RUN,
            "dataset_path": inp.dataset_path,
            "data_yaml_path": inp.data_yaml_path,
            "training_config": cfg.model_dump(),
            "would_run_command": (
                f"yolo train "
                f"model={cfg.model} "
                f"data={inp.data_yaml_path} "
                f"epochs={cfg.epochs} "
                f"imgsz={cfg.imgsz} "
                f"batch={cfg.batch} "
                f"optimizer={cfg.optimizer} "
                f"patience={cfg.patience} "
                f"project={cfg.project} "
                f"name={cfg.name}"
            ),
            "note": "Dry-run: no actual training was performed.",
        }

        plan_path = artifacts_dir / "training_request.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        artifact = TrainingArtifact(
            name="training_request",
            path=str(plan_path),
            artifact_type="plan",
        )

        logger.info(
            "Dry-run plan written",
            extra={"path": str(plan_path), "job_id": job_id},
        )

        return TrainingOutput(
            success=True,
            message=f"Dry-run plan created. No training performed. (job_id={job_id})",
            job_id=job_id,
            mlflow_run_id=None,
            best_weights_path=None,
            training_plan_path=str(plan_path),
            job_status=TrainingJobStatus.COMPLETED,
            mode=TrainingMode.LOCAL_DRY_RUN,
            training_artifacts=[artifact],
            artifacts=[str(plan_path)],
        )

    # ── Azure train ────────────────────────────────────────────────────────────

    def _azure_train(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        if self._azure_runner is not None:
            return self._azure_runner.run(inp, artifacts_dir)
        # Legacy stub path — raises NotImplementedError unless a real client was injected

        cfg = inp.training_config
        mlflow_run_id: str | None = None

        try:
            mlflow_run_id = self._mlflow.start_run(
                experiment_name=cfg.project,
                run_name=cfg.name,
            )
            self._mlflow.log_params(mlflow_run_id, cfg.model_dump())
        except NotImplementedError:
            logger.warning("MLflow tracking skipped (stub client).")

        job_id = self._azure.submit_training_job(
            config=cfg,
            dataset_path=inp.dataset_path,
            data_yaml_path=inp.data_yaml_path,
        )

        status = self._azure.get_job_status(job_id)

        artifact_paths = self._azure.get_job_artifacts(job_id, str(artifacts_dir))

        if mlflow_run_id:
            try:
                self._mlflow.end_run(mlflow_run_id)
            except NotImplementedError:
                pass

        training_artifacts = [
            TrainingArtifact(name=Path(p).name, path=p, artifact_type="weights")
            for p in artifact_paths
        ]

        return TrainingOutput(
            success=status == TrainingJobStatus.COMPLETED,
            message=f"Azure ML job {job_id} finished with status: {status}.",
            job_id=job_id,
            mlflow_run_id=mlflow_run_id,
            job_status=status,
            mode=TrainingMode.AZURE_TRAIN,
            training_artifacts=training_artifacts,
            artifacts=artifact_paths,
        )
