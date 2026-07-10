"""YOLO trainer tool — handles local dry-run, local train, and Azure ML train modes.

local_dry_run : writes training_request.json, no YOLO or Azure call.
local_train   : delegates to LocalYOLOTrainingRunner (requires ultralytics).
azure_train   : delegates to an injected AzureMLTrainingRunner (SDK v2).
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
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.tools.training_runner import LocalYOLOTrainingRunner

if TYPE_CHECKING:
    from agentic_mlops.tools.training_runner import AzureMLTrainingRunner

logger = get_logger(__name__)


class YoloTrainer:
    """Executes YOLO training in the requested mode.

    Inject AzureMLTrainingRunner via azure_runner to enable Azure ML SDK v2 jobs;
    azure_train mode raises RuntimeError without one.
    """

    def __init__(
        self,
        azure_runner: AzureMLTrainingRunner | None = None,
    ) -> None:
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
        if self._azure_runner is None:
            raise RuntimeError(
                "Azure ML training requires an AzureMLTrainingRunner — "
                "pass --azure-config on the CLI or inject azure_runner."
            )
        return self._azure_runner.run(inp, artifacts_dir)
