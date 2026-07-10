"""Training runners: local YOLO (Ultralytics) and Azure ML SDK v2.

LocalYOLOTrainingRunner: local_train mode only — no Azure ML, no MLflow.
AzureMLTrainingRunner: azure_train mode — submits CommandJob, streams logs,
    downloads outputs, returns the same TrainingOutput contract.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_mlops.contracts.training import (
    TrainingArtifact,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
    TrainingOutput,
)
from agentic_mlops.observability.logging import get_logger

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = get_logger(__name__)


@contextmanager
def _suppress_ultralytics_mlflow():
    """Temporarily disable Ultralytics built-in MLflow autologging.

    Saves the current 'mlflow' setting, disables it for the duration of the
    block, then restores the original value in a finally clause. Falls back
    silently if ultralytics is not importable or settings access fails.
    """
    ult_settings = None
    prev_mlflow = None
    try:
        from ultralytics import settings as _s  # noqa: PLC0415

        ult_settings = _s
        prev_mlflow = _s.get("mlflow", True)
        _s.update({"mlflow": False})
    except Exception:
        pass
    try:
        yield
    finally:
        if ult_settings is not None:
            try:
                ult_settings.update({"mlflow": prev_mlflow})
            except Exception:
                pass


def _import_yolo():
    """Return the YOLO class from ultralytics, or raise if not installed."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc


class LocalYOLOTrainingRunner:
    """Runs YOLO training locally using Ultralytics.

    Collects best.pt, last.pt, and results.csv into artifacts_dir.
    Writes training_output.json. Does not call Azure ML or MLflow.
    """

    def run(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        cfg = inp.training_config

        logger.info(
            "LocalYOLOTrainingRunner starting",
            extra={"job_id": job_id, "model": cfg.model, "epochs": cfg.epochs},
        )

        YOLO = _import_yolo()

        with _suppress_ultralytics_mlflow():
            try:
                model = YOLO(cfg.model)
                model.train(
                    data=inp.data_yaml_path,
                    epochs=cfg.epochs,
                    imgsz=cfg.imgsz,
                    batch=cfg.batch,
                    optimizer=cfg.optimizer,
                    patience=cfg.patience,
                    project=str(artifacts_dir / (cfg.project or "train")),
                    name=cfg.name or "run",
                )
                save_dir = Path(model.trainer.save_dir)
            except Exception as exc:
                logger.error("YOLO training failed", extra={"error": str(exc)})
                return TrainingOutput(
                    success=False,
                    message=f"Local YOLO training failed: {exc}",
                    job_id=job_id,
                    job_status=TrainingJobStatus.FAILED,
                    mode=TrainingMode.LOCAL_TRAIN,
                    errors=[str(exc)],
                )

        training_artifacts: list[TrainingArtifact] = []
        best_weights_path: str | None = None

        for weight_name in ("best.pt", "last.pt"):
            src = save_dir / "weights" / weight_name
            if src.exists():
                dst = artifacts_dir / weight_name
                shutil.copy2(src, dst)
                if weight_name == "best.pt":
                    best_weights_path = str(dst)
                training_artifacts.append(
                    TrainingArtifact(name=weight_name, path=str(dst), artifact_type="weights")
                )
                logger.info("Collected weight", extra={"weight_name": weight_name, "dst": str(dst)})

        results_csv_src = save_dir / "results.csv"
        if results_csv_src.exists():
            dst = artifacts_dir / "results.csv"
            shutil.copy2(results_csv_src, dst)
            training_artifacts.append(
                TrainingArtifact(name="results.csv", path=str(dst), artifact_type="log")
            )
            logger.info("Collected results.csv", extra={"dst": str(dst)})

        output = TrainingOutput(
            success=True,
            message=f"Local YOLO training completed. (job_id={job_id})",
            job_id=job_id,
            mlflow_run_id=None,
            best_weights_path=best_weights_path,
            training_plan_path=None,
            job_status=TrainingJobStatus.COMPLETED,
            mode=TrainingMode.LOCAL_TRAIN,
            training_artifacts=training_artifacts,
            artifacts=[a.path for a in training_artifacts],
        )

        output_json_path = artifacts_dir / "training_output.json"
        output_json_path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        output.artifacts.append(str(output_json_path))

        logger.info(
            "LocalYOLOTrainingRunner done",
            extra={
                "job_id": job_id,
                "artifact_count": len(training_artifacts),
                "best_weights": best_weights_path,
            },
        )

        return output


# ── Azure ML runner ────────────────────────────────────────────────────────────

_AZURE_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested", "NotResponding"}


class AzureMLTrainingRunner:
    """Submits a YOLO CommandJob to Azure ML, streams logs, downloads outputs.

    Inject FakeAzureMLClientFactory for tests — no real Azure calls made.
    """

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )
            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    def run(self, inp: TrainingInput, artifacts_dir: Path) -> TrainingOutput:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cfg = inp.training_config
        started_at = datetime.now(tz=UTC).isoformat()

        logger.info(
            "AzureMLTrainingRunner starting",
            extra={"model": cfg.model, "compute": self._config.compute_name},
        )

        try:
            ml_client = self._factory.create(self._config)
            job = self._build_job(inp, cfg)
            self._write_job_request(job, artifacts_dir)
            submitted = ml_client.jobs.create_or_update(job)
            azure_name: str = submitted.name

            logger.info("Azure ML job submitted", extra={"azure_job_name": azure_name})

            if self._config.job.stream_logs:
                ml_client.jobs.stream(azure_name)
                final = ml_client.jobs.get(azure_name)
            else:
                final = self._wait_for_completion(ml_client, azure_name)

            completed_at = datetime.now(tz=UTC).isoformat()

            if final.status != "Completed":
                return self._failed_output(
                    cfg,
                    azure_name,
                    final.status,
                    f"Azure ML job ended with status '{final.status}'",
                    started_at,
                    completed_at,
                )

            if self._config.job.download_outputs:
                ml_client.jobs.download(
                    azure_name,
                    output_name=self._config.job.output_name,
                    download_path=str(artifacts_dir),
                )

            best_pt = self._find_and_promote(artifacts_dir, "best.pt")
            if best_pt is None:
                return self._failed_output(
                    cfg,
                    azure_name,
                    "Completed",
                    "best.pt not found in downloaded outputs",
                    started_at,
                    completed_at,
                )

            training_artifacts = self._collect_artifacts(artifacts_dir)
            output = TrainingOutput(
                success=True,
                message=f"Azure ML job '{azure_name}' completed successfully.",
                job_id=azure_name,
                job_status=TrainingJobStatus.COMPLETED,
                best_weights_path=str(best_pt),
                mode=TrainingMode.AZURE_TRAIN,
                azure_job_name=azure_name,
                azure_job_status="Completed",
                azure_studio_url=getattr(submitted, "studio_url", None),
                azure_compute_name=self._config.compute_name,
                azure_experiment_name=self._config.experiment_name,
                azure_output_name=self._config.job.output_name,
                remote_started_at=started_at,
                remote_completed_at=completed_at,
                training_artifacts=training_artifacts,
                artifacts=[a.path for a in training_artifacts],
            )
            self._write_output(output, artifacts_dir)
            return output

        except Exception as exc:
            logger.error("AzureMLTrainingRunner failed", extra={"error": str(exc)})
            return TrainingOutput(
                success=False,
                message=str(exc),
                mode=TrainingMode.AZURE_TRAIN,
                job_status=TrainingJobStatus.FAILED,
                errors=[str(exc)],
            )

    # ── Job construction ───────────────────────────────────────────────────────

    def _build_job(self, inp: TrainingInput, cfg: Any) -> Any:
        from azure.ai.ml import Input, Output, command  # noqa: PLC0415
        from azure.ai.ml.constants import AssetTypes, InputOutputModes  # noqa: PLC0415

        dataset_path = self._config.data.asset_uri or inp.dataset_path
        input_mode = (
            InputOutputModes.RO_MOUNT
            if self._config.data.input_mode == "ro_mount"
            else InputOutputModes.DOWNLOAD
        )
        dataset_input = Input(type=AssetTypes.URI_FOLDER, path=dataset_path, mode=input_mode)
        datayaml_input = Input(type=AssetTypes.URI_FILE, path=inp.data_yaml_path)
        model_output = Output(type=AssetTypes.URI_FOLDER)

        code_dir = str(Path(__file__).parent.parent / "azure_jobs")
        cmd = (
            "python train_yolo.py"
            " --dataset-path ${{inputs.dataset}}"
            " --data-yaml ${{inputs.data_yaml}}"
            f" --model {cfg.model}"
            f" --epochs {cfg.epochs}"
            f" --imgsz {cfg.imgsz}"
            f" --batch {cfg.batch}"
            f" --patience {cfg.patience}"
            " --output-dir ${{outputs.model_output}}"
        )

        env = self._build_environment()

        return command(
            code=code_dir,
            command=cmd,
            inputs={"dataset": dataset_input, "data_yaml": datayaml_input},
            outputs={self._config.job.output_name: model_output},
            environment=env,
            compute=self._config.compute_name,
            experiment_name=self._config.experiment_name,
            display_name=f"agentic-mlops-{cfg.name}",
            tags=self._config.job.tags,
        )

    def _build_environment(self) -> Any:
        from azure.ai.ml.entities import Environment  # noqa: PLC0415

        env_cfg = self._config.environment
        if env_cfg.mode.value == "registered":
            return env_cfg.registered_environment
        # inline
        return Environment(image=env_cfg.base_image, conda_file=env_cfg.conda_file)

    # ── Polling ────────────────────────────────────────────────────────────────

    def _wait_for_completion(self, ml_client: Any, azure_name: str) -> Any:
        deadline = time.monotonic() + self._config.job.timeout_minutes * 60
        poll_interval = 30

        while time.monotonic() < deadline:
            job = ml_client.jobs.get(azure_name)
            if job.status in _AZURE_TERMINAL_STATES:
                return job
            logger.info(
                "Azure ML job polling",
                extra={"azure_job_name": azure_name, "status": job.status},
            )
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0, remaining)))

        logger.warning(
            "Azure ML job timed out",
            extra={
                "azure_job_name": azure_name,
                "timeout_minutes": self._config.job.timeout_minutes,
            },
        )
        try:
            ml_client.jobs.cancel(azure_name)
        except Exception:
            pass
        return ml_client.jobs.get(azure_name)

    # ── Artifact helpers ───────────────────────────────────────────────────────

    def _find_and_promote(self, artifacts_dir: Path, filename: str) -> Path | None:
        """Find a file anywhere under artifacts_dir; copy to artifacts_dir root if in subdir."""
        matches = list(artifacts_dir.rglob(filename))
        if not matches:
            return None
        found = matches[0]
        target = artifacts_dir / filename
        if found != target:
            shutil.copy2(found, target)
        return target

    def _collect_artifacts(self, artifacts_dir: Path) -> list[TrainingArtifact]:
        # Promote all key files to artifacts_dir root (they may be in subdirectories)
        for fname in ("last.pt", "results.csv", "args.yaml"):
            self._find_and_promote(artifacts_dir, fname)
        artifacts: list[TrainingArtifact] = []
        for weight_name in ("best.pt", "last.pt"):
            p = artifacts_dir / weight_name
            if p.exists():
                artifacts.append(
                    TrainingArtifact(name=weight_name, path=str(p), artifact_type="weights")
                )
        for fname, atype in (("results.csv", "log"), ("args.yaml", "config")):
            p = artifacts_dir / fname
            if p.exists():
                artifacts.append(TrainingArtifact(name=fname, path=str(p), artifact_type=atype))
        return artifacts

    # ── Output helpers ─────────────────────────────────────────────────────────

    def _failed_output(
        self,
        cfg: Any,
        azure_name: str | None,
        azure_status: str,
        message: str,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> TrainingOutput:
        return TrainingOutput(
            success=False,
            message=message,
            job_id=azure_name,
            mode=TrainingMode.AZURE_TRAIN,
            job_status=TrainingJobStatus.FAILED,
            azure_job_name=azure_name,
            azure_job_status=azure_status,
            azure_compute_name=self._config.compute_name,
            azure_experiment_name=self._config.experiment_name,
            remote_started_at=started_at,
            remote_completed_at=completed_at,
            errors=[message],
        )

    def _write_job_request(self, job: Any, artifacts_dir: Path) -> None:
        payload: dict[str, Any] = {
            "compute": self._config.compute_name,
            "experiment_name": self._config.experiment_name,
            "tags": self._config.job.tags,
            "output_name": self._config.job.output_name,
        }
        dn = getattr(job, "display_name", None)
        if isinstance(dn, str):
            payload["display_name"] = dn
        p = artifacts_dir / "azure_job_request.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_output(self, output: TrainingOutput, artifacts_dir: Path) -> None:
        p = artifacts_dir / "training_output.json"
        p.write_text(json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8")
        if str(p) not in output.artifacts:
            output.artifacts.append(str(p))
