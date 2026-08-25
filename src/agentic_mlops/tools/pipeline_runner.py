"""Azure ML Pipeline runner: submits a 2-step train+eval PipelineJob.

AzureMLPipelineRunner builds a PipelineJob where:
- train_step runs train_yolo.py and produces model_output (URI_FOLDER with best.pt)
- eval_step receives train_step.outputs.model_output directly (no local download/re-upload)

This eliminates the round-trip download of best.pt between the two CommandJobs that the
separate AzureMLTrainingRunner + AzureMLEvaluationRunner approach requires.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_mlops.contracts.evaluation import (
    EvaluationConfig,
    EvaluationInput,
    EvaluationMode,
    EvaluationOutput,
    EvaluationRecommendation,
)
from agentic_mlops.contracts.training import (
    TrainingArtifact,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
    TrainingOutput,
)
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.tools.evaluation_runner import (
    evaluate_metrics_against_policy,
    load_policy,
    metrics_from_json,
)

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = get_logger(__name__)

_AZURE_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested", "NotResponding"}

# Sub-directory names used inside the pipeline's combined download tree
_TRAIN_STEP_NAME = "train_step"
_EVAL_STEP_NAME = "eval_step"


class AzureMLPipelineRunner:
    """Submits a train+eval Azure ML PipelineJob; downloads both sets of outputs.

    The eval step receives the training weights directly from the pipeline's data-flow
    (``${{parent.jobs.train_step.outputs.model_output}}/best.pt``), so no intermediate
    download/re-upload of best.pt is needed between steps.

    Inject ``FakeAzureMLClientFactory`` for tests — no real Azure calls made.
    """

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )

            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(
        self,
        training_inp: TrainingInput,
        evaluation_inp: EvaluationInput,
        training_artifacts_dir: Path,
        evaluation_artifacts_dir: Path,
    ) -> tuple[TrainingOutput, EvaluationOutput]:
        """Submit the pipeline, wait for completion, download both outputs.

        Returns ``(TrainingOutput, EvaluationOutput)`` — the same contracts as
        AzureMLTrainingRunner.run() / AzureMLEvaluationRunner.run() so callers can
        treat pipeline-mode and single-job mode uniformly.
        """
        training_artifacts_dir.mkdir(parents=True, exist_ok=True)
        evaluation_artifacts_dir.mkdir(parents=True, exist_ok=True)

        cfg = training_inp.training_config
        eval_cfg: EvaluationConfig | None = None
        if evaluation_inp.evaluation_config_path:
            eval_cfg = EvaluationConfig.from_yaml(evaluation_inp.evaluation_config_path)

        started_at = datetime.now(tz=UTC).isoformat()

        logger.info(
            "AzureMLPipelineRunner starting",
            extra={"model": cfg.model, "compute": self._config.compute_name},
        )

        try:
            ml_client = self._factory.create(self._config)
            pipeline_job = self._build_pipeline(training_inp, cfg, eval_cfg)
            submitted = ml_client.jobs.create_or_update(pipeline_job)
            pipeline_name: str = submitted.name

            logger.info("Azure ML pipeline submitted", extra={"pipeline_name": pipeline_name})

            pcfg = self._config.pipeline
            if pcfg.stream_logs:
                ml_client.jobs.stream(pipeline_name)
                final = ml_client.jobs.get(pipeline_name)
            else:
                final = self._wait_for_completion(ml_client, pipeline_name)

            completed_at = datetime.now(tz=UTC).isoformat()

            if final.status != "Completed":
                return self._failed_outputs(
                    cfg,
                    pipeline_name,
                    final.status,
                    f"Azure ML pipeline ended with status '{final.status}'",
                    started_at,
                    completed_at,
                    evaluation_inp,
                )

            if pcfg.download_outputs:
                # Download all step outputs into a shared base dir; each step gets its own subdir
                base_dir = training_artifacts_dir / "_pipeline_outputs"
                ml_client.jobs.download(
                    pipeline_name,
                    all_outputs=True,
                    download_path=str(base_dir),
                )
            else:
                base_dir = training_artifacts_dir / "_pipeline_outputs"

            train_out = self._build_training_output(
                cfg,
                pipeline_name,
                base_dir,
                training_artifacts_dir,
                started_at,
                completed_at,
                submitted,
            )
            eval_out = self._build_evaluation_output(
                evaluation_inp,
                eval_cfg,
                base_dir,
                evaluation_artifacts_dir,
                pipeline_name,
                started_at,
                completed_at,
            )
            return train_out, eval_out

        except Exception as exc:
            logger.error("AzureMLPipelineRunner failed", extra={"error": str(exc)})
            failed_train = TrainingOutput(
                success=False,
                message=str(exc),
                mode=TrainingMode.AZURE_PIPELINE,
                job_status=TrainingJobStatus.FAILED,
                errors=[str(exc)],
            )
            failed_eval = EvaluationOutput(
                success=False,
                message=str(exc),
                mode=EvaluationMode.AZURE_PIPELINE_EVAL,
                runner="azure-ml-pipeline",
                model_path=evaluation_inp.weights_path,
                started_at=started_at,
                completed_at=datetime.now(tz=UTC).isoformat(),
                errors=[str(exc)],
            )
            return failed_train, failed_eval

    # ── Pipeline construction ──────────────────────────────────────────────────

    def _build_pipeline(
        self, inp: TrainingInput, cfg: Any, eval_cfg: EvaluationConfig | None
    ) -> Any:
        from azure.ai.ml import Input, Output, command, dsl  # noqa: PLC0415
        from azure.ai.ml.constants import AssetTypes, InputOutputModes  # noqa: PLC0415

        dataset_path = self._config.data.asset_uri or inp.dataset_path
        input_mode = (
            InputOutputModes.RO_MOUNT
            if self._config.data.input_mode == "ro_mount"
            else InputOutputModes.DOWNLOAD
        )

        code_dir = str(Path(__file__).parent.parent / "azure_jobs")
        env = self._build_environment()
        pcfg = self._config.pipeline

        # ── Train component ────────────────────────────────────────────────────
        train_cmd = (
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
        train_component = command(
            name="train_yolo",
            code=code_dir,
            command=train_cmd,
            inputs={
                "dataset": Input(type=AssetTypes.URI_FOLDER, mode=input_mode),
                "data_yaml": Input(type=AssetTypes.URI_FILE),
            },
            outputs={pcfg.train_output_name: Output(type=AssetTypes.URI_FOLDER)},
            environment=env,
            compute=self._config.compute_name,
        )

        # ── Eval component ─────────────────────────────────────────────────────
        imgsz = eval_cfg.imgsz if eval_cfg else 640
        batch = eval_cfg.batch if eval_cfg else 8
        device = eval_cfg.device if eval_cfg else "cpu"

        eval_cmd = (
            "python eval_yolo.py"
            # best.pt lives at the root of train_step's model_output folder
            " --weights ${{inputs.weights}}/best.pt"
            " --dataset-path ${{inputs.dataset}}"
            " --data-yaml ${{inputs.data_yaml}}"
            f" --imgsz {imgsz}"
            f" --batch {batch}"
            f" --device {device}"
            " --output-dir ${{outputs.eval_output}}"
        )
        eval_component = command(
            name="eval_yolo",
            code=code_dir,
            command=eval_cmd,
            inputs={
                "dataset": Input(type=AssetTypes.URI_FOLDER, mode=input_mode),
                "data_yaml": Input(type=AssetTypes.URI_FILE),
                # Receives the entire model_output folder from train_step
                "weights": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={pcfg.eval_output_name: Output(type=AssetTypes.URI_FOLDER)},
            environment=env,
            compute=self._config.compute_name,
        )

        # ── Wire the two steps together ────────────────────────────────────────
        @dsl.pipeline(
            name="agentic-mlops-train-eval",
            description="YOLO training + evaluation pipeline",
            tags=self._config.job.tags,
        )
        def _pipeline_func(dataset: Input, data_yaml: Input) -> None:  # type: ignore[type-arg]
            train_step = train_component(dataset=dataset, data_yaml=data_yaml)
            train_step.name = _TRAIN_STEP_NAME
            eval_step = eval_component(
                dataset=dataset,
                data_yaml=data_yaml,
                weights=train_step.outputs[pcfg.train_output_name],
            )
            eval_step.name = _EVAL_STEP_NAME

        pipeline_job = _pipeline_func(
            dataset=Input(type=AssetTypes.URI_FOLDER, path=dataset_path, mode=input_mode),
            data_yaml=Input(type=AssetTypes.URI_FILE, path=inp.data_yaml_path),
        )
        pipeline_job.settings.default_compute = self._config.compute_name
        pipeline_job.experiment_name = self._config.experiment_name
        pipeline_job.display_name = f"agentic-mlops-pipeline-{cfg.name}"
        return pipeline_job

    def _build_environment(self) -> Any:
        from azure.ai.ml.entities import Environment  # noqa: PLC0415

        env_cfg = self._config.environment
        if env_cfg.mode.value == "registered":
            return env_cfg.registered_environment
        return Environment(image=env_cfg.base_image, conda_file=env_cfg.conda_file)

    # ── Polling ────────────────────────────────────────────────────────────────

    def _wait_for_completion(self, ml_client: Any, pipeline_name: str) -> Any:
        deadline = time.monotonic() + self._config.pipeline.timeout_minutes * 60
        poll_interval = 30

        while time.monotonic() < deadline:
            job = ml_client.jobs.get(pipeline_name)
            if job.status in _AZURE_TERMINAL_STATES:
                return job
            logger.info(
                "Azure ML pipeline polling",
                extra={"pipeline_name": pipeline_name, "status": job.status},
            )
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0, remaining)))

        logger.warning(
            "Azure ML pipeline timed out",
            extra={
                "pipeline_name": pipeline_name,
                "timeout_minutes": self._config.pipeline.timeout_minutes,
            },
        )
        try:
            ml_client.jobs.cancel(pipeline_name)
        except Exception:
            pass
        return ml_client.jobs.get(pipeline_name)

    # ── Artifact helpers ───────────────────────────────────────────────────────

    def _find_and_promote(self, search_root: Path, filename: str, dest_dir: Path) -> Path | None:
        matches = list(search_root.rglob(filename))
        if not matches:
            return None
        found = matches[0]
        target = dest_dir / filename
        if found != target:
            shutil.copy2(found, target)
        return target

    def _collect_training_artifacts(
        self, base_dir: Path, artifacts_dir: Path
    ) -> list[TrainingArtifact]:
        train_root = base_dir / _TRAIN_STEP_NAME
        for fname in ("last.pt", "results.csv", "args.yaml"):
            self._find_and_promote(train_root, fname, artifacts_dir)
        result: list[TrainingArtifact] = []
        for weight_name in ("best.pt", "last.pt"):
            p = artifacts_dir / weight_name
            if p.exists():
                result.append(
                    TrainingArtifact(name=weight_name, path=str(p), artifact_type="weights")
                )
        for fname, atype in (("results.csv", "log"), ("args.yaml", "config")):
            p = artifacts_dir / fname
            if p.exists():
                result.append(TrainingArtifact(name=fname, path=str(p), artifact_type=atype))
        return result

    def _collect_eval_plot_artifacts(self, base_dir: Path, artifacts_dir: Path) -> list[str]:
        eval_root = base_dir / _EVAL_STEP_NAME
        candidate_names = (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "PR_curve.png",
            "F1_curve.png",
        )
        found: list[str] = []
        for name in candidate_names:
            matches = list(eval_root.rglob(name))
            if not matches:
                continue
            dst = artifacts_dir / name
            if matches[0] != dst:
                shutil.copy2(matches[0], dst)
            found.append(str(dst))
        return found

    # ── Output builders ────────────────────────────────────────────────────────

    def _build_training_output(
        self,
        cfg: Any,
        pipeline_name: str,
        base_dir: Path,
        artifacts_dir: Path,
        started_at: str,
        completed_at: str,
        submitted: Any,
    ) -> TrainingOutput:
        train_root = base_dir / _TRAIN_STEP_NAME
        best_pt = self._find_and_promote(train_root, "best.pt", artifacts_dir)
        if best_pt is None:
            return TrainingOutput(
                success=False,
                message="best.pt not found in pipeline train_step outputs",
                job_id=pipeline_name,
                mode=TrainingMode.AZURE_PIPELINE,
                job_status=TrainingJobStatus.FAILED,
                azure_job_name=pipeline_name,
                azure_job_status="Completed",
                azure_compute_name=self._config.compute_name,
                azure_experiment_name=self._config.experiment_name,
                remote_started_at=started_at,
                remote_completed_at=completed_at,
                errors=["best.pt not found in pipeline train_step outputs"],
            )

        training_artifacts = self._collect_training_artifacts(base_dir, artifacts_dir)
        output = TrainingOutput(
            success=True,
            message=f"Azure ML pipeline '{pipeline_name}' train step completed successfully.",
            job_id=pipeline_name,
            job_status=TrainingJobStatus.COMPLETED,
            best_weights_path=str(best_pt),
            mode=TrainingMode.AZURE_PIPELINE,
            azure_job_name=pipeline_name,
            azure_job_status="Completed",
            azure_studio_url=getattr(submitted, "studio_url", None),
            azure_compute_name=self._config.compute_name,
            azure_experiment_name=self._config.experiment_name,
            azure_output_name=self._config.pipeline.train_output_name,
            remote_started_at=started_at,
            remote_completed_at=completed_at,
            training_artifacts=training_artifacts,
            artifacts=[a.path for a in training_artifacts],
        )
        p = artifacts_dir / "training_output.json"
        p.write_text(json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8")
        if str(p) not in output.artifacts:
            output.artifacts.append(str(p))
        return output

    def _build_evaluation_output(
        self,
        evaluation_inp: EvaluationInput,
        eval_cfg: EvaluationConfig | None,
        base_dir: Path,
        artifacts_dir: Path,
        pipeline_name: str,
        started_at: str,
        completed_at: str,
    ) -> EvaluationOutput:
        eval_root = base_dir / _EVAL_STEP_NAME
        metrics_matches = list(eval_root.rglob("metrics.json"))
        if not metrics_matches:
            return EvaluationOutput(
                success=False,
                message="metrics.json not found in pipeline eval_step outputs",
                mode=EvaluationMode.AZURE_PIPELINE_EVAL,
                runner="azure-ml-pipeline",
                model_path=evaluation_inp.weights_path,
                started_at=started_at,
                completed_at=completed_at,
                errors=["metrics.json not found in pipeline eval_step outputs"],
                metadata={"pipeline_name": pipeline_name},
            )

        metrics_dst = artifacts_dir / "metrics.json"
        if metrics_matches[0] != metrics_dst:
            shutil.copy2(metrics_matches[0], metrics_dst)

        metrics = metrics_from_json(json.loads(metrics_dst.read_text(encoding="utf-8")))
        eval_artifacts = self._collect_eval_plot_artifacts(base_dir, artifacts_dir)
        eval_artifacts.insert(0, str(metrics_dst))

        policy = load_policy(evaluation_inp)
        recommendation, passed_checks, failed_checks = evaluate_metrics_against_policy(
            metrics, policy
        )

        output = EvaluationOutput(
            success=recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE,
            message=(
                f"Azure ML pipeline eval step complete. Recommendation: {recommendation}"
                f" (pipeline={pipeline_name})"
            ),
            metrics=metrics,
            recommendation=recommendation,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            artifacts=eval_artifacts,
            mode=EvaluationMode.AZURE_PIPELINE_EVAL,
            runner="azure-ml-pipeline",
            model_path=evaluation_inp.weights_path,
            started_at=started_at,
            completed_at=completed_at,
        )
        p = artifacts_dir / "evaluation_output.json"
        p.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if str(p) not in output.artifacts:
            output.artifacts.append(str(p))

        logger.info(
            "AzureMLPipelineRunner done",
            extra={"pipeline_name": pipeline_name, "recommendation": str(recommendation)},
        )
        return output

    # ── Failure helpers ────────────────────────────────────────────────────────

    def _failed_outputs(
        self,
        cfg: Any,
        pipeline_name: str,
        azure_status: str,
        message: str,
        started_at: str,
        completed_at: str,
        evaluation_inp: EvaluationInput,
    ) -> tuple[TrainingOutput, EvaluationOutput]:
        train_out = TrainingOutput(
            success=False,
            message=message,
            job_id=pipeline_name,
            mode=TrainingMode.AZURE_PIPELINE,
            job_status=TrainingJobStatus.FAILED,
            azure_job_name=pipeline_name,
            azure_job_status=azure_status,
            azure_compute_name=self._config.compute_name,
            azure_experiment_name=self._config.experiment_name,
            remote_started_at=started_at,
            remote_completed_at=completed_at,
            errors=[message],
        )
        eval_out = EvaluationOutput(
            success=False,
            message=message,
            mode=EvaluationMode.AZURE_PIPELINE_EVAL,
            runner="azure-ml-pipeline",
            model_path=evaluation_inp.weights_path,
            started_at=started_at,
            completed_at=completed_at,
            errors=[message],
            metadata={"pipeline_name": pipeline_name},
        )
        return train_out, eval_out
