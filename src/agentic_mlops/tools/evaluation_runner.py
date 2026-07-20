"""YOLO evaluation runners.

local_dry_run : deterministic fake metrics, no YOLO call.
local_eval    : real Ultralytics YOLO .val() call.
azure_eval    : submits a CommandJob to Azure ML, downloads metrics.json + plots.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import yaml

from agentic_mlops.contracts.evaluation import (
    EvaluationConfig,
    EvaluationInput,
    EvaluationMetrics,
    EvaluationMode,
    EvaluationOutput,
    EvaluationRecommendation,
    PerClassMetrics,
)
from agentic_mlops.observability.logging import get_logger
from agentic_mlops.workflows.policies import (
    PolicyThresholds,
    PromotionPolicy,
    evaluate_metrics_against_policy,
)

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = get_logger(__name__)

# ── Dry-run default metrics ───────────────────────────────────────────────────

_DRY_RUN_GLOBAL = EvaluationMetrics(
    map50=0.862,
    map50_95=0.591,
    precision=0.84,
    recall=0.79,
)
_DRY_RUN_PER_CLASS = PerClassMetrics(precision=0.82, recall=0.78, map50=0.85, map50_95=0.56)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_class_names(data_yaml_path: str) -> list[str]:
    with open(data_yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def load_policy(inp: EvaluationInput) -> PromotionPolicy:
    """Load PromotionPolicy from evaluation_config_path, promotion_policy_path, or use default."""
    if inp.evaluation_config_path:
        cfg = EvaluationConfig.from_yaml(inp.evaluation_config_path)
        return _config_to_policy(cfg)
    if inp.promotion_policy_path:
        return PromotionPolicy.from_yaml(inp.promotion_policy_path)
    return PromotionPolicy.default()


def metrics_from_json(data: dict) -> EvaluationMetrics:
    """Parse the plain-dict metrics.json produced by azure_jobs/eval_yolo.py."""
    per_class_raw: dict = data.get("per_class_metrics", {})
    per_class = {
        name: PerClassMetrics(
            precision=pcm.get("precision", 0.0),
            recall=pcm.get("recall", 0.0),
            map50=pcm.get("map50", 0.0),
            map50_95=pcm.get("map50_95", 0.0),
        )
        for name, pcm in per_class_raw.items()
    }
    return EvaluationMetrics(
        map50=data.get("map50", 0.0),
        map50_95=data.get("map50_95", 0.0),
        precision=data.get("precision", 0.0),
        recall=data.get("recall", 0.0),
        per_class_metrics=per_class,
    )


def _config_to_policy(cfg: EvaluationConfig) -> PromotionPolicy:
    """Convert EvaluationConfig to PromotionPolicy (avoids circular import in contracts)."""
    m = cfg.metrics
    thresholds = PolicyThresholds(
        map50_min=m.min_map50,
        map50_95_min=m.min_map50_95,
        precision_min=m.min_precision,
        recall_min=m.min_recall,
    )
    critical_classes = list(cfg.critical_classes.keys())
    per_class_recall_min = {
        cls: cc.min_recall for cls, cc in cfg.critical_classes.items()
    }
    return PromotionPolicy(
        thresholds=thresholds,
        critical_classes=critical_classes,
        per_class_recall_min=per_class_recall_min,
    )


def _import_yolo():
    """Return the YOLO class from ultralytics, or raise if not installed."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc


def _extract_metrics(results, class_names: list[str]) -> EvaluationMetrics:
    """Parse Ultralytics val() results into EvaluationMetrics."""
    box = results.box
    if hasattr(box, "map50"):
        map50 = float(box.map50)
    elif hasattr(box, "map") and len(box.map) > 0:
        map50 = float(box.map[0])
    else:
        map50 = 0.0
    map50_95 = float(box.map)
    precision = float(box.mp) if hasattr(box, "mp") else 0.0
    recall = float(box.mr) if hasattr(box, "mr") else 0.0

    per_class: dict[str, PerClassMetrics] = {}
    ap_class_index = getattr(box, "ap_class_index", None)
    ap50_list = getattr(box, "ap50", None)
    ap_list = getattr(box, "ap", None)
    p_list = getattr(box, "p", None)
    r_list = getattr(box, "r", None)

    if ap_class_index is not None:
        names = getattr(results, "names", {}) or {}
        for i, cls_idx in enumerate(ap_class_index):
            idx = int(cls_idx)
            cls_name = names.get(idx) or (
                class_names[idx] if idx < len(class_names) else str(idx)
            )
            per_class[cls_name] = PerClassMetrics(
                precision=float(p_list[i]) if p_list is not None and i < len(p_list) else 0.0,
                recall=float(r_list[i]) if r_list is not None and i < len(r_list) else 0.0,
                map50=float(ap50_list[i]) if ap50_list is not None and i < len(ap50_list) else 0.0,
                map50_95=float(ap_list[i]) if ap_list is not None and i < len(ap_list) else 0.0,
            )

    return EvaluationMetrics(
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        per_class_metrics=per_class,
    )


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class EvaluationRunnerProtocol(Protocol):
    def run(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput: ...


# ── FakeEvaluationRunner ──────────────────────────────────────────────────────


class FakeEvaluationRunner:
    """Deterministic dry-run evaluator — no YOLO, no Azure, no MLflow.

    Args:
        override_metrics: Inject custom metrics (for tests).
    """

    def __init__(self, override_metrics: EvaluationMetrics | None = None) -> None:
        self._override = override_metrics

    def run(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        started_at = datetime.now(tz=UTC).isoformat()
        class_names = _load_class_names(inp.data_yaml_path)

        metrics = self._override or EvaluationMetrics(
            **_DRY_RUN_GLOBAL.model_dump(exclude={"per_class_metrics"}),
            per_class_metrics={cls: _DRY_RUN_PER_CLASS for cls in class_names},
        )

        policy = load_policy(inp)
        recommendation, passed_checks, failed_checks = evaluate_metrics_against_policy(
            metrics, policy
        )

        job_id = f"dry_eval_{uuid.uuid4().hex[:8]}"
        plan = {
            "job_id": job_id,
            "workflow_id": inp.workflow_id,
            "generated_at": started_at,
            "mode": inp.mode,
            "dataset_path": inp.dataset_path,
            "data_yaml_path": inp.data_yaml_path,
            "weights_path": inp.weights_path,
            "note": "Dry-run: no YOLO, Azure, or MLflow calls were made.",
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        plan_path = artifacts_dir / "evaluation_request.json"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        completed_at = datetime.now(tz=UTC).isoformat()

        return EvaluationOutput(
            success=recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE,
            message=f"Dry-run evaluation complete. Recommendation: {recommendation}",
            metrics=metrics,
            recommendation=recommendation,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            artifacts=[str(plan_path)],
            mode=EvaluationMode.LOCAL_DRY_RUN,
            runner="fake",
            model_path=inp.weights_path,
            started_at=started_at,
            completed_at=completed_at,
        )


# ── LocalYOLOEvaluationRunner ─────────────────────────────────────────────────


class LocalYOLOEvaluationRunner:
    """Runs real YOLO evaluation using Ultralytics .val().

    Collects confusion_matrix.png, PR_curve.png, results.csv, writes evaluation_output.json.
    """

    def run(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"local_eval_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(tz=UTC).isoformat()

        weights_path = inp.weights_path
        if not Path(weights_path).exists():
            logger.error("Weights file not found", extra={"weights_path": weights_path})
            return EvaluationOutput(
                success=False,
                message=f"Weights file not found: {weights_path}",
                job_id=job_id,
                mode=EvaluationMode.LOCAL_EVAL,
                runner="local-yolo",
                model_path=weights_path,
                started_at=started_at,
                completed_at=datetime.now(tz=UTC).isoformat(),
                errors=[f"Weights file not found: {weights_path}"],
            )

        class_names = _load_class_names(inp.data_yaml_path)

        eval_cfg: EvaluationConfig | None = None
        if inp.evaluation_config_path:
            eval_cfg = EvaluationConfig.from_yaml(inp.evaluation_config_path)

        logger.info(
            "LocalYOLOEvaluationRunner starting",
            extra={"job_id": job_id, "weights": weights_path},
        )

        try:
            YOLO = _import_yolo()
            model = YOLO(weights_path)
            val_kwargs: dict = {
                "data": inp.data_yaml_path,
                "imgsz": eval_cfg.imgsz if eval_cfg else 640,
                "batch": eval_cfg.batch if eval_cfg else 8,
                "device": eval_cfg.device if eval_cfg else "cpu",
                "project": str(artifacts_dir / (eval_cfg.project if eval_cfg else "val_runs")),
                "name": eval_cfg.name if eval_cfg else "val_run",
            }
            results = model.val(**val_kwargs)
            save_dir = Path(results.save_dir)
        except Exception as exc:
            logger.error("YOLO evaluation failed", extra={"error": str(exc)})
            return EvaluationOutput(
                success=False,
                message=f"Local YOLO evaluation failed: {exc}",
                job_id=job_id,
                mode=EvaluationMode.LOCAL_EVAL,
                runner="local-yolo",
                model_path=weights_path,
                started_at=started_at,
                completed_at=datetime.now(tz=UTC).isoformat(),
                errors=[str(exc)],
            )

        metrics = _extract_metrics(results, class_names)

        # Collect artifacts — ultralytics 8.x uses Box-prefixed curve names for detect
        eval_artifacts: list[str] = []
        candidate_artifacts = (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            # Fallback names for older ultralytics versions
            "PR_curve.png",
            "F1_curve.png",
        )
        for artifact_name in candidate_artifacts:
            src = save_dir / artifact_name
            if src.exists():
                dst = artifacts_dir / artifact_name
                shutil.copy2(src, dst)
                eval_artifacts.append(str(dst))
                logger.info("Collected artifact", extra={"artifact": artifact_name})

        policy = load_policy(inp)
        recommendation, passed_checks, failed_checks = evaluate_metrics_against_policy(
            metrics, policy
        )

        completed_at = datetime.now(tz=UTC).isoformat()

        output = EvaluationOutput(
            success=recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE,
            message=(
                f"Local YOLO evaluation complete. Recommendation: {recommendation}"
                f" (job_id={job_id})"
            ),
            metrics=metrics,
            recommendation=recommendation,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            artifacts=eval_artifacts,
            mode=EvaluationMode.LOCAL_EVAL,
            runner="local-yolo",
            model_path=weights_path,
            started_at=started_at,
            completed_at=completed_at,
        )

        output_json_path = artifacts_dir / "evaluation_output.json"
        output_json_path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output.artifacts.append(str(output_json_path))

        logger.info(
            "LocalYOLOEvaluationRunner done",
            extra={"job_id": job_id, "recommendation": str(recommendation)},
        )

        return output


# ── AzureMLEvaluationRunner ───────────────────────────────────────────────────

_AZURE_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested", "NotResponding"}


class AzureMLEvaluationRunner:
    """Submits a YOLO evaluation CommandJob to Azure ML, downloads metrics.json + plots.

    Inject a FakeAzureMLClientFactory for tests — no real Azure calls made.
    """

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )
            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    def run(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"azure_eval_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(tz=UTC).isoformat()

        eval_cfg: EvaluationConfig | None = None
        if inp.evaluation_config_path:
            eval_cfg = EvaluationConfig.from_yaml(inp.evaluation_config_path)

        logger.info(
            "AzureMLEvaluationRunner starting",
            extra={"weights": inp.weights_path, "compute": self._config.compute_name},
        )

        try:
            ml_client = self._factory.create(self._config)
            job = self._build_job(inp, eval_cfg)
            submitted = ml_client.jobs.create_or_update(job)
            azure_name: str = submitted.name

            logger.info("Azure ML evaluation job submitted", extra={"azure_job_name": azure_name})

            if self._config.job.stream_logs:
                ml_client.jobs.stream(azure_name)
                final = ml_client.jobs.get(azure_name)
            else:
                final = self._wait_for_completion(ml_client, azure_name)

            completed_at = datetime.now(tz=UTC).isoformat()

            if final.status != "Completed":
                return self._failed_output(
                    job_id,
                    azure_name,
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

            metrics_path = self._find(artifacts_dir, "metrics.json")
            if metrics_path is None:
                return self._failed_output(
                    job_id,
                    azure_name,
                    "metrics.json not found in downloaded outputs",
                    started_at,
                    completed_at,
                )

            metrics = metrics_from_json(json.loads(metrics_path.read_text(encoding="utf-8")))
            eval_artifacts = self._collect_plot_artifacts(artifacts_dir)

            policy = load_policy(inp)
            recommendation, passed_checks, failed_checks = evaluate_metrics_against_policy(
                metrics, policy
            )

            output = EvaluationOutput(
                success=recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE,
                message=(
                    f"Azure ML evaluation complete. Recommendation: {recommendation}"
                    f" (job_id={job_id})"
                ),
                metrics=metrics,
                recommendation=recommendation,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                artifacts=eval_artifacts,
                mode=EvaluationMode.AZURE_EVAL,
                runner="azure-ml",
                model_path=inp.weights_path,
                started_at=started_at,
                completed_at=completed_at,
            )
            self._write_output(output, artifacts_dir)

            logger.info(
                "AzureMLEvaluationRunner done",
                extra={"job_id": job_id, "recommendation": str(recommendation)},
            )
            return output

        except Exception as exc:
            logger.error("AzureMLEvaluationRunner failed", extra={"error": str(exc)})
            return EvaluationOutput(
                success=False,
                message=str(exc),
                mode=EvaluationMode.AZURE_EVAL,
                runner="azure-ml",
                model_path=inp.weights_path,
                started_at=started_at,
                completed_at=datetime.now(tz=UTC).isoformat(),
                errors=[str(exc)],
            )

    # ── Job construction ───────────────────────────────────────────────────────

    def _build_job(self, inp: EvaluationInput, eval_cfg: EvaluationConfig | None) -> Any:
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
        weights_input = Input(type=AssetTypes.URI_FILE, path=inp.weights_path)
        eval_output = Output(type=AssetTypes.URI_FOLDER)

        imgsz = eval_cfg.imgsz if eval_cfg else 640
        batch = eval_cfg.batch if eval_cfg else 8
        device = eval_cfg.device if eval_cfg else "cpu"

        code_dir = str(Path(__file__).parent.parent / "azure_jobs")
        cmd = (
            "python eval_yolo.py"
            " --weights ${{inputs.weights}}"
            " --dataset-path ${{inputs.dataset}}"
            " --data-yaml ${{inputs.data_yaml}}"
            f" --imgsz {imgsz}"
            f" --batch {batch}"
            f" --device {device}"
            " --output-dir ${{outputs.eval_output}}"
        )

        env = self._build_environment()

        return command(
            code=code_dir,
            command=cmd,
            inputs={
                "dataset": dataset_input,
                "data_yaml": datayaml_input,
                "weights": weights_input,
            },
            outputs={self._config.job.output_name: eval_output},
            environment=env,
            compute=self._config.compute_name,
            experiment_name=self._config.experiment_name,
            display_name="agentic-mlops-eval",
            tags=self._config.job.tags,
        )

    def _build_environment(self) -> Any:
        from azure.ai.ml.entities import Environment  # noqa: PLC0415

        env_cfg = self._config.environment
        if env_cfg.mode.value == "registered":
            return env_cfg.registered_environment
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
                "Azure ML evaluation job polling",
                extra={"azure_job_name": azure_name, "status": job.status},
            )
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0, remaining)))

        logger.warning(
            "Azure ML evaluation job timed out",
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

    def _find(self, artifacts_dir: Path, filename: str) -> Path | None:
        matches = list(artifacts_dir.rglob(filename))
        return matches[0] if matches else None

    def _collect_plot_artifacts(self, artifacts_dir: Path) -> list[str]:
        candidate_artifacts = (
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
        for name in candidate_artifacts:
            src = self._find(artifacts_dir, name)
            if src is None:
                continue
            dst = artifacts_dir / name
            if src != dst:
                shutil.copy2(src, dst)
            found.append(str(dst))
        return found

    # ── Output helpers ─────────────────────────────────────────────────────────

    def _failed_output(
        self,
        job_id: str,
        azure_name: str | None,
        message: str,
        started_at: str | None,
        completed_at: str | None,
    ) -> EvaluationOutput:
        return EvaluationOutput(
            success=False,
            message=message,
            mode=EvaluationMode.AZURE_EVAL,
            runner="azure-ml",
            started_at=started_at,
            completed_at=completed_at,
            errors=[message],
            metadata={"job_id": job_id, "azure_job_name": azure_name or ""},
        )

    def _write_output(self, output: EvaluationOutput, artifacts_dir: Path) -> None:
        p = artifacts_dir / "evaluation_output.json"
        p.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if str(p) not in output.artifacts:
            output.artifacts.append(str(p))
