"""Model registry client abstraction and implementations.

Supported backends:
  - LocalModelRegistryClient   — filesystem-based, ships with this package
  - FakeModelRegistryClient    — in-memory test double
  - MLflowModelRegistryClient  — MLflow Model Registry (requires the mlflow package)
  - AzureMLModelRegistryClient — Azure ML Model asset (requires azure-ai-ml + an
    AzureMLConfig injected at construction — see create_registry_client() docstring)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_mlops.contracts.model_registry import (
    ModelLineage,
    ModelRegistrationInput,
    ModelRegistrationOutput,
    RegistrationArtifact,
    RegistrationStatus,
    RegistryBackend,
)
from agentic_mlops.observability.logging import get_logger

if TYPE_CHECKING:
    from agentic_mlops.contracts.azure_ml import AzureMLConfig

logger = get_logger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_card_md(
    model_name: str,
    version: int,
    lineage: ModelLineage,
    registered_at: str,
) -> str:
    metrics_lines = ""
    if lineage.map50 is not None:
        metrics_lines += f"| mAP@0.5       | {lineage.map50:.4f}     |\n"
    if lineage.map50_95 is not None:
        metrics_lines += f"| mAP@0.5:0.95  | {lineage.map50_95:.4f}     |\n"
    if lineage.precision is not None:
        metrics_lines += f"| Precision     | {lineage.precision:.4f}     |\n"
    if lineage.recall is not None:
        metrics_lines += f"| Recall        | {lineage.recall:.4f}     |\n"

    mlflow_section = ""
    if lineage.mlflow_run_id:
        mlflow_section = f"""
## MLflow
| Field        | Value |
|--------------|-------|
| Run ID       | `{lineage.mlflow_run_id}` |
| Experiment   | {lineage.mlflow_experiment_name or "N/A"} |
| Tracking URI | {lineage.mlflow_tracking_uri or "N/A"} |
"""

    return f"""# Model Card: {model_name} v{version}

**Registered:** {registered_at}

## Identity
- **Model Name:** {model_name}
- **Version:** {version}
- **Training Job ID:** {lineage.training_job_id or "N/A"}
- **Runner:** {lineage.training_runner or "N/A"}

## Intended Use
YOLO object detection. For research and development purposes.

## Prohibited Use
Not validated for safety-critical applications, medical imaging, or surveillance.

## Dataset
- **Dataset Path:** {lineage.dataset_path or "N/A"}
- **Data YAML:** {lineage.data_yaml or "N/A"}

## Training
- **Base Model:** {lineage.training_model or "N/A"}

## Evaluation
| Metric        | Value     |
|---------------|-----------|
{metrics_lines}
**Recommendation:** {lineage.evaluation_recommendation or "N/A"}

## Human Approval
| Field     | Value |
|-----------|-------|
| Approver  | {lineage.approved_by or "N/A"} |
| Action    | {lineage.approval_action or "N/A"} |
| Timestamp | {lineage.approval_timestamp or "N/A"} |
| Comment   | {lineage.approval_comment or "N/A"} |

## Lineage
- Training output: `{lineage.training_output_path or "N/A"}`
- Evaluation output: `{lineage.evaluation_output_path or "N/A"}`
- Approval decision: `{lineage.approval_decision_path or "N/A"}`
{mlflow_section}
## Known Limitations
- Trained on a small dataset; performance may degrade on out-of-distribution images.
- Precision may be lower for under-represented classes.
"""


def _resolve_best_weights_path(inp: ModelRegistrationInput) -> Path:
    """Locate best.pt: sibling of training_output.json, else best_weights_path within it."""
    src_weights = Path(inp.training_output_path).parent / "best.pt"
    if src_weights.exists():
        return src_weights
    train_data = json.loads(Path(inp.training_output_path).read_text(encoding="utf-8"))
    src_weights_str = train_data.get("best_weights_path")
    if not src_weights_str:
        raise FileNotFoundError("best.pt not found and not referenced in training_output.json")
    src_weights = Path(src_weights_str)
    if not src_weights.exists():
        raise FileNotFoundError(f"best.pt not found at '{src_weights}'")
    return src_weights


def _lineage_tags(lineage: ModelLineage) -> dict[str, str]:
    """Flatten key lineage fields into string tags for registry backends."""
    tags: dict[str, str] = {}
    if lineage.map50 is not None:
        tags["map50"] = f"{lineage.map50:.4f}"
    if lineage.map50_95 is not None:
        tags["map50_95"] = f"{lineage.map50_95:.4f}"
    if lineage.precision is not None:
        tags["precision"] = f"{lineage.precision:.4f}"
    if lineage.recall is not None:
        tags["recall"] = f"{lineage.recall:.4f}"
    if lineage.evaluation_recommendation:
        tags["evaluation_recommendation"] = lineage.evaluation_recommendation
    if lineage.approved_by:
        tags["approved_by"] = lineage.approved_by
    if lineage.training_job_id:
        tags["training_job_id"] = lineage.training_job_id
    return tags


class ModelRegistryClientBase(ABC):
    @abstractmethod
    def register(
        self,
        inp: ModelRegistrationInput,
        lineage: ModelLineage,
        artifacts_dir: Path,
    ) -> ModelRegistrationOutput: ...


class LocalModelRegistryClient(ModelRegistryClientBase):
    """Filesystem model registry.

    Registry layout:
      <registry_dir>/<model_name>/
        versions/
          <N>/
            model/best.pt
            lineage.json
            model_card.md
            registration_output.json
        latest.json   ← updated only after successful registration
    """

    def register(
        self,
        inp: ModelRegistrationInput,
        lineage: ModelLineage,
        artifacts_dir: Path,
    ) -> ModelRegistrationOutput:
        registry_root = Path(inp.registry_dir) / inp.model_name
        versions_dir = registry_root / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version number
        existing = sorted(
            int(p.name) for p in versions_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        version = (existing[-1] + 1) if existing else 1

        version_dir = versions_dir / str(version)
        model_dir = version_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        registered_at = datetime.now(tz=UTC).isoformat()

        try:
            # Copy best.pt
            src_weights = _resolve_best_weights_path(inp)
            dst_weights = model_dir / "best.pt"
            shutil.copy2(src_weights, dst_weights)

            # SHA-256 verification
            src_hash = _sha256(src_weights)
            dst_hash = _sha256(dst_weights)
            if src_hash != dst_hash:
                raise RuntimeError(f"SHA-256 mismatch after copy: src={src_hash} dst={dst_hash}")

            # Write lineage.json
            lineage_path = version_dir / "lineage.json"
            lineage_path.write_text(
                json.dumps(lineage.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            # Write model_card.md
            card_path = version_dir / "model_card.md"
            card_path.write_text(
                _model_card_md(inp.model_name, version, lineage, registered_at),
                encoding="utf-8",
            )

            registration_artifacts = [
                RegistrationArtifact(
                    name="best.pt",
                    path=str(dst_weights),
                    artifact_type="weights",
                ),
                RegistrationArtifact(
                    name="lineage.json",
                    path=str(lineage_path),
                    artifact_type="lineage",
                ),
                RegistrationArtifact(
                    name="model_card.md",
                    path=str(card_path),
                    artifact_type="model_card",
                ),
            ]

            output = ModelRegistrationOutput(
                success=True,
                message=f"Model '{inp.model_name}' registered as version {version}.",
                status=RegistrationStatus.REGISTERED,
                model_name=inp.model_name,
                version=version,
                registry_path=str(version_dir),
                lineage=lineage,
                registration_artifacts=registration_artifacts,
                artifacts=[a.path for a in registration_artifacts],
            )

            # Write registration_output.json inside version dir
            reg_out_path = version_dir / "registration_output.json"
            reg_out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            output.artifacts.append(str(reg_out_path))
            registration_artifacts.append(
                RegistrationArtifact(
                    name="registration_output.json",
                    path=str(reg_out_path),
                    artifact_type="output",
                )
            )

            # Also write to agent artifacts_dir for workflow tracking
            agent_out_path = artifacts_dir / "registration_output.json"
            agent_out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            # Update latest.json only after all writes succeed
            latest_path = registry_root / "latest.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "model_name": inp.model_name,
                        "version": version,
                        "registry_path": str(version_dir),
                        "registered_at": registered_at,
                        "sha256": dst_hash,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "Model registered",
                extra={
                    "model_name": inp.model_name,
                    "version": version,
                    "sha256": dst_hash,
                    "registry_path": str(version_dir),
                },
            )

            return output

        except Exception as exc:
            # Clean up partial version dir on failure
            if version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)
            logger.error(
                "LocalModelRegistryClient registration failed",
                extra={"error": str(exc)},
            )
            return ModelRegistrationOutput(
                success=False,
                message=f"Registration failed: {exc}",
                status=RegistrationStatus.FAILED,
                model_name=inp.model_name,
                errors=[str(exc)],
            )


class FakeModelRegistryClient(ModelRegistryClientBase):
    """In-memory test double. Records calls; never touches disk."""

    def __init__(self) -> None:
        self.calls: list[tuple[ModelRegistrationInput, ModelLineage]] = []
        self._next_version: int = 1

    def register(
        self,
        inp: ModelRegistrationInput,
        lineage: ModelLineage,
        artifacts_dir: Path,
    ) -> ModelRegistrationOutput:
        self.calls.append((inp, lineage))
        version = self._next_version
        self._next_version += 1
        return ModelRegistrationOutput(
            success=True,
            message=f"Fake registration: '{inp.model_name}' v{version}.",
            status=RegistrationStatus.REGISTERED,
            model_name=inp.model_name,
            version=version,
            registry_path=f"/fake/registry/{inp.model_name}/versions/{version}",
            lineage=lineage,
        )


class MLflowModelRegistryClient(ModelRegistryClientBase):
    """Registers best.pt into the MLflow Model Registry.

    Logs the weights file as an artifact on ``inp.mlflow_run_id`` (the parent
    workflow run, when tracking was enabled) or on a short-lived standalone run,
    then calls ``mlflow.register_model`` to create a new registered model version.
    Lineage is attached as model-version tags. Requires the ``mlflow`` package.
    """

    def register(
        self,
        inp: ModelRegistrationInput,
        lineage: ModelLineage,
        artifacts_dir: Path,
    ) -> ModelRegistrationOutput:
        try:
            import mlflow  # noqa: PLC0415
        except ImportError:
            msg = "mlflow is not installed. Install it with: pip install mlflow"
            return ModelRegistrationOutput(
                success=False,
                message=msg,
                status=RegistrationStatus.FAILED,
                model_name=inp.model_name,
                errors=[msg],
            )

        try:
            if inp.mlflow_tracking_uri:
                mlflow.set_tracking_uri(inp.mlflow_tracking_uri)

            src_weights = _resolve_best_weights_path(inp)
            client = mlflow.tracking.MlflowClient()

            run_id = inp.mlflow_run_id
            owns_run = run_id is None
            if owns_run:
                experiment_name = inp.mlflow_experiment_name or "agentic-mlops-registration"
                experiment = mlflow.set_experiment(experiment_name)
                run = client.create_run(experiment.experiment_id)
                run_id = run.info.run_id

            artifact_subdir = "model"
            client.log_artifact(run_id, str(src_weights), artifact_path=artifact_subdir)
            model_uri = f"runs:/{run_id}/{artifact_subdir}/{src_weights.name}"

            # Use the lower-level create_model_version (not mlflow.register_model): the
            # latter requires an MLflow 3.x "Logged Model" entity, which a plain
            # log_artifact() call does not create. create_model_version accepts any
            # run-relative artifact source directly.
            from mlflow.exceptions import MlflowException  # noqa: PLC0415

            try:
                client.create_registered_model(inp.model_name)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
            mv = client.create_model_version(inp.model_name, source=model_uri, run_id=run_id)
            version = int(mv.version)

            for key, value in _lineage_tags(lineage).items():
                client.set_model_version_tag(inp.model_name, mv.version, key, value)

            if owns_run:
                client.set_terminated(run_id, status="FINISHED")

            registered_at = datetime.now(tz=UTC).isoformat()
            registry_path = f"models:/{inp.model_name}/{version}"

            # Local audit copies alongside the agent's artifacts (mirrors LocalModelRegistryClient)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            lineage_path = artifacts_dir / "lineage.json"
            lineage_path.write_text(
                json.dumps(lineage.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            card_path = artifacts_dir / "model_card.md"
            card_path.write_text(
                _model_card_md(inp.model_name, version, lineage, registered_at),
                encoding="utf-8",
            )

            registration_artifacts = [
                RegistrationArtifact(
                    name=src_weights.name, path=model_uri, artifact_type="weights"
                ),
                RegistrationArtifact(
                    name="lineage.json", path=str(lineage_path), artifact_type="lineage"
                ),
                RegistrationArtifact(
                    name="model_card.md", path=str(card_path), artifact_type="model_card"
                ),
            ]

            output = ModelRegistrationOutput(
                success=True,
                message=f"Model '{inp.model_name}' registered as MLflow version {version}.",
                status=RegistrationStatus.REGISTERED,
                model_name=inp.model_name,
                version=version,
                registry_path=registry_path,
                lineage=lineage,
                registration_artifacts=registration_artifacts,
                artifacts=[str(lineage_path), str(card_path)],
                metadata={"mlflow_run_id": run_id, "mlflow_model_uri": model_uri},
            )

            reg_out_path = artifacts_dir / "registration_output.json"
            reg_out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            output.artifacts.append(str(reg_out_path))

            logger.info(
                "Model registered in MLflow",
                extra={
                    "model_name": inp.model_name,
                    "version": version,
                    "mlflow_run_id": run_id,
                },
            )
            return output

        except Exception as exc:
            logger.error("MLflowModelRegistryClient registration failed", extra={"error": str(exc)})
            return ModelRegistrationOutput(
                success=False,
                message=f"MLflow registration failed: {exc}",
                status=RegistrationStatus.FAILED,
                model_name=inp.model_name,
                errors=[str(exc)],
            )


class AzureMLModelRegistryClient(ModelRegistryClientBase):
    """Registers best.pt as an Azure ML Model asset (MLClient.models).

    Unlike Local/MLflow, this backend needs external connection info (subscription,
    resource group, workspace) that ModelRegistrationInput does not carry — so it
    must be constructed explicitly with an AzureMLConfig and injected as
    ``registry_client=`` (create_registry_client() cannot build one on its own).
    See CLI: ``register-model --backend azure_ml --azure-config configs/azure_ml.yaml``.

    Inject a client_factory (e.g. FakeAzureMLClientFactory) for tests — no real
    Azure calls made.
    """

    def __init__(self, config: AzureMLConfig, client_factory: Any = None) -> None:
        self._config = config
        if client_factory is None:
            from agentic_mlops.integrations.azure_ml_client import (  # noqa: PLC0415
                DefaultAzureMLClientFactory,
            )

            client_factory = DefaultAzureMLClientFactory()
        self._factory = client_factory

    def register(
        self,
        inp: ModelRegistrationInput,
        lineage: ModelLineage,
        artifacts_dir: Path,
    ) -> ModelRegistrationOutput:
        try:
            src_weights = _resolve_best_weights_path(inp)
            ml_client = self._factory.create(self._config)
            model_asset = self._build_model_asset(inp, lineage, src_weights)
            registered = ml_client.models.create_or_update(model_asset)
            version = int(registered.version)
            registry_path = getattr(registered, "id", None) or (
                f"azureml:{inp.model_name}:{version}"
            )
            registered_at = datetime.now(tz=UTC).isoformat()

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            lineage_path = artifacts_dir / "lineage.json"
            lineage_path.write_text(
                json.dumps(lineage.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            card_path = artifacts_dir / "model_card.md"
            card_path.write_text(
                _model_card_md(inp.model_name, version, lineage, registered_at),
                encoding="utf-8",
            )

            registration_artifacts = [
                RegistrationArtifact(
                    name=src_weights.name, path=registry_path, artifact_type="weights"
                ),
                RegistrationArtifact(
                    name="lineage.json", path=str(lineage_path), artifact_type="lineage"
                ),
                RegistrationArtifact(
                    name="model_card.md", path=str(card_path), artifact_type="model_card"
                ),
            ]

            output = ModelRegistrationOutput(
                success=True,
                message=(
                    f"Model '{inp.model_name}' registered as Azure ML model version {version}."
                ),
                status=RegistrationStatus.REGISTERED,
                model_name=inp.model_name,
                version=version,
                registry_path=registry_path,
                lineage=lineage,
                registration_artifacts=registration_artifacts,
                artifacts=[str(lineage_path), str(card_path)],
                metadata={
                    "azure_workspace": self._config.workspace_name,
                    "azure_model_id": registry_path,
                },
            )

            reg_out_path = artifacts_dir / "registration_output.json"
            reg_out_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            output.artifacts.append(str(reg_out_path))

            logger.info(
                "Model registered in Azure ML",
                extra={"model_name": inp.model_name, "version": version, "id": registry_path},
            )
            return output

        except Exception as exc:
            logger.error(
                "AzureMLModelRegistryClient registration failed", extra={"error": str(exc)}
            )
            return ModelRegistrationOutput(
                success=False,
                message=f"Azure ML registration failed: {exc}",
                status=RegistrationStatus.FAILED,
                model_name=inp.model_name,
                errors=[str(exc)],
            )

    def _build_model_asset(
        self, inp: ModelRegistrationInput, lineage: ModelLineage, src_weights: Path
    ) -> Any:
        from azure.ai.ml.constants import AssetTypes  # noqa: PLC0415
        from azure.ai.ml.entities import Model  # noqa: PLC0415

        return Model(
            path=str(src_weights),
            name=inp.model_name,
            type=AssetTypes.CUSTOM_MODEL,
            description="YOLO object-detection model registered by agentic-mlops.",
            tags=_lineage_tags(lineage),
        )


def create_registry_client(backend: RegistryBackend) -> ModelRegistryClientBase:
    """Resolve the registry client for a given backend.

    Used by ModelRegistryAgent when no client was explicitly injected, so that
    ModelRegistrationInput.backend actually determines where the model is registered.

    AZURE_ML cannot be resolved here: it needs an AzureMLConfig (subscription,
    resource group, workspace) that ModelRegistrationInput does not carry. Build
    AzureMLModelRegistryClient(config) yourself and pass it as registry_client=.
    """
    if backend == RegistryBackend.LOCAL:
        return LocalModelRegistryClient()
    if backend == RegistryBackend.MLFLOW:
        return MLflowModelRegistryClient()
    if backend == RegistryBackend.AZURE_ML:
        raise ValueError(
            "Azure ML Model Registry backend requires an AzureMLConfig. Construct "
            "AzureMLModelRegistryClient(config) explicitly and pass it as registry_client= "
            "(see CLI: register-model --backend azure_ml --azure-config ...)."
        )
    raise ValueError(f"Unknown registry backend: {backend}")
