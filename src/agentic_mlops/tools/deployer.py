"""Model deployer — the core tool used by the Deployment Agent.

Four backends:
  - "local" (default): exports a registered model (ONNX via Ultralytics, or a
    plain .pt passthrough copy), runs smoke tests, and writes a versioned
    release to a local staging/production directory — mirroring the same
    "N-th release + current.json" pattern already used by
    LocalModelRegistryClient and LocalDatasetVersionRegistry.
  - "azure_ml": real serving via a Managed Online Endpoint (Azure ML SDK v2)
    — see integrations/azure_ml_online_endpoint.py::AzureMLOnlineEndpointDeployer,
    injected as azure_deployer= (same "needs external connection info, inject
    it explicitly" pattern as AzureMLModelRegistryClient).
  - "docker": build a Docker image containing the model + scoring script and
    push it to a container registry. Requires a DockerConfig injected as
    docker_client= (same pattern). Calls docker_client.build() + .push().
  - "aks": docker build+push followed by kubectl apply of a generated
    Deployment+Service manifest. Requires both a DockerConfig (for image name)
    and an AksClient injected as aks_client=.

Safety rules from the spec, enforced here for BOTH backends:
  - staging can proceed automatically once smoke tests pass ("semi-automatic")
  - production requires BOTH a rollback_plan and an approved
    production_approval_path (mirrors the approval_decision.json gate
    ModelRegistryAgent already checks — same shape, same "status" field)
"""

from __future__ import annotations

import json
import shutil
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_mlops.contracts.deployment import (
    DeploymentBackend,
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
    DeploymentTarget,
    ExportFormat,
)
from agentic_mlops.contracts.docker import AksConfig, DockerConfig
from agentic_mlops.integrations.azure_ml_online_endpoint import AzureMLOnlineEndpointDeployer
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


def _import_yolo():
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc


class ModelExporter:
    """Exports a YOLO model to the requested serving format."""

    def export(self, model_path: Path, export_format: ExportFormat, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_format == ExportFormat.PT:
            dst = output_dir / model_path.name
            shutil.copy2(model_path, dst)
            return dst

        YOLO = _import_yolo()
        model = YOLO(str(model_path))
        exported = model.export(format=export_format.value)
        exported_path = Path(exported)
        dst = output_dir / exported_path.name
        if exported_path.resolve() != dst.resolve():
            shutil.copy2(exported_path, dst)
        return dst


def _run_smoke_tests(exported_path: Path, export_format: ExportFormat) -> tuple[bool, list[str]]:
    """Sanity-check the exported artifact. Returns (passed, check_messages)."""
    if not exported_path.exists():
        return False, [f"Exported model file not found: {exported_path}"]

    size = exported_path.stat().st_size
    if size == 0:
        return False, ["Exported model file is empty (0 bytes)."]
    checks = [f"Exported file exists and is non-empty ({size} bytes)."]

    if export_format == ExportFormat.ONNX:
        try:
            import onnx  # noqa: PLC0415

            onnx_model = onnx.load(str(exported_path))
            onnx.checker.check_model(onnx_model)
            checks.append("ONNX model structure is valid (onnx.checker.check_model passed).")
        except ImportError:
            checks.append(
                "onnx package not installed — skipped structural validation (best-effort)."
            )
        except Exception as exc:
            return False, [f"ONNX structural validation failed: {exc}"]

    return True, checks


class ModelDeployer:
    """Orchestrates export → smoke test → local release / Azure ML / Docker / AKS."""

    def __init__(
        self,
        exporter: ModelExporter | None = None,
        azure_deployer: AzureMLOnlineEndpointDeployer | None = None,
        docker_client: Any | None = None,
        docker_config: DockerConfig | None = None,
        aks_client: Any | None = None,
        aks_config: AksConfig | None = None,
    ) -> None:
        self._exporter = exporter or ModelExporter()
        self._azure_deployer = azure_deployer
        self._docker_client = docker_client
        self._docker_config = docker_config
        self._aks_client = aks_client
        self._aks_config = aks_config

    def deploy(self, inp: DeploymentInput, artifacts_dir: Path | None = None) -> DeploymentOutput:
        if inp.target == DeploymentTarget.PRODUCTION:
            gate_failure = self._check_production_gate(inp)
            if gate_failure is not None:
                return gate_failure

        if inp.backend == DeploymentBackend.AZURE_ML:
            if self._azure_deployer is None:
                return _failed(
                    "backend='azure_ml' requires an azure_deployer to be injected "
                    "(CLI: deploy-model --backend azure_ml --azure-config ...)."
                )
            return self._azure_deployer.deploy(inp, artifacts_dir or Path(inp.deployment_dir))

        if inp.backend in (DeploymentBackend.DOCKER, DeploymentBackend.AKS):
            return self._deploy_docker_or_aks(inp, artifacts_dir or Path(inp.deployment_dir))

        if not inp.model_path:
            return _failed("model_path is required for backend='local'.")
        model_path = Path(inp.model_path)
        if not model_path.exists():
            return _failed(f"model_path not found: {model_path}")

        endpoint_name = inp.endpoint_name or f"{inp.model_name}-{inp.target.value}"
        endpoint_root = Path(inp.deployment_dir) / endpoint_name
        releases_dir = endpoint_root / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            int(p.name) for p in releases_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        release = (existing[-1] + 1) if existing else 1
        release_dir = releases_dir / str(release)

        try:
            exported_path = self._exporter.export(model_path, inp.export_format, release_dir)
        except Exception as exc:
            shutil.rmtree(release_dir, ignore_errors=True)
            logger.error("Model export failed", extra={"error": str(exc)})
            return _failed(f"Model export failed: {exc}")

        smoke_ok, smoke_checks = _run_smoke_tests(exported_path, inp.export_format)
        if not smoke_ok:
            shutil.rmtree(release_dir, ignore_errors=True)
            logger.warning(
                "Smoke tests failed — deployment aborted", extra={"checks": smoke_checks}
            )
            return DeploymentOutput(
                success=False,
                message="Smoke tests failed — deployment aborted.",
                status=DeploymentStatus.FAILED,
                endpoint_name=endpoint_name,
                errors=smoke_checks,
                smoke_test_results=smoke_checks,
            )

        deployed_at = datetime.now(tz=UTC).isoformat()
        status = (
            DeploymentStatus.DEPLOYED_TO_PRODUCTION
            if inp.target == DeploymentTarget.PRODUCTION
            else DeploymentStatus.DEPLOYED_TO_STAGING
        )

        manifest = {
            "model_name": inp.model_name,
            "model_version": inp.model_version,
            "endpoint_name": endpoint_name,
            "target": inp.target,
            "release": release,
            "export_format": inp.export_format,
            "exported_model_path": str(exported_path),
            "deployed_at": deployed_at,
            "rollback_plan": inp.rollback_plan,
            "smoke_test_results": smoke_checks,
        }
        manifest_path = release_dir / "deployment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        current_path = endpoint_root / "current.json"
        current_path.write_text(
            json.dumps(
                {
                    "endpoint_name": endpoint_name,
                    "release": release,
                    "target": str(inp.target),
                    "exported_model_path": str(exported_path),
                    "deployed_at": deployed_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Model deployed",
            extra={"endpoint_name": endpoint_name, "release": release, "target": str(inp.target)},
        )

        return DeploymentOutput(
            success=True,
            message=(
                f"Model deployed to {inp.target.value} as release {release} " f"({endpoint_name})."
            ),
            status=status,
            endpoint_name=endpoint_name,
            exported_model_path=str(exported_path),
            release=release,
            smoke_test_results=smoke_checks,
            artifacts=[str(manifest_path)],
        )

    # ── Docker / AKS ──────────────────────────────────────────────────────────

    def _deploy_docker_or_aks(self, inp: DeploymentInput, artifacts_dir: Path) -> DeploymentOutput:
        if self._docker_client is None or self._docker_config is None:
            return _failed(
                f"backend='{inp.backend}' requires a docker_client and docker_config "
                "to be injected (CLI: deploy-model --backend docker --docker-config ...)."
            )
        if inp.backend == DeploymentBackend.AKS and self._aks_client is None:
            return _failed(
                "backend='aks' requires an aks_client to be injected "
                "(CLI: deploy-model --backend aks --aks-config ...)."
            )
        if not inp.model_path:
            return _failed(f"model_path is required for backend='{inp.backend}'.")
        model_path = Path(inp.model_path)
        if not model_path.exists():
            return _failed(f"model_path not found: {model_path}")

        version_tag = f"v{inp.model_version}" if inp.model_version else "latest"
        image_tag = self._docker_config.image_tag(inp.model_name, version_tag)
        endpoint_name = inp.endpoint_name or f"{inp.model_name}-{inp.target.value}"
        build_context = artifacts_dir / "docker_build"
        model_dir = build_context / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Export model into the build context
        try:
            exported_path = self._exporter.export(model_path, inp.export_format, model_dir)
        except Exception as exc:
            shutil.rmtree(build_context, ignore_errors=True)
            return _failed(f"Model export failed: {exc}")

        # Write Dockerfile
        dockerfile = _generate_dockerfile(
            self._docker_config.base_image,
            self._docker_config.extra_requirements,
            self._docker_config.port,
        )
        (build_context / "Dockerfile").write_text(dockerfile, encoding="utf-8")

        # Write minimal FastAPI scoring script
        (build_context / "score.py").write_text(_SCORE_PY, encoding="utf-8")

        deployed_at = datetime.now(tz=UTC).isoformat()
        status = (
            DeploymentStatus.DEPLOYED_TO_PRODUCTION
            if inp.target == DeploymentTarget.PRODUCTION
            else DeploymentStatus.DEPLOYED_TO_STAGING
        )

        try:
            self._docker_client.build(build_context, image_tag)
            self._docker_client.push(image_tag)
        except Exception as exc:
            logger.error("Docker build/push failed", extra={"error": str(exc)})
            return _failed(f"Docker build/push failed: {exc}")

        scoring_uri: str | None = None
        k8s_deployment_name: str | None = None

        if inp.backend == DeploymentBackend.AKS:
            k8s_deployment_name = endpoint_name.replace("_", "-")
            aks_cfg = self._aks_config or AksConfig()
            manifest = _generate_k8s_manifest(
                deployment_name=k8s_deployment_name,
                namespace=aks_cfg.namespace,
                image=image_tag,
                replicas=aks_cfg.replicas,
                port=aks_cfg.port,
                service_type=aks_cfg.service_type,
                cpu_request=aks_cfg.cpu_request,
                memory_request=aks_cfg.memory_request,
                cpu_limit=aks_cfg.cpu_limit,
                memory_limit=aks_cfg.memory_limit,
            )
            try:
                self._aks_client.apply(manifest)
            except Exception as exc:
                logger.error("kubectl apply failed", extra={"error": str(exc)})
                return _failed(f"AKS deployment failed: {exc}")

            scoring_uri = self._aks_client.get_service_url(aks_cfg.namespace, k8s_deployment_name)
            (artifacts_dir / "k8s_manifest.yaml").write_text(manifest, encoding="utf-8")

        deployment_record: dict = {
            "backend": inp.backend.value,
            "endpoint_name": endpoint_name,
            "image_tag": image_tag,
            "model_name": inp.model_name,
            "model_version": inp.model_version,
            "target": inp.target.value,
            "export_format": inp.export_format.value,
            "deployed_at": deployed_at,
            "rollback_plan": inp.rollback_plan,
            "scoring_uri": scoring_uri,
            "k8s_deployment_name": k8s_deployment_name,
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifacts_dir / "docker_deployment_manifest.json"
        manifest_path.write_text(json.dumps(deployment_record, indent=2), encoding="utf-8")

        logger.info(
            "Docker/AKS deployment complete",
            extra={"image_tag": image_tag, "endpoint_name": endpoint_name},
        )

        return DeploymentOutput(
            success=True,
            message=f"Model deployed as Docker image '{image_tag}' ({inp.backend.value}).",
            status=status,
            endpoint_name=endpoint_name,
            exported_model_path=str(exported_path),
            scoring_uri=scoring_uri,
            image_tag=image_tag,
            k8s_deployment_name=k8s_deployment_name,
            artifacts=[str(manifest_path)],
        )

    def _check_production_gate(self, inp: DeploymentInput) -> DeploymentOutput | None:
        if not inp.rollback_plan:
            return _blocked("rollback_plan is required for production deployment.")
        if not inp.production_approval_path:
            return _blocked(
                "production_approval_path is required for production deployment (H6 gate)."
            )
        try:
            approval_data = json.loads(
                Path(inp.production_approval_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            return _failed(f"Cannot read production_approval_path: {exc}")
        if approval_data.get("status") != "approved":
            return _blocked(
                f"Production approval status is '{approval_data.get('status')}', " "not 'approved'."
            )
        return None


_SCORE_PY = textwrap.dedent("""\
    \"\"\"Minimal FastAPI scoring script for Docker/AKS deployment.\"\"\"
    import glob
    import os
    from pathlib import Path

    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()


    def _find_model() -> str:
        model_dir = Path("/app/model")
        for ext in ("*.onnx", "*.pt"):
            matches = sorted(model_dir.glob(ext))
            if matches:
                return str(matches[0])
        raise RuntimeError(f"No model file found in {model_dir}")


    MODEL_PATH = _find_model()


    class PredictRequest(BaseModel):
        image_path: str


    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": MODEL_PATH}


    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            model = YOLO(MODEL_PATH)
            results = model.predict(req.image_path)
            return {"detections": [r.tojson() for r in results]}
        except Exception as exc:
            return {"error": str(exc)}
""")


def _generate_dockerfile(base_image: str, extra_requirements: list[str], port: int) -> str:
    extra_pip = ""
    if extra_requirements:
        pkgs = " ".join(extra_requirements)
        extra_pip = f"\nRUN pip install {pkgs} --quiet"
    return textwrap.dedent(f"""\
        FROM {base_image}
        WORKDIR /app
        RUN pip install onnxruntime fastapi uvicorn --quiet{extra_pip}
        COPY model/ /app/model/
        COPY score.py /app/
        EXPOSE {port}
        CMD ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "{port}"]
    """)


def _generate_k8s_manifest(
    *,
    deployment_name: str,
    namespace: str,
    image: str,
    replicas: int,
    port: int,
    service_type: str,
    cpu_request: str,
    memory_request: str,
    cpu_limit: str,
    memory_limit: str,
) -> str:
    return textwrap.dedent(f"""\
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: {deployment_name}
          namespace: {namespace}
        spec:
          replicas: {replicas}
          selector:
            matchLabels:
              app: {deployment_name}
          template:
            metadata:
              labels:
                app: {deployment_name}
            spec:
              containers:
              - name: model-server
                image: {image}
                ports:
                - containerPort: {port}
                resources:
                  requests:
                    cpu: {cpu_request}
                    memory: {memory_request}
                  limits:
                    cpu: {cpu_limit}
                    memory: {memory_limit}
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: {deployment_name}
          namespace: {namespace}
        spec:
          selector:
            app: {deployment_name}
          ports:
          - port: 80
            targetPort: {port}
          type: {service_type}
    """)


def _blocked(message: str) -> DeploymentOutput:
    return DeploymentOutput(
        success=False,
        message=message,
        status=DeploymentStatus.BLOCKED,
        block_reason=message,
        errors=[message],
    )


def _failed(message: str) -> DeploymentOutput:
    return DeploymentOutput(
        success=False, message=message, status=DeploymentStatus.FAILED, errors=[message]
    )
