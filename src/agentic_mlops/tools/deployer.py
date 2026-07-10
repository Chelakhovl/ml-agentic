"""Model deployer — the core tool used by the Deployment Agent.

Exports a registered model (ONNX via Ultralytics, or a plain .pt passthrough copy),
runs smoke tests on the exported artifact, and writes a versioned release to a
local staging/production directory — mirroring the same "N-th release +
current.json" pattern already used by LocalModelRegistryClient and
LocalDatasetVersionRegistry.

No real serving infrastructure exists in this codebase: there is no Docker image
build, no Azure ML Online Endpoint client, no AKS/CI-CD integration (all explicitly
out of scope — see agentic_mlops_workflow_docs/docs/13_backlog.md). "Deploying"
here means writing a smoke-tested release to <deployment_dir>/<endpoint_name>/ —
actually serving traffic from it is a separate, not-yet-implemented concern.

Safety rules from the spec, enforced here:
  - staging can proceed automatically once smoke tests pass ("semi-automatic")
  - production requires BOTH a rollback_plan and an approved
    production_approval_path (mirrors the approval_decision.json gate
    ModelRegistryAgent already checks — same shape, same "status" field)
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.contracts.deployment import (
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
    DeploymentTarget,
    ExportFormat,
)
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
    """Orchestrates export → smoke test → versioned local release."""

    def __init__(self, exporter: ModelExporter | None = None) -> None:
        self._exporter = exporter or ModelExporter()

    def deploy(self, inp: DeploymentInput) -> DeploymentOutput:
        model_path = Path(inp.model_path)
        if not model_path.exists():
            return _failed(f"model_path not found: {model_path}")

        if inp.target == DeploymentTarget.PRODUCTION:
            gate_failure = self._check_production_gate(inp)
            if gate_failure is not None:
                return gate_failure

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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

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
                f"Model deployed to {inp.target.value} as release {release} "
                f"({endpoint_name})."
            ),
            status=status,
            endpoint_name=endpoint_name,
            exported_model_path=str(exported_path),
            release=release,
            smoke_test_results=smoke_checks,
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
                f"Production approval status is '{approval_data.get('status')}', "
                "not 'approved'."
            )
        return None


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
