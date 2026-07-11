"""Unit tests for the Deployment Agent, deployer tool, and CLI command.

Coverage matrix:
    1.  model_path not found -> failed
    2.  Staging deploy (pt passthrough) succeeds -> status deployed_to_staging
    3.  Release number increments across repeated deployments to the same endpoint
    4.  Custom endpoint_name is respected
    5.  Default endpoint_name is '<model_name>-<target>'
    6.  Production without rollback_plan -> blocked
    7.  Production without production_approval_path -> blocked
    8.  Production approval file unreadable -> failed
    9.  Production approval status != 'approved' -> blocked
   10.  Production with valid approval + rollback -> deployed_to_production
   11.  ONNX export via a mocked YOLO model succeeds
   12.  Export failure -> failed, partial release directory cleaned up
   13.  Smoke test failure (empty exported file) -> failed, release directory removed
   14.  onnx package not installed -> smoke test still passes (best-effort note)
   15.  onnx package installed and validates -> passed_checks includes structural note
   16.  onnx structural validation failure -> smoke tests fail
   17.  current.json is only written after a successful deployment
   18.  deployment_manifest.json contains the rollback_plan
   19.  DeploymentAgent writes deployment_report.json / .md
   20.  DeploymentAgent logs to MLflow only on success
   21.  CLI: deploy-model succeeds on staging (exit 0)
   22.  CLI: deploy-model exits 1 on missing model_path
   23.  CLI: deploy-model exits 1 on production without approval (blocked)
   24.  CLI: deploy-model exits 1 on invalid --target
   25.  CLI: deploy-model exits 1 on invalid --export-format
   26.  backend='azure_ml' without an injected azure_deployer -> failed
   27.  backend='azure_ml' delegates to the injected azure_deployer
   28.  backend='azure_ml' still enforces the H6 production gate before delegating
   29.  backend='local' without model_path -> failed
   30.  CLI: deploy-model --backend azure_ml without --azure-config exits 1
   31.  CLI: deploy-model --backend local without model_path exits 1
   32.  CLI: deploy-model exits 1 on invalid --backend
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_mlops.agents.deployment import DeploymentAgent
from agentic_mlops.contracts.deployment import (
    DeploymentBackend,
    DeploymentInput,
    DeploymentOutput,
    DeploymentStatus,
    DeploymentTarget,
    ExportFormat,
)
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.deployer import ModelDeployer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_weights(tmp_path: Path, name: str = "best.pt", content: bytes = b"fake-weights") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


class _FakeYoloModel:
    def __init__(self, export_path: Path) -> None:
        self._export_path = export_path

    def export(self, format):  # noqa: ANN001, A002
        return str(self._export_path)


def _patch_yolo(export_path: Path):
    return patch(
        "agentic_mlops.tools.deployer._import_yolo",
        return_value=lambda path: _FakeYoloModel(export_path),
    )


# ── 1. Structural failure ─────────────────────────────────────────────────────


def test_model_path_not_found_fails(tmp_path: Path) -> None:
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(tmp_path / "nope.pt"),
            model_name="m",
            deployment_dir=str(tmp_path / "deployments"),
        )
    )
    assert result.success is False
    assert result.status == DeploymentStatus.FAILED


# ── 2-5. Staging deploy basics ────────────────────────────────────────────────


def test_staging_deploy_succeeds_with_pt_passthrough(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.STAGING,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )
    assert result.success is True
    assert result.status == DeploymentStatus.DEPLOYED_TO_STAGING
    assert result.release == 1
    assert Path(result.exported_model_path).exists()


def test_release_increments_on_repeated_deploy(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    deployer = ModelDeployer()
    inp = DeploymentInput(
        model_path=str(weights),
        model_name="m",
        export_format=ExportFormat.PT,
        deployment_dir=str(tmp_path / "deployments"),
    )
    r1 = deployer.deploy(inp)
    r2 = deployer.deploy(inp)

    assert r1.release == 1
    assert r2.release == 2


def test_custom_endpoint_name_respected(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            endpoint_name="custom-endpoint",
        )
    )
    assert result.endpoint_name == "custom-endpoint"


def test_default_endpoint_name(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="factory_detector",
            target=DeploymentTarget.STAGING,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )
    assert result.endpoint_name == "factory_detector-staging"


# ── 6-10. Production H6 gate ──────────────────────────────────────────────────


def test_production_without_rollback_plan_blocked(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )
    assert result.success is False
    assert result.status == DeploymentStatus.BLOCKED
    assert "rollback_plan" in result.block_reason


def test_production_without_approval_path_blocked(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            rollback_plan="Revert to previous release.",
        )
    )
    assert result.status == DeploymentStatus.BLOCKED
    assert "production_approval_path" in result.block_reason


def test_production_approval_file_unreadable_fails(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            rollback_plan="Revert to previous release.",
            production_approval_path=str(tmp_path / "nope.json"),
        )
    )
    assert result.success is False
    assert result.status == DeploymentStatus.FAILED


def test_production_approval_not_approved_blocked(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"status": "rejected"}), encoding="utf-8")

    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            rollback_plan="Revert to previous release.",
            production_approval_path=str(approval_path),
        )
    )
    assert result.status == DeploymentStatus.BLOCKED
    assert "rejected" in result.block_reason


def test_production_with_valid_gate_succeeds(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"status": "approved"}), encoding="utf-8")

    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            rollback_plan="Revert to previous release.",
            production_approval_path=str(approval_path),
        )
    )
    assert result.success is True
    assert result.status == DeploymentStatus.DEPLOYED_TO_PRODUCTION


# ── 11-13. Export + smoke test failure paths ──────────────────────────────────


def test_onnx_export_via_mocked_yolo_succeeds(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    fake_exported = tmp_path / "fake_export" / "best.onnx"
    fake_exported.parent.mkdir(parents=True, exist_ok=True)
    fake_exported.write_bytes(b"fake-onnx-bytes")

    with _patch_yolo(fake_exported):
        result = ModelDeployer().deploy(
            DeploymentInput(
                model_path=str(weights),
                model_name="m",
                export_format=ExportFormat.ONNX,
                deployment_dir=str(tmp_path / "deployments"),
            )
        )

    assert result.success is True
    assert Path(result.exported_model_path).name == "best.onnx"
    assert Path(result.exported_model_path).exists()


def test_export_failure_cleans_up_release_dir(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)

    def _broken_import():
        raise RuntimeError("ultralytics is not installed.")

    with patch("agentic_mlops.tools.deployer._import_yolo", side_effect=_broken_import):
        result = ModelDeployer().deploy(
            DeploymentInput(
                model_path=str(weights),
                model_name="m",
                export_format=ExportFormat.ONNX,
                deployment_dir=str(tmp_path / "deployments"),
            )
        )

    assert result.success is False
    assert result.status == DeploymentStatus.FAILED
    releases_dir = tmp_path / "deployments" / "m-staging" / "releases"
    assert not releases_dir.exists() or not any(releases_dir.iterdir())


def test_empty_exported_file_fails_smoke_test(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    fake_exported = tmp_path / "fake_export" / "best.onnx"
    fake_exported.parent.mkdir(parents=True, exist_ok=True)
    fake_exported.write_bytes(b"")  # empty -> smoke test must fail

    with _patch_yolo(fake_exported):
        result = ModelDeployer().deploy(
            DeploymentInput(
                model_path=str(weights),
                model_name="m",
                export_format=ExportFormat.ONNX,
                deployment_dir=str(tmp_path / "deployments"),
            )
        )

    assert result.success is False
    assert result.status == DeploymentStatus.FAILED
    assert any("empty" in e for e in result.errors)
    releases_dir = tmp_path / "deployments" / "m-staging" / "releases"
    assert not releases_dir.exists() or not any(releases_dir.iterdir())


# ── 14-16. onnx package availability ──────────────────────────────────────────


def test_onnx_package_unavailable_still_passes(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    fake_exported = tmp_path / "fake_export" / "best.onnx"
    fake_exported.parent.mkdir(parents=True, exist_ok=True)
    fake_exported.write_bytes(b"fake-onnx-bytes")

    real_import = __import__

    def _block_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("onnx not installed")
        return real_import(name, *args, **kwargs)

    cached = sys.modules.pop("onnx", None)
    try:
        with _patch_yolo(fake_exported), patch("builtins.__import__", side_effect=_block_onnx):
            result = ModelDeployer().deploy(
                DeploymentInput(
                    model_path=str(weights),
                    model_name="m",
                    export_format=ExportFormat.ONNX,
                    deployment_dir=str(tmp_path / "deployments"),
                )
            )
    finally:
        if cached is not None:
            sys.modules["onnx"] = cached

    assert result.success is True
    assert any("not installed" in c for c in result.smoke_test_results)


def test_onnx_package_available_validates_structure(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    fake_exported = tmp_path / "fake_export" / "best.onnx"
    fake_exported.parent.mkdir(parents=True, exist_ok=True)
    fake_exported.write_bytes(b"fake-onnx-bytes")

    fake_onnx = MagicMock()
    fake_onnx.load.return_value = MagicMock()
    fake_onnx.checker.check_model.return_value = None

    with _patch_yolo(fake_exported), patch.dict(sys.modules, {"onnx": fake_onnx}):
        result = ModelDeployer().deploy(
            DeploymentInput(
                model_path=str(weights),
                model_name="m",
                export_format=ExportFormat.ONNX,
                deployment_dir=str(tmp_path / "deployments"),
            )
        )

    assert result.success is True
    assert any("valid" in c for c in result.smoke_test_results)
    fake_onnx.checker.check_model.assert_called_once()


def test_onnx_structural_validation_failure_fails_deploy(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    fake_exported = tmp_path / "fake_export" / "best.onnx"
    fake_exported.parent.mkdir(parents=True, exist_ok=True)
    fake_exported.write_bytes(b"fake-onnx-bytes")

    fake_onnx = MagicMock()
    fake_onnx.load.return_value = MagicMock()
    fake_onnx.checker.check_model.side_effect = ValueError("corrupt graph")

    with _patch_yolo(fake_exported), patch.dict(sys.modules, {"onnx": fake_onnx}):
        result = ModelDeployer().deploy(
            DeploymentInput(
                model_path=str(weights),
                model_name="m",
                export_format=ExportFormat.ONNX,
                deployment_dir=str(tmp_path / "deployments"),
            )
        )

    assert result.success is False
    assert any("corrupt graph" in e for e in result.errors)


# ── 17-18. Artifacts ────────────────────────────────────────────────────────────


def test_current_json_only_written_on_success(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    deployment_dir = tmp_path / "deployments"

    ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(deployment_dir),
        )
    )  # blocked (no rollback/approval) -> no endpoint dir written at all

    assert not (deployment_dir / "m-production").exists()


def test_manifest_contains_rollback_plan(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"status": "approved"}), encoding="utf-8")

    result = ModelDeployer().deploy(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
            rollback_plan="Revert traffic to release 3.",
            production_approval_path=str(approval_path),
        )
    )

    manifest_path = (
        tmp_path / "deployments" / "m-production" / "releases" / str(result.release)
        / "deployment_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rollback_plan"] == "Revert traffic to release 3."


# ── 19-20. DeploymentAgent ─────────────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    agent = DeploymentAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )

    assert (artifacts_dir / "deployment_report.json").exists()
    assert (artifacts_dir / "deployment_report.md").exists()
    assert result.deployment_report_path == str(artifacts_dir / "deployment_report.json")

    data = json.loads((artifacts_dir / "deployment_report.json").read_text(encoding="utf-8"))
    assert data["status"] == "deployed_to_staging"


def test_agent_logs_to_mlflow_only_on_success(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = DeploymentAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )

    assert client.runs[run_id]["tags"].get("workflow_step") == "deployment"

    client2 = FakeMLflowClient()
    run_id2 = client2.start_run("exp", "run")
    agent2 = DeploymentAgent(
        artifacts_dir=tmp_path / "artifacts2", mlflow_client=client2, mlflow_run_id=run_id2
    )
    agent2.run(
        DeploymentInput(
            model_path=str(weights),
            model_name="m",
            target=DeploymentTarget.PRODUCTION,
            export_format=ExportFormat.PT,
            deployment_dir=str(tmp_path / "deployments"),
        )
    )  # blocked
    assert client2.runs[run_id2]["tags"] == {}


# ── 21-25. CLI ─────────────────────────────────────────────────────────────────


def test_cli_deploy_model_staging_succeeds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    weights = _make_weights(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "deploy-model",
            str(weights),
            "--model-name", "m",
            "--export-format", "pt",
            "--deployment-dir", str(tmp_path / "deployments"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DEPLOYED_TO_STAGING" in result.output


def test_cli_deploy_model_exits_1_on_missing_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "deploy-model",
            str(tmp_path / "nope.pt"),
            "--model-name", "m",
            "--deployment-dir", str(tmp_path / "deployments"),
        ],
    )
    assert result.exit_code == 1


def test_cli_deploy_model_exits_1_on_production_without_gate(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    weights = _make_weights(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "deploy-model",
            str(weights),
            "--model-name", "m",
            "--export-format", "pt",
            "--target", "production",
            "--deployment-dir", str(tmp_path / "deployments"),
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_cli_deploy_model_exits_1_on_invalid_target(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    weights = _make_weights(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "deploy-model",
            str(weights),
            "--model-name", "m",
            "--target", "bogus",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid target" in result.output


def test_cli_deploy_model_exits_1_on_invalid_export_format(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    weights = _make_weights(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "deploy-model",
            str(weights),
            "--model-name", "m",
            "--export-format", "bogus",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid export-format" in result.output


# ── 26-32. backend='azure_ml' ────────────────────────────────────────────────────


class _FakeAzureDeployer:
    def __init__(self, output: DeploymentOutput) -> None:
        self._output = output
        self.calls: list[DeploymentInput] = []

    def deploy(self, inp: DeploymentInput, artifacts_dir: Path) -> DeploymentOutput:
        self.calls.append(inp)
        return self._output


def test_azure_backend_without_injected_deployer_fails(tmp_path: Path) -> None:
    result = ModelDeployer().deploy(
        DeploymentInput(
            model_name="m",
            backend=DeploymentBackend.AZURE_ML,
            azure_model_name="m-model",
            azure_model_version=1,
        )
    )
    assert result.success is False
    assert any("azure_deployer" in e for e in result.errors)


def test_azure_backend_delegates_to_injected_deployer(tmp_path: Path) -> None:
    fake_output = DeploymentOutput(
        success=True, message="ok", status=DeploymentStatus.DEPLOYED_TO_STAGING,
        endpoint_name="m-staging", scoring_uri="https://fake/score",
    )
    fake_azure = _FakeAzureDeployer(fake_output)
    deployer = ModelDeployer(azure_deployer=fake_azure)

    result = deployer.deploy(
        DeploymentInput(
            model_name="m", backend=DeploymentBackend.AZURE_ML,
            azure_model_name="m-model", azure_model_version=1,
        ),
        tmp_path / "artifacts",
    )

    assert result.success is True
    assert result.scoring_uri == "https://fake/score"
    assert len(fake_azure.calls) == 1


def test_azure_backend_still_enforces_h6_production_gate(tmp_path: Path) -> None:
    fake_azure = _FakeAzureDeployer(DeploymentOutput(success=True, message="ok"))
    deployer = ModelDeployer(azure_deployer=fake_azure)

    result = deployer.deploy(
        DeploymentInput(
            model_name="m", backend=DeploymentBackend.AZURE_ML,
            azure_model_name="m-model", azure_model_version=1,
            target=DeploymentTarget.PRODUCTION,
        )
    )

    assert result.success is False
    assert result.status == DeploymentStatus.BLOCKED
    assert "rollback_plan" in result.block_reason
    assert fake_azure.calls == []  # never reached — gate blocks before delegation


def test_local_backend_without_model_path_fails(tmp_path: Path) -> None:
    result = ModelDeployer().deploy(DeploymentInput(model_name="m"))
    assert result.success is False
    assert any("model_path" in e for e in result.errors)


def test_cli_deploy_model_azure_ml_requires_azure_config(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        ["deploy-model", "--model-name", "m", "--backend", "azure_ml"],
    )
    assert result.exit_code == 1
    assert "azure-config" in result.output


def test_cli_deploy_model_local_requires_model_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(app, ["deploy-model", "--model-name", "m"])
    assert result.exit_code == 1
    assert "model_path is required" in result.output


def test_cli_deploy_model_exits_1_on_invalid_backend(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    weights = _make_weights(tmp_path)
    result = CliRunner().invoke(
        app,
        ["deploy-model", str(weights), "--model-name", "m", "--backend", "bogus"],
    )
    assert result.exit_code == 1
    assert "Invalid backend" in result.output
