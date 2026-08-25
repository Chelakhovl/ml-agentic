"""Azure ML end-to-end pipeline demo.

Shows the full Azure ML path:
  1. Train   — submit a training CommandJob via AzureMLTrainingRunner
  2. Eval    — submit an evaluation CommandJob via AzureMLEvaluationRunner
  3. Approve — non-interactive approval gate (approve_model)
  4. Register — register the model as an Azure ML Model asset
  5. Deploy  — deploy to an Azure ML Managed Online Endpoint

By default the script uses FakeAzureMLClientFactory + fake runners so it runs
offline without real credentials or compute costs.  Swap --mode real to run
against your actual workspace (requires az login / SP env vars and a valid
configs/azure_ml.yaml).

Usage:
    # Dry run (default — no Azure calls)
    pip install -e ".[dev]"
    python examples/azure_ml_pipeline.py

    # Real Azure ML run
    python examples/azure_ml_pipeline.py --mode real --azure-config configs/azure_ml.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# ── Dataset / config helpers ─────────────────────────────────────────────────


def _make_dataset(root: Path) -> tuple[Path, Path]:
    """Write a minimal valid YOLO dataset layout and return (dataset_path, data_yaml)."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img = root / "images" / split / f"img_{i:02d}.jpg"
            lbl = root / "labels" / split / f"img_{i:02d}.txt"
            img.write_bytes(b"\xff\xd8\xff\xe0" + f"img{i}".encode().ljust(8, b"x") + b"\xff\xd9")
            lbl.write_text(f"0 0.5 0.5 0.{i + 1} 0.{i + 1}\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n" f"nc: 1\nnames:\n  0: scratch\n"
    )
    return root, data_yaml


def _make_training_config(path: Path, mode: str = "fake") -> None:
    runner = "azure_train" if mode == "real" else "local_dry_run"
    path.write_text(
        f"mode: {runner}\nepochs: 1\nbatch: 2\nimgsz: 32\ndevice: cpu\n"
        "compute_target: cpu-cluster\nenvironment_name: aml-yolo-env\nenvironment_version: 1\n"
    )


def _make_promotion_policy(path: Path) -> None:
    path.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n"
        "require_improvement_over_baseline: false\n"
    )


def _make_azure_config(path: Path) -> None:
    """Write a minimal AzureMLConfig YAML (fake values — not used when mode=fake)."""
    path.write_text(
        "subscription_id: 00000000-0000-0000-0000-000000000000\n"
        "resource_group: my-resource-group\n"
        "workspace_name: my-ml-workspace\n"
        "compute_name: cpu-cluster\n"
        "experiment_name: demo-pipeline\n"
        "environment:\n"
        "  mode: registered\n"
        "  registered_environment: aml-yolo-env:1\n"
    )


# ── Pipeline steps ────────────────────────────────────────────────────────────


def step_train(
    dataset_path: Path,
    data_yaml: Path,
    training_cfg: Path,
    azure_cfg_path: Path,
    out_dir: Path,
    fake: bool,
) -> tuple[Path, str]:
    """Submit training job. Returns (artifacts_dir, best_weights_path)."""
    from agentic_mlops.agents.training import TrainingAgent
    from agentic_mlops.contracts.azure_ml import AzureMLConfig
    from agentic_mlops.contracts.training import TrainingConfig, TrainingInput, TrainingMode
    from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory
    from agentic_mlops.tools.training_runner import AzureMLTrainingRunner

    cfg = TrainingConfig.from_yaml(str(training_cfg))
    artifacts_dir = out_dir / "training"

    cfg.mode = TrainingMode.AZURE_TRAIN
    azure_cfg = AzureMLConfig.from_yaml(str(azure_cfg_path))
    if fake:
        # FakeAzureMLClientFactory intercepts SDK calls; still writes training_output.json
        runner = AzureMLTrainingRunner(azure_cfg, client_factory=FakeAzureMLClientFactory())
    else:
        runner = AzureMLTrainingRunner(azure_cfg)

    agent = TrainingAgent(artifacts_dir=artifacts_dir, azure_runner=runner)
    result = agent.run(
        TrainingInput(
            dataset_path=str(dataset_path),
            data_yaml_path=str(data_yaml),
            training_config=cfg,
            workflow_id="azure-ml-demo",
        )
    )
    if not result.success:
        print(f"  FAILED: {result.message}")
        sys.exit(1)

    weights = result.best_weights_path or "dry_run"
    print(f"  Job status : {result.job_status}")
    print(f"  Weights    : {weights}")
    return artifacts_dir, weights


def step_eval(
    dataset_path: Path,
    data_yaml: Path,
    weights: str,
    policy_path: Path,
    azure_cfg_path: Path,
    out_dir: Path,
    fake: bool,
) -> Path:
    """Submit evaluation job. Returns artifacts_dir."""
    from agentic_mlops.agents.evaluation import EvaluationAgent
    from agentic_mlops.contracts.azure_ml import AzureMLConfig
    from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMode
    from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner

    artifacts_dir = out_dir / "evaluation"

    azure_cfg = AzureMLConfig.from_yaml(str(azure_cfg_path))
    if fake:
        # LOCAL_DRY_RUN returns synthetic metrics and still writes evaluation_report.json
        mode = EvaluationMode.LOCAL_DRY_RUN
        runner = None
    else:
        mode = EvaluationMode.AZURE_EVAL
        runner = AzureMLEvaluationRunner(azure_cfg)

    agent = EvaluationAgent(artifacts_dir=artifacts_dir, azure_runner=runner)
    result = agent.run(
        EvaluationInput(
            dataset_path=str(dataset_path),
            data_yaml_path=str(data_yaml),
            weights_path=weights,
            workflow_id="azure-ml-demo",
            mode=mode,
            promotion_policy_path=str(policy_path),
        )
    )
    if not result.success:
        print(f"  FAILED: {result.message}")
        sys.exit(1)

    print(f"  Recommendation : {result.recommendation}")
    map50 = getattr(result.metrics, "map50", None) if result.metrics else None
    print(f"  mAP50          : {map50 if map50 is not None else 'n/a'}")

    # Write full output so ModelRegistryAgent Gate 3 can read success=True
    (artifacts_dir / "evaluation_output.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return artifacts_dir


def step_approve(eval_artifacts_dir: Path, out_dir: Path) -> Path:
    """Non-interactive approval gate. Returns approval artifacts_dir."""
    from agentic_mlops.agents.human_approval import HumanApprovalAgent
    from agentic_mlops.contracts.approvals import ApprovalInput

    artifacts_dir = out_dir / "approval"
    eval_output_path = eval_artifacts_dir / "evaluation_report.json"

    agent = HumanApprovalAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        ApprovalInput(
            evaluation_output_path=str(eval_output_path),
            approver="demo-script",
            output_dir=str(artifacts_dir),
            interactive=False,
            action="approve_model",
        )
    )
    if not result.success:
        print(f"  FAILED: {result.message}")
        sys.exit(1)

    print(f"  Decision : {result.action}")
    print(f"  Status   : {result.status}")
    return artifacts_dir


def step_register(
    training_artifacts_dir: Path,
    eval_artifacts_dir: Path,
    approval_artifacts_dir: Path,
    azure_cfg_path: Path,
    registry_dir: Path,
    model_name: str,
    fake: bool,
) -> str | None:
    """Register model. Returns version string."""
    from agentic_mlops.agents.model_registry import ModelRegistryAgent
    from agentic_mlops.contracts.model_registry import ModelRegistrationInput

    training_output = training_artifacts_dir / "training_output.json"
    eval_output = eval_artifacts_dir / "evaluation_output.json"
    approval_decision = approval_artifacts_dir / "approval_decision.json"

    if fake:
        agent = ModelRegistryAgent(artifacts_dir=registry_dir / "agent")
        result = agent.run(
            ModelRegistrationInput(
                model_name=model_name,
                training_output_path=str(training_output),
                evaluation_output_path=str(eval_output),
                approval_decision_path=str(approval_decision),
                registry_dir=str(registry_dir),
                backend="local",
            )
        )
    else:
        from agentic_mlops.contracts.azure_ml import AzureMLConfig
        from agentic_mlops.integrations.model_registry import AzureMLModelRegistryClient

        azure_cfg = AzureMLConfig.from_yaml(str(azure_cfg_path))
        registry_client = AzureMLModelRegistryClient(azure_cfg)
        agent = ModelRegistryAgent(
            artifacts_dir=registry_dir / "agent", registry_client=registry_client
        )
        result = agent.run(
            ModelRegistrationInput(
                model_name=model_name,
                training_output_path=str(training_output),
                evaluation_output_path=str(eval_output),
                approval_decision_path=str(approval_decision),
                backend="azure_ml",
            )
        )

    if not result.success:
        print(f"  FAILED: {result.message}")
        sys.exit(1)

    version = result.metadata.get("version") if result.metadata else None
    print(f"  Registered version : {version}")
    print(f"  Registry path      : {result.registry_path or '(azure ml)'}")
    return str(version) if version else None


def step_deploy(
    model_name: str,
    azure_model_version: str | None,
    weights: str,
    azure_cfg_path: Path,
    deploy_dir: Path,
    fake: bool,
) -> None:
    """Deploy to local staging (fake) or Azure ML Managed Online Endpoint (real)."""
    from agentic_mlops.agents.deployment import DeploymentAgent
    from agentic_mlops.contracts.deployment import DeploymentBackend, DeploymentInput

    if fake:
        # Create a stub .pt file — dry-run training produces no real weights
        stub = deploy_dir / "stub_weights.pt"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"PT_STUB")
        model_path = str(stub)

        agent = DeploymentAgent(artifacts_dir=deploy_dir / "agent")
        result = agent.run(
            DeploymentInput(
                model_path=model_path,
                model_name=model_name,
                target="staging",
                export_format="pt",
                backend=DeploymentBackend.LOCAL,
                deployment_dir=str(deploy_dir / "releases"),
                endpoint_name=f"{model_name}-staging",
            )
        )
    else:
        from agentic_mlops.contracts.azure_ml import AzureMLConfig
        from agentic_mlops.integrations.azure_ml_online_endpoint import (
            AzureMLOnlineEndpointDeployer,
        )

        azure_cfg = AzureMLConfig.from_yaml(str(azure_cfg_path))
        deployer = AzureMLOnlineEndpointDeployer(azure_cfg)
        agent = DeploymentAgent(artifacts_dir=deploy_dir / "agent", deployer=deployer)
        result = agent.run(
            DeploymentInput(
                model_name=model_name,
                target="staging",
                backend=DeploymentBackend.AZURE_ML,
                endpoint_name=f"{model_name}-endpoint",
                azure_model_name=model_name,
                azure_model_version=azure_model_version or "1",
            )
        )

    if not result.success:
        print(f"  FAILED: {result.message}")
        sys.exit(1)

    print(f"  Status   : {result.status}")
    print(f"  Endpoint : {result.endpoint_name or '(local staging)'}")
    print(f"  Release  : {result.release if result.release is not None else '(azure ml)'}")


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Azure ML pipeline demo")
    parser.add_argument(
        "--mode",
        choices=["fake", "real"],
        default="fake",
        help="'fake' uses FakeAzureMLClientFactory (default); 'real' makes actual Azure ML calls",
    )
    parser.add_argument(
        "--azure-config",
        default="configs/azure_ml.yaml",
        help="Path to azure_ml.yaml (required only for --mode real)",
    )
    parser.add_argument(
        "--model-name",
        default="demo-yolo-model",
        help="Model name used throughout the pipeline (default: demo-yolo-model)",
    )
    args = parser.parse_args()

    fake = args.mode == "fake"
    print(f"Mode: {'FAKE (offline)' if fake else 'REAL (Azure ML)'}")
    print(f"Model name: {args.model_name}")
    print()

    with tempfile.TemporaryDirectory(prefix="azure_ml_demo_") as _tmp:
        tmp = Path(_tmp)
        dataset_dir = tmp / "dataset"
        cfg_dir = tmp / "configs"
        out_dir = tmp / "outputs"
        registry_dir = tmp / "registry"
        cfg_dir.mkdir(parents=True)

        dataset_path, data_yaml = _make_dataset(dataset_dir)
        training_cfg = cfg_dir / "training.yaml"
        policy_cfg = cfg_dir / "promotion_policy.yaml"
        azure_cfg_path = Path(args.azure_config)

        _make_training_config(training_cfg, mode=args.mode)
        _make_promotion_policy(policy_cfg)

        if fake and not azure_cfg_path.exists():
            # Write a stub azure config so runners can be constructed
            azure_cfg_path = cfg_dir / "azure_ml.yaml"
            _make_azure_config(azure_cfg_path)

        # ── Step 1: Train ─────────────────────────────────────────────────────
        print("[1/5] Train")
        train_dir, weights = step_train(
            dataset_path, data_yaml, training_cfg, azure_cfg_path, out_dir, fake
        )

        # ── Step 2: Evaluate ──────────────────────────────────────────────────
        print("\n[2/5] Evaluate")
        eval_dir = step_eval(
            dataset_path, data_yaml, weights, policy_cfg, azure_cfg_path, out_dir, fake
        )

        # ── Step 3: Approve ───────────────────────────────────────────────────
        print("\n[3/5] Approve")
        approval_dir = step_approve(eval_dir, out_dir)

        # ── Step 4: Register ──────────────────────────────────────────────────
        print("\n[4/5] Register")
        az_version = step_register(
            train_dir,
            eval_dir,
            approval_dir,
            azure_cfg_path,
            registry_dir,
            args.model_name,
            fake,
        )

        # ── Step 5: Deploy ────────────────────────────────────────────────────
        print("\n[5/5] Deploy")
        step_deploy(args.model_name, az_version, weights, azure_cfg_path, out_dir, fake)

        print("\nPipeline completed successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
