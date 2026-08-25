"""Active learning loop demo.

Demonstrates the full closed-loop cycle:

  1. Monitor   — scan a predictions log for hard samples (low-confidence /
                 zero-detection images)
  2. Ingest    — stage found hard samples and run DataIntakeAgent over them
  3. Structure — build a YOLO-compatible split dataset from the staged images
  4. Version   — register the new dataset version in the local registry
  5. Train     — run the Orchestrator (fake runner) to validate + train +
                 evaluate + approve + register a new model version

All steps use fake/local runners so the demo runs without Azure ML, YOLO,
or any GPU.  Swap `runner: fake` for `runner: local-yolo` (or `azure-ml`)
to run for real.

Usage:
    pip install -e ".[dev]"
    python examples/active_learning_loop.py [--iterations N] [--runs-dir runs]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_fake_predictions_log(log_path: Path, n_hard: int = 3, n_ok: int = 10) -> None:
    """Write a fake JSONL predictions log with some hard samples."""
    import hashlib

    records = []
    # Hard samples — zero detections
    for i in range(n_hard):
        img_id = f"hard_{i:04d}.jpg"
        records.append(
            {
                "timestamp": f"2026-08-25T10:{i:02d}:00Z",
                "image_id": img_id,
                "latency_ms": 45.0 + i,
                "error": False,
                "detections": [],  # zero detections → hard sample
                "_sha256": hashlib.sha256(img_id.encode()).hexdigest(),
            }
        )
    # Normal samples — confident detections
    for i in range(n_ok):
        img_id = f"ok_{i:04d}.jpg"
        records.append(
            {
                "timestamp": f"2026-08-25T10:{i + n_hard:02d}:00Z",
                "image_id": img_id,
                "latency_ms": 38.0 + i,
                "error": False,
                "detections": [{"class": "scratch", "confidence": 0.92}],
            }
        )
    with open(log_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_fake_images(images_dir: Path, names: list[str]) -> None:
    """Write minimal JPEG stubs for each named image."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        (images_dir / name).write_bytes(
            b"\xff\xd8\xff\xe0" + name.encode()[:8].ljust(8, b"x") + b"\xff\xd9"
        )


def _make_training_config(path: Path) -> None:
    path.write_text("epochs: 1\nbatch: 2\nimgsz: 32\ndevice: cpu\nrunner: fake\n")


def _make_promotion_policy(path: Path) -> None:
    path.write_text(
        "map50_threshold: 0.0\nprecision_threshold: 0.0\nrecall_threshold: 0.0\n"
        "require_improvement_over_baseline: false\n"
    )


# ── main loop ─────────────────────────────────────────────────────────────────


def run_iteration(
    iteration: int,
    work_dir: Path,
    runs_dir: Path,
    dataset_registry_dir: Path,
    model_registry_dir: Path,
) -> bool:
    """Run one active-learning iteration. Returns True on success."""
    from agentic_mlops.agents.monitoring import MonitoringAgent
    from agentic_mlops.contracts.monitoring import MonitoringInput, MonitoringThresholds

    print(f"\n{'='*60}")
    print(f" Iteration {iteration}")
    print(f"{'='*60}")

    iter_dir = work_dir / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Monitor ───────────────────────────────────────────────────────
    print("\n[1/5] Monitor — scanning predictions log for hard samples…")

    log_path = iter_dir / "predictions.jsonl"
    images_dir = iter_dir / "original_images"
    _make_fake_predictions_log(log_path, n_hard=3, n_ok=8)
    _make_fake_images(images_dir, [f"hard_{i:04d}.jpg" for i in range(3)])
    _make_fake_images(images_dir, [f"ok_{i:04d}.jpg" for i in range(8)])

    monitor_agent = MonitoringAgent(artifacts_dir=iter_dir / "monitoring")
    monitor_result = monitor_agent.run(
        MonitoringInput(
            predictions_log_path=str(log_path),
            endpoint_name=f"factory-defects-v{iteration}",
            output_dir=iter_dir / "monitoring",
            thresholds=MonitoringThresholds(low_confidence_threshold=0.5),
        )
    )
    if not monitor_result.success:
        print(f"  Monitor failed: {monitor_result.message}")
        return False

    hard_manifest_path = monitor_result.hard_samples_manifest_path
    if not hard_manifest_path or not Path(hard_manifest_path).exists():
        print("  No hard samples manifest produced — nothing to do.")
        return True

    hard_manifest = Path(hard_manifest_path)
    with open(hard_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    # manifest is a list of image_id strings
    hard_sample_list = manifest if isinstance(manifest, list) else manifest.get("hard_samples", [])
    n_hard = len(hard_sample_list)
    print(f"  Found {n_hard} hard samples.")
    if n_hard == 0:
        print("  No hard samples — skipping ingest.")
        return True

    # ── Step 2: Ingest hard samples ───────────────────────────────────────────
    print("\n[2/5] Ingest — staging hard samples via DataIntakeAgent…")

    from agentic_mlops.agents.hard_sample_ingestion import HardSampleIngestionAgent
    from agentic_mlops.contracts.hard_sample_ingestion import HardSampleIngestionInput

    ingest_agent = HardSampleIngestionAgent(artifacts_dir=iter_dir / "ingest")
    ingest_result = ingest_agent.run(
        HardSampleIngestionInput(
            hard_samples_manifest_path=str(hard_manifest),
            images_source_dir=str(images_dir),
            dataset_name=f"factory_defects_hard_v{iteration}",
            # Relax corruption threshold — minimal JPEG stubs won't pass Pillow
            corrupted_ratio_threshold=1.0,
        )
    )
    if not ingest_result.success:
        print(f"  Ingest failed: {ingest_result.message}")
        return False
    staging_dir = Path(ingest_result.staging_dir or (iter_dir / "hard_samples_staging"))
    print(f"  Staged images to: {staging_dir}")
    # Write empty YOLO label stubs so DatasetStructurer + DatasetValidator are satisfied
    for img_file in staging_dir.rglob("*.jpg"):
        img_file.with_suffix(".txt").write_text("")  # empty = valid background sample

    # ── Step 3: Structure dataset ─────────────────────────────────────────────
    print("\n[3/5] Structure — building YOLO split dataset…")

    from agentic_mlops.agents.dataset_structuring import DatasetStructuringAgent
    from agentic_mlops.contracts.dataset_structuring import DatasetStructuringInput

    structured_dir = iter_dir / "structured_dataset"
    structure_agent = DatasetStructuringAgent(artifacts_dir=iter_dir / "structure")
    structure_result = structure_agent.run(
        DatasetStructuringInput(
            raw_data_path=str(staging_dir),
            output_dataset_path=str(structured_dir),
            classes=["scratch", "dent", "crack"],
            train_ratio=0.7,
            val_ratio=0.3,
            test_ratio=0.0,
        )
    )
    if not structure_result.success:
        print(f"  Structure failed: {structure_result.message}")
        return False
    print(
        f"  Split: {structure_result.metadata.get('train_count', '?')} train, "
        f"{structure_result.metadata.get('val_count', '?')} val."
    )

    # ── Step 4: Version dataset ───────────────────────────────────────────────
    print("\n[4/5] Version — registering dataset in local registry…")

    from agentic_mlops.agents.dataset_versioning import DatasetVersioningAgent
    from agentic_mlops.contracts.dataset_versioning import DatasetVersioningInput

    version_agent = DatasetVersioningAgent(artifacts_dir=iter_dir / "versioning")
    version_result = version_agent.run(
        DatasetVersioningInput(
            dataset_path=str(structured_dir),
            dataset_name="factory_defects_hard",
            registry_dir=str(dataset_registry_dir),
            output_dir=str(iter_dir / "versioning"),
        )
    )
    if not version_result.success:
        print(f"  Versioning failed: {version_result.message}")
        return False
    print(f"  Registered as version {version_result.metadata.get('version', '?')}.")

    # ── Step 5: Run orchestrator (validate → train → evaluate → approve → register)
    print("\n[5/5] Orchestrate — running pipeline with fake runner…")

    from agentic_mlops.contracts.orchestrator import OrchestratorInput
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    training_cfg = iter_dir / "training.yaml"
    policy_cfg = iter_dir / "promotion_policy.yaml"
    data_yaml = structured_dir / "data.yaml"
    _make_training_config(training_cfg)
    _make_promotion_policy(policy_cfg)

    if not data_yaml.exists():
        print(f"  data.yaml not found at {data_yaml} — skipping orchestrator step.")
        return True

    import time as _time

    workflow_id = f"al_iter_{iteration:02d}_{int(_time.time())}"
    orch_input = OrchestratorInput(
        workflow_id=workflow_id,
        runs_dir=str(runs_dir),
        dataset_path=str(structured_dir),
        data_yaml_path=str(data_yaml),
        training_config_path=str(training_cfg),
        promotion_policy_path=str(policy_cfg),
        # Dry-run: skip dataset_validation (synthetic stubs) and model_registry (no real weights).
        # For a real run: dry_run=False, steps=[..., "dataset_validation", "model_registry"].
        steps=["training", "evaluation", "approval"],
        approval_action="approve_model",
        interactive_approval=False,
        dry_run=True,
        model_name=f"factory-defects-v{iteration}",
        registry_dir=str(model_registry_dir),
        output_dir=str(runs_dir / workflow_id / "artifacts"),
    )
    workflow = OrchestratorWorkflow()
    orch_result = workflow.run(orch_input)

    print(f"  Workflow status: {orch_result.status}")
    if orch_result.status != "completed":
        print(f"  Message: {orch_result.message}")
        return False

    registered = orch_result.metadata.get("model_registry_path", "") if orch_result.metadata else ""
    print(f"  Model registered at: {registered or '(see registry)'}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Active learning loop demo")
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of iterations (default 2)"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Workflow runs directory"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate files (default: temp dir)",
    )
    args = parser.parse_args()

    use_tmp = args.work_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="al_loop_")) if use_tmp else args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_registry_dir = work_dir / "dataset_registry"
    model_registry_dir = work_dir / "model_registry"
    runs_dir = args.runs_dir

    print(f"Work directory : {work_dir}")
    print(f"Runs directory : {runs_dir}")
    print(f"Iterations     : {args.iterations}")

    success_count = 0
    for i in range(1, args.iterations + 1):
        ok = run_iteration(
            iteration=i,
            work_dir=work_dir,
            runs_dir=runs_dir,
            dataset_registry_dir=dataset_registry_dir,
            model_registry_dir=model_registry_dir,
        )
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f" Done: {success_count}/{args.iterations} iterations succeeded.")
    print(f"{'='*60}")

    if use_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0 if success_count == args.iterations else 1


if __name__ == "__main__":
    sys.exit(main())
