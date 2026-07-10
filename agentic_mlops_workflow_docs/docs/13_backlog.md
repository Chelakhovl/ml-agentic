# Backlog

> Statuses last verified 2026-07-10: 292/292 unit tests passing, `ruff check` clean.

## Phase 0 — Data Ingestion

Precedes Phase 1 in the full architecture (`New Data → Data Intake Agent → Dataset
Structuring Agent → Dataset Validation Agent → ...`), not part of the original MVP scope.

- [x] Data Intake Agent (`agents/data_intake.py`, `tools/data_intake_scanner.py`) — scans a local raw image directory (no Azure Blob/ADLS client — local path only, consistent with the rest of this MVP), checks format against `expected_formats`, detects corrupted files (Pillow-based when available via the new `vision` extra — `pip install -e ".[vision]"` — falls back to a size-only sanity check otherwise, `pillow_available` reported in the manifest), SHA-256-hashes files to find duplicates. Status: `passed` / `needs_human_source_approval` (missing `source`, duplicate ratio over threshold, or unexpected-format files present) / `failed` (path not found, too few files, corrupted ratio over threshold). Standalone CLI (`agentic-mlops data-intake`) — not wired into `run-mvp`.
- [x] Dataset manifest (`dataset_manifest.json`/`.md` — corrupted images and duplicate groups listed separately, matching the spec's acceptance criteria).
- [ ] Dataset Structuring Agent (raw/curated → YOLO folder layout, COCO/VOC → YOLO label conversion, grouped train/val/test split that avoids video-frame leakage).

## Phase 1 — MVP

- [x] Создать repo structure.
- [x] Добавить `pyproject.toml`.
- [x] Добавить CLI через Typer.
- [x] Создать Pydantic contracts.
- [x] Реализовать Dataset Validation Tool.
- [x] Реализовать Dataset Validation Agent.
- [x] Реализовать Training Agent с local dry-run.
- [x] Реализовать MLflow adapter (`NoOp` / `Local` / `Fake` clients in `integrations/mlflow_client.py`).
- [x] Реализовать Evaluation Agent с mocked metrics для dry-run.
- [x] Реализовать Decision Policy (`workflows/policies.py`).
- [x] Реализовать Human Approval через CLI.
- [x] Написать unit tests (292 tests across 13 files).
- [x] Написать README с examples.
- [x] Реализовать Model Registry Agent + local filesystem backend (pulled forward from Phase 4).

## Phase 2 — Azure ML Integration

- [x] Azure ML client adapter (`AzureMLTrainingRunner`, SDK v2, `tools/training_runner.py`).
- [x] Submit Azure ML training job (CommandJob, log streaming, polling, artifact download).
- [x] Run YOLO training on Azure ML compute (`azure_jobs/train_yolo.py`).
- [x] Azure ML evaluation (`AzureMLEvaluationRunner` in `tools/evaluation_runner.py` — submits a CommandJob running `azure_jobs/eval_yolo.py`, downloads `metrics.json` + plots, applies the promotion policy; wired through `YoloEvaluator` → `EvaluationAgent` → CLI `evaluate --runner azure-ml --azure-config ...`).
- [ ] Azure ML pipeline components (multi-step AML pipeline instead of single CommandJob).
- [ ] Register dataset as Azure ML Data Asset.
- [ ] Store artifacts in Azure Blob/ADLS.
- [x] Log metrics/artifacts to MLflow (via `LocalMLflowTrackingClient`, per-agent logging).
- [x] Retire the legacy stub path — `AzureMLTrainingClient`/`AzureMLTrainingClientBase`/`FakeAzureMLTrainingClient` deleted from `integrations/azure_ml_client.py`; `YoloTrainer` now only takes `azure_runner` and raises `RuntimeError` (not a silent `NotImplementedError` fallback) when azure_train is requested without one — matches `YoloEvaluator`'s existing pattern for azure_eval.
- [x] `run-mvp` end-to-end on Azure ML — `MVPWorkflowInput.azure_config_path` + `--training-runner azure-ml --evaluation-runner azure-ml --registry-backend azure_ml --azure-config ...`; previously Azure ML was reachable only via the standalone `train`/`evaluate`/`register-model` CLI commands, not the chained workflow.

## Phase 3 — Labeling Loop

- [ ] Annotation / Pseudo-label Agent.
- [ ] YOLO predict tool.
- [ ] Confidence routing.
- [x] Label QA Agent (`agents/label_qa.py`, `tools/label_qa_checker.py`) — deterministic geometric/statistical checks (too small/large bbox, near-boundary, aspect ratio, missing label file, class imbalance) plus an optional reference-model disagreement check (IoU-matched against a YOLO model's predictions when `reference_model_path` is given). Standalone CLI (`agentic-mlops label-qa`) — not wired into `run-mvp`, since QA is normally a one-off gate after (pseudo-)labeling, not part of every training run. Never modifies labels.
- [x] Suspicious labels report (`label_quality_report.json`/`.md`, status `passed`/`review_required`/`failed`).
- [ ] Integration with CVAT / Label Studio / Azure ML Data Labeling.

## Phase 4 — Registry and Deployment

- [x] Model Registry Agent (`agents/model_registry.py`, 5 pre-registration gates).
- [x] Model card generation (`integrations/model_registry.py::_model_card_md`).
- [x] MLflow Model Registry backend (`MLflowModelRegistryClient` — logs weights as a run artifact, calls `create_model_version`, tags the version with lineage metrics; `ModelRegistryAgent` now routes on `ModelRegistrationInput.backend` via `create_registry_client()`).
- [x] Azure ML Model Registry backend (`AzureMLModelRegistryClient` — registers `best.pt` as an Azure ML Model asset via `MLClient.models.create_or_update`, tags it with lineage metrics. Needs an `AzureMLConfig` that `ModelRegistrationInput` doesn't carry, so it's injected explicitly — `create_registry_client()` raises a clear error for this backend; CLI: `register-model --backend azure_ml --azure-config ...`).
- [ ] Model export to ONNX.
- [ ] Deployment Agent.
- [ ] Staging endpoint.
- [ ] Smoke tests.
- [ ] Production approval gate.

## Phase 5 — Monitoring and Retraining

- [ ] Monitoring Agent.
- [ ] Azure Monitor integration.
- [ ] Low-confidence sample collection.
- [ ] Drift detection.
- [ ] Hard sample dataset generation.
- [ ] Retraining request workflow.

## Future improvements

- [ ] Web UI for approvals.
- [ ] Teams/Slack interactive approvals.
- [ ] AutoML baseline comparison.
- [ ] Multi-model comparison.
- [ ] Cost tracking.
- [ ] Model risk scoring.
- [ ] Data quality dashboard.
