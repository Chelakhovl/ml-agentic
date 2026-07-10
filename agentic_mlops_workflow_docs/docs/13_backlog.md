# Backlog

> Statuses last verified 2026-07-10: 249/249 unit tests passing, `ruff check` clean.

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
- [x] Написать unit tests (249 tests across 11 files).
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
- [ ] Retire the legacy stub path in `integrations/azure_ml_client.py::AzureMLTrainingClient` — dead code once `AzureMLTrainingRunner` is always injected.

## Phase 3 — Labeling Loop

- [ ] Annotation / Pseudo-label Agent.
- [ ] YOLO predict tool.
- [ ] Confidence routing.
- [ ] Label QA Agent.
- [ ] Suspicious labels report.
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
