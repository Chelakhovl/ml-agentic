# Backlog

> Statuses last verified 2026-07-10: 419/419 unit tests passing, `ruff check` clean.

## Phase 0 — Data Ingestion

Precedes Phase 1 in the full architecture (`New Data → Data Intake Agent → Dataset
Structuring Agent → Dataset Validation Agent → ...`), not part of the original MVP scope.

- [x] Data Intake Agent (`agents/data_intake.py`, `tools/data_intake_scanner.py`) — scans a local raw image directory (no Azure Blob/ADLS client — local path only, consistent with the rest of this MVP), checks format against `expected_formats`, detects corrupted files (Pillow-based when available via the new `vision` extra — `pip install -e ".[vision]"` — falls back to a size-only sanity check otherwise, `pillow_available` reported in the manifest), SHA-256-hashes files to find duplicates. Status: `passed` / `needs_human_source_approval` (missing `source`, duplicate ratio over threshold, or unexpected-format files present) / `failed` (path not found, too few files, corrupted ratio over threshold). Standalone CLI (`agentic-mlops data-intake`) — not wired into `run-mvp`.
- [x] Dataset manifest (`dataset_manifest.json`/`.md` — corrupted images and duplicate groups listed separately, matching the spec's acceptance criteria).
- [x] Dataset Structuring Agent (`agents/dataset_structuring.py`, `tools/dataset_structurer.py`) — raw images (+ YOLO or COCO labels) → `images/{train,val,test}` + `labels/{train,val,test}` + valid `data.yaml`. COCO→YOLO bbox conversion (VOC not implemented — not started). `split_strategy=grouped_by_source` + `group_by_regex` keeps whole groups (e.g. all frames of one video) in a single split via a greedy largest-deficit balancer, avoiding the leakage the spec warns about; falls back to per-file grouping with a warning if no regex is given. `split_report.json`/`.md` (`agentic-mlops structure-dataset`) — standalone CLI, not wired into `run-mvp`, matching Data Intake / Label QA.

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
- [x] Написать unit tests (419 tests across 19 files).
- [x] Написать README с examples.
- [x] Реализовать Model Registry Agent + local filesystem backend (pulled forward from Phase 4).
- [x] Model Decision Agent (`agents/model_decision.py`, `tools/model_decider.py`) — was originally MVP step "4. Decision" per `01_mvp_scope.md`, but the actual implementation folded threshold checks directly into `EvaluationAgent` + `HumanApprovalAgent` instead of giving it a standalone agent; this backfills that as a genuinely additive step rather than duplicating existing logic. Reads `evaluation_report.json` (already produced by `EvaluationAgent`, which already ran `workflows.policies.evaluate_metrics_against_policy` — **not recomputed here**) and maps its 7-way `EvaluationRecommendation` onto the spec's 5-way `PROMOTE`/`REJECT`/`RETRAIN`/`NEED_MORE_DATA`/`NEED_LABEL_REVIEW`. Adds two checks that existed only as unused Pydantic fields nowhere else in the codebase until now: (1) **baseline comparison** — `PromotionPolicy.require_improvement_over_baseline`/`baseline_improvement_min_map50` (defined in `workflows/policies.py` since the MVP but never read by any code path); (2) **runtime budget** — `EvaluationConfig.runtime.max_latency_ms`/`max_model_size_mb` (defined in `contracts/evaluation.py` since the MVP, also never read anywhere). Either check failing downgrades a `PROMOTE` to `RETRAIN` (never the reverse, never further downgrades an already-non-PROMOTE decision). Does not benchmark inference itself — accepts an externally-measured `measured_latency_ms`. Never auto-approves anything — `HumanApprovalAgent` remains the sole approval gate. CLI: `agentic-mlops model-decision <evaluation_report_path> [--promotion-policy ...] [--evaluation-config ...] [--baseline-report ...] [--measured-latency-ms ...]` — standalone, not wired into `run-mvp`. Writes `decision_report.json`/`.md`.

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

- [x] Annotation / Pseudo-label Agent (`agents/annotation.py`, `tools/pseudo_labeler.py`) — runs an approved YOLO model's `.predict()` over unlabeled/partially-labeled images, writes candidate YOLO labels to a **separate** `pseudo_labels/` directory (never merges into an existing dataset's `labels/` — matches the spec's "not final without review" safety rule), and routes each image by its *weakest* detection's confidence: all detections ≥ `auto_candidate` → `high` (still sample-audited, never auto-final); ≥ `human_review` → `medium`; below → `low` (hard sample). Images with **zero** detections are routed to `medium` rather than auto-accepted as empty, since the model gives no confidence signal to trust either way. Skips (does not re-predict) any image that already has a label under `existing_labels_path` when `skip_existing_labels=True` (default) — human labels are never overwritten. `review_queue.json` lists only medium/low images. Standalone CLI `agentic-mlops pseudo-label` — not wired into `run-mvp`, matching Label QA/Data Intake/Dataset Structuring.
- [x] YOLO predict tool (built into `tools/pseudo_labeler.py`, not split out separately).
- [x] Confidence routing (see above).
- [x] Label QA Agent (`agents/label_qa.py`, `tools/label_qa_checker.py`) — deterministic geometric/statistical checks (too small/large bbox, near-boundary, aspect ratio, missing label file, class imbalance) plus an optional reference-model disagreement check (IoU-matched against a YOLO model's predictions when `reference_model_path` is given). Standalone CLI (`agentic-mlops label-qa`) — not wired into `run-mvp`, since QA is normally a one-off gate after (pseudo-)labeling, not part of every training run. Never modifies labels.
- [x] Suspicious labels report (`label_quality_report.json`/`.md`, status `passed`/`review_required`/`failed`).
- [ ] Integration with CVAT / Label Studio / Azure ML Data Labeling.
- [x] Dataset Versioning Agent (`agents/dataset_versioning.py`, `integrations/dataset_registry.py::LocalDatasetVersionRegistry`) — mirrors `ModelRegistryAgent`/`LocalModelRegistryClient`'s local-registry pattern for datasets. Reads `validation_report_path`/`label_quality_report_path` when given and blocks (`status=blocked`) if either says `status="failed"` — "registers a **clean** dataset" per the spec title. Content hash = SHA-256 over sorted (relative_path, per-file-SHA-256) pairs; a new registration whose hash matches an existing version's `lineage.json` returns that version instead of copying again (`status=deduplicated`) — satisfies "при одинаковом input hash не создаются лишние версии". On a genuine new version, the full dataset tree is copied into `<registry_dir>/<name>/versions/<N>/dataset/` (mirrors how `LocalModelRegistryClient` owns an immutable copy of `best.pt`) alongside `lineage.json` (classes, hash, parent_version, workflow_id, source_batches, approved_by, validation/label-QA status). Local filesystem backend only — **Azure ML Data Asset registration is not implemented** (tracked separately under Phase 2's "Register dataset as Azure ML Data Asset"). CLI: `agentic-mlops version-dataset <dataset_path> --dataset-name ... [--validation-report ...] [--label-quality-report ...]` — standalone, not wired into `run-mvp`.

## Phase 4 — Registry and Deployment

- [x] Model Registry Agent (`agents/model_registry.py`, 5 pre-registration gates).
- [x] Model card generation (`integrations/model_registry.py::_model_card_md`).
- [x] MLflow Model Registry backend (`MLflowModelRegistryClient` — logs weights as a run artifact, calls `create_model_version`, tags the version with lineage metrics; `ModelRegistryAgent` now routes on `ModelRegistrationInput.backend` via `create_registry_client()`).
- [x] Azure ML Model Registry backend (`AzureMLModelRegistryClient` — registers `best.pt` as an Azure ML Model asset via `MLClient.models.create_or_update`, tags it with lineage metrics. Needs an `AzureMLConfig` that `ModelRegistrationInput` doesn't carry, so it's injected explicitly — `create_registry_client()` raises a clear error for this backend; CLI: `register-model --backend azure_ml --azure-config ...`).
- [x] Model export to ONNX (`tools/deployer.py::ModelExporter` — real Ultralytics `.export(format="onnx")`, soft dependency on the `onnx` package; `pt` passthrough copy also supported for local-only serving without any export dependency).
- [x] Deployment Agent (`agents/deployment.py`, `tools/deployer.py::ModelDeployer`) — export → smoke test → versioned local release, mirroring the same "N-th release + `current.json`" pattern as `LocalModelRegistryClient`/`LocalDatasetVersionRegistry`. **No real serving infrastructure exists in this codebase** — no Docker image build, no Azure ML Online Endpoint client, no AKS, no CI/CD trigger (all explicitly out of scope, not started); "deploying" means writing a smoke-tested release to `<deployment_dir>/<endpoint_name>/`, not serving live traffic. `endpoint_name` defaults to `<model_name>-<target>`.
- [x] Staging endpoint (semi-automatic per spec — proceeds without human approval once smoke tests pass; still just a local release directory, not a live network endpoint).
- [x] Smoke tests (exported file exists/non-empty; for ONNX, structural validation via `onnx.checker.check_model()` when the `onnx` package is installed, best-effort skip otherwise; failure removes the partial release).
- [x] Production approval gate (H6) — `target=production` requires BOTH a non-empty `rollback_plan` and a `production_approval_path` JSON with `status="approved"` (same shape/gate pattern `ModelRegistryAgent` already checks). Missing either blocks (`status=blocked`) before any export happens. CLI: `agentic-mlops deploy-model <model_path> --model-name ... [--target production --rollback-plan ... --production-approval ...]` — standalone, not wired into `run-mvp`.

## Phase 5 — Monitoring and Retraining

- [x] Monitoring Agent (`agents/monitoring.py`, `tools/monitor.py::ModelMonitor`) — reads a local JSON-Lines predictions log (one inference record per line: `timestamp`, `image_id`, `latency_ms`, `error`, `detections: [{class, confidence}, ...]`) and computes error rate, p95 latency, low-confidence ratio, and (given a baseline) drift score / critical-class-drop / new-class detection. **No Azure Monitor / Application Insights integration exists** — same "standalone, no real infra" pattern as the Deployment Agent having no real serving infra; a real serving stack would need to export its logs to this same JSONL shape first. CLI: `agentic-mlops monitor <predictions_log> --endpoint-name ... [--baseline-class-distribution ... --critical-classes ...]` — standalone, not wired into `run-mvp`.
- [ ] Azure Monitor / Application Insights integration (log ingestion is a local JSONL file only; no live endpoint metrics/traces).
- [x] Low-confidence sample collection (`hard_samples_manifest.json` — an image is a hard sample if it has zero detections, or its *weakest* detection's confidence is below `low_confidence_threshold`, same "weakest detection decides" rule as the Annotation Agent's confidence buckets).
- [x] Drift detection (`_drift_score()` — total variation distance between the current window's class distribution and a baseline `{class: count_or_proportion}` JSON; dependency-free, no scipy). Skipped entirely (`drift_score=0.0`) when no baseline is given.
- [ ] Hard sample dataset generation (the manifest lists hard samples for a human to triage; it is not fed back into Data Intake / Dataset Structuring automatically — that hand-off is still a manual step).
- [x] Retraining trigger thresholds from the spec's `triggers:` YAML block (`MonitoringThresholds`: `low_confidence_ratio`, `p95_latency_ms`, `critical_class_drop`, `drift_score`, `error_rate`) mapped to a `recommended_action` (`no_action` / `notify_ops` / `need_more_data` / `model_review` / `create_retraining_request`), highest-severity trigger wins when several fire at once (`critical_class_drop` > `drift_score` > `low_confidence_ratio` > latency/error). **Never triggers retraining itself — only recommends** (matches the "no auto-promote" rule every other agent in this codebase already follows); actually acting on `create_retraining_request` (looping back to Data Intake) remains a manual/human step.

## Future improvements

- [ ] Web UI for approvals.
- [ ] Teams/Slack interactive approvals.
- [ ] AutoML baseline comparison.
- [ ] Multi-model comparison.
- [ ] Cost tracking.
- [ ] Model risk scoring.
- [ ] Data quality dashboard.
