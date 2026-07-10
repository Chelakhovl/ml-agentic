# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with optional extras
pip install -e ".[dev,mlflow]"   # adds LocalMLflowTrackingClient
pip install -e ".[dev,azure]"    # adds AzureMLTrainingRunner
pip install -e ".[dev,vision]"   # adds Pillow-based corruption checks to DataIntakeAgent

# Run all unit tests
pytest tests/unit -v

# Run a single test file
pytest tests/unit/test_dataset_validator.py -v

# Run with coverage
pytest tests/unit --cov=agentic_mlops

# Run Azure integration test (requires valid azure_ml.yaml + Azure auth)
pytest -m azure_integration --azure-config configs/azure_ml.yaml

# Lint / format
ruff check src tests
ruff format src tests

# CLI: scan raw images and write a dataset manifest (standalone — not part of run-mvp)
agentic-mlops data-intake /path/to/raw_images --dataset-name my_dataset --source camera_batch_1

# CLI: structure raw images (+ YOLO/COCO labels) into a split YOLO dataset (standalone)
agentic-mlops structure-dataset /path/to/raw_images \
  --output-dataset-path ./runs/structured --classes scratch,dent,crack
agentic-mlops structure-dataset /path/to/raw_video_frames \
  --output-dataset-path ./runs/structured --classes scratch,dent \
  --split-strategy grouped_by_source --group-by-regex '^(video\d+)_'

# CLI: pre-label images with an approved YOLO model (standalone — not part of run-mvp)
agentic-mlops pseudo-label /path/to/unlabeled_images --model-path models/approved/best.pt

# CLI: register a clean, structured dataset as a new version + lineage (standalone)
agentic-mlops version-dataset ./runs/structured --dataset-name factory_defects \
  --validation-report ./runs/structured/validation_out/dataset_quality_report.json

# CLI: turn an evaluation report into an explainable promote/reject/retrain decision
agentic-mlops model-decision ./runs/evaluation/evaluation_report.json \
  --promotion-policy configs/promotion_policy.yaml --baseline-report ./baseline_eval/evaluation_report.json

# CLI: validate a YOLO dataset
agentic-mlops validate-dataset /path/to/dataset

# CLI: check label quality (standalone — not part of run-mvp)
agentic-mlops label-qa /path/to/dataset
agentic-mlops label-qa /path/to/dataset --reference-model models/approved/best.pt

# CLI: train
#   fake = write plan only; local-yolo = Ultralytics; azure-ml = Azure ML SDK
agentic-mlops train --dataset-path /path/to/dataset --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml --runner fake

agentic-mlops train --dataset-path /path/to/dataset --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml --runner azure-ml \
  --azure-config configs/azure_ml.yaml --output-dir ./runs/azure_run_001

# CLI: evaluate (auto-resolves weights from --training-output if provided)
agentic-mlops evaluate --dataset-path /path/to/dataset --data-yaml /path/to/data.yaml \
  --promotion-policy configs/promotion_policy.yaml --runner fake

# CLI: evaluate on Azure ML compute (submits a CommandJob running azure_jobs/eval_yolo.py)
agentic-mlops evaluate --dataset-path /path/to/dataset --data-yaml /path/to/data.yaml \
  --training-output ./runs/azure_run_001/training_output.json \
  --runner azure-ml --azure-config configs/azure_ml.yaml --output-dir ./runs/azure_eval_001

# CLI: approve (interactive by default; --no-interactive requires --action)
agentic-mlops approve --evaluation-output ./runs/evaluation/evaluation_report.json \
  --action approve_model --no-interactive

# CLI: register-model (standalone; gates are re-checked from disk artifacts)
#   --backend local (default) | mlflow (needs mlflow tracking) | azure_ml (needs --azure-config)
agentic-mlops register-model \
  --model-name my-yolo-model \
  --training-output ./runs/training/training_output.json \
  --evaluation-output ./runs/evaluation/evaluation_output.json \
  --approval-decision ./runs/approval/approval_decision.json \
  --registry-dir outputs/model_registry

# CLI: dry-run the full MVP pipeline
agentic-mlops run-mvp \
  --dataset-path /path/to/dataset \
  --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml \
  --output-dir ./runs/workflow_001 \
  --dry-run --no-interactive

# CLI: run-mvp with registry and MLflow
agentic-mlops run-mvp \
  --dataset-path /path/to/dataset \
  --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml \
  --output-dir ./runs/workflow_001 \
  --register-approved-model --model-name my-model \
  --mlflow-config configs/mlflow.example.yaml --enable-mlflow \
  --no-dry-run --no-interactive --approval-action approve_model

# CLI: run-mvp entirely on Azure ML (training + evaluation + registry)
agentic-mlops run-mvp \
  --dataset-path /path/to/dataset \
  --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml \
  --output-dir ./runs/workflow_azure_001 \
  --training-runner azure-ml --evaluation-runner azure-ml \
  --register-approved-model --registry-backend azure_ml \
  --azure-config configs/azure_ml.yaml \
  --no-dry-run --no-interactive --approval-action approve_model
```

Copy `configs/training.example.yaml`, `configs/promotion_policy.example.yaml`,
`configs/mlflow.example.yaml`, and `configs/azure_ml.example.yaml` as starting configs.

## Architecture

This is a **sequential multi-agent MLOps pipeline** for YOLO object detection. Agents orchestrate and report; tools execute deterministically. Data flows through typed Pydantic v2 contracts at every boundary.

### Pipeline (5 MVP agents, run in order)

| Agent | Input → Output Contract | Gate |
|---|---|---|
| `DatasetValidationAgent` | `DatasetValidationInput → DatasetValidationOutput` | Validates YOLO structure, labels, bbox ranges, cross-split duplicates |
| `TrainingAgent` | `TrainingInput → TrainingOutput` | Blocked if `dataset_validation_status == "failed"` |
| `EvaluationAgent` | `EvaluationInput → EvaluationOutput` | Blocked if training `status in {failed, cancelled}` |
| `HumanApprovalAgent` | `ApprovalInput → ApprovalOutput` | Interactive or non-interactive; requires `--force` for risky actions |
| `ModelRegistryAgent` | `ModelRegistrationInput → ModelRegistrationOutput` | Conditional — only runs when approval `action == "approve_model"`; 5 gate checks |

### Key packages

- `agents/` — one class per pipeline stage, all extend `BaseAgent` (takes `artifacts_dir`, attaches structured logger, implements `run(input) → output`); agents never hold ML state — all computation is delegated to tools
- `contracts/` — Pydantic v2 I/O models; `ToolResult` is the shared output base (carries `success`, `message`, `artifacts`, `warnings`, `errors`, `metadata`); `WorkflowState` StrEnum (14 states) lives in `contracts/common.py`
- `tools/` — `DatasetValidator`, `YoloTrainer`, `YoloEvaluator`, `ReportWriter`, `TrainingRunner` (local + `AzureMLTrainingRunner`), `EvaluationRunner` (local + `AzureMLEvaluationRunner`) — pure deterministic execution
- `workflows/` — `MVPWorkflow` chains all five agents and stops on first failure via factory injection (see below); `PromotionPolicy` enforces mAP/precision/recall thresholds loaded from YAML; `run-mvp --training-runner azure-ml --evaluation-runner azure-ml --registry-backend azure_ml --azure-config configs/azure_ml.yaml` runs the whole pipeline on Azure ML (one shared `AzureMLConfig` resolved once at the top of `MVPWorkflow.run()`, fails fast with a clear error if `azure_config_path` is missing for a step that needs it — before any step, including validation, executes)
- `integrations/` — MLflow tracking hierarchy (see below) + model registry clients (Local/MLflow/Azure ML, see below); `azure_ml_client.py` only holds the SDK v2 `MLClient` factories (`DefaultAzureMLClientFactory` / `FakeAzureMLClientFactory`) — real Azure ML training/evaluation go through `tools/training_runner.py::AzureMLTrainingRunner` / `tools/evaluation_runner.py::AzureMLEvaluationRunner`, always injected by the CLI. `YoloTrainer`/`YoloEvaluator` raise a clear `RuntimeError` for `azure_train`/`azure_eval` mode if no runner was injected — there is no legacy fallback path anymore (removed 2026-07-10; it used to raise `NotImplementedError` via a now-deleted `AzureMLTrainingClient` stub)
- `azure_jobs/` — entry scripts submitted to Azure ML as command jobs: `train_yolo.py` (training), `eval_yolo.py` (evaluation, writes `metrics.json` + plots). Both are self-contained (no `agentic_mlops` package import) since only this directory is uploaded as the job's code snapshot
- `observability/` — JSON-line structured logging via `configure_logging()`; use `--json-logs` CLI flag; all modules use `get_logger(__name__)` with `extra=` for structured fields
- `cli/main.py` — Typer app with twelve commands: `data-intake`, `structure-dataset`, `pseudo-label`, `validate-dataset`, `label-qa`, `version-dataset`, `model-decision`, `train`, `evaluate`, `approve`, `register-model`, `run-mvp`

### Training and evaluation runners

Both `train` and `evaluate` CLI commands support the same three runners; `azure-ml` requires `--azure-config configs/azure_ml.yaml` and `az login` or service principal env vars (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`).

| Runner flag | Training | Evaluation |
|---|---|---|
| `--runner fake` (default in dry-run) | Writes `training_request.json`; no YOLO call | Deterministic fake metrics (mAP50=0.862); no YOLO call |
| `--runner local-yolo` | Calls Ultralytics locally (requires `ultralytics` installed) | Real Ultralytics `.val()` locally |
| `--runner azure-ml` | Submits a CommandJob running `azure_jobs/train_yolo.py`; downloads `best.pt`/`last.pt`/`results.csv` | Submits a CommandJob running `azure_jobs/eval_yolo.py`; downloads `metrics.json` + confusion-matrix/PR-curve plots |

### Dual evaluation policy system

There are two independent config models that both feed the evaluation step — do not conflate them:

- **`PromotionPolicy`** (`workflows/policies.py`, loaded from `promotion_policy.yaml`) — drives `evaluate_metrics_against_policy()` which returns `(recommendation, passed_checks, failed_checks)`. Decision tree: critical class recall fail → `NEED_LABEL_REVIEW`; only recall fails → `COLLECT_MORE_DATA`; only precision fails → `REVIEW_LABELS`; else → `RETRAIN`; all pass → `PROMOTE_CANDIDATE`.
- **`EvaluationConfig`** (`contracts/evaluation.py`) — YOLO runtime config (imgsz, batch, device, per-class thresholds) passed via `--evaluation-config`. Separate from promotion policy.

### MLflow integration hierarchy

Four implementations in `integrations/mlflow_client.py`:
- `NoOpMLflowTrackingClient` — safe default; does nothing; no mlflow package required
- `LocalMLflowTrackingClient` — uses `mlflow` package; supports SQLite (`sqlite:///...`) or file-based tracking URIs
- `FakeMLflowTrackingClient` — in-memory test double; records all calls in `self.runs` dict (aliased as `FakeMLflowClient`)
- `MLflowClient` — legacy stub; all methods raise `NotImplementedError`; deprecated

Enable via `--mlflow-config configs/mlflow.example.yaml --enable-mlflow` on any command, or pass `mlflow_client` + `mlflow_config` directly to `MVPWorkflow`.

### Model Registry

`ModelRegistryAgent` runs after `HumanApprovalAgent` and enforces 5 gates before registering:
1. `approval.status == "approved"`
2. `approval.action == "approve_model"`
3. `evaluation.success == True`
4. `training.job_status == "completed"`
5. `best.pt` exists at `training.best_weights_path` (falls back to sibling of `training_output.json`)

Three registry backends, all fully implemented in `integrations/model_registry.py`:
- `LocalModelRegistryClient` (default) — writes to `<registry_dir>/<model_name>/versions/<N>/model/best.pt` plus `lineage.json`, `model_card.md`, `registration_output.json`; updates `<registry_dir>/<model_name>/latest.json` only after all writes succeed; SHA-256 verified copy, partial version dir cleaned up on failure.
- `MLflowModelRegistryClient` — logs `best.pt` as a run artifact, then `MlflowClient().create_model_version()` (not `mlflow.register_model()` — MLflow ≥3.x's version requires a "Logged Model" entity that a plain `log_artifact()` doesn't create); tags the version with lineage metrics.
- `AzureMLModelRegistryClient` — registers `best.pt` as an Azure ML Model asset via `MLClient.models.create_or_update()`.

**Backend routing nuance:** `ModelRegistryAgent` resolves its client via `create_registry_client(inp.backend)` when no `registry_client` is injected — this works for `local`/`mlflow` (both have everything they need in `ModelRegistrationInput`). `azure_ml` needs external connection info (subscription/resource group/workspace) that the input contract doesn't carry, so `create_registry_client()` raises a clear `ValueError` for it — the CLI (`register-model --backend azure_ml --azure-config ...`) builds `AzureMLModelRegistryClient(AzureMLConfig.from_yaml(...))` itself and injects it explicitly, same pattern as `train`/`evaluate --runner azure-ml`.

### Data Intake Agent (standalone, not part of MVPWorkflow)

`DataIntakeAgent`/`DataIntakeScanner` (`agents/data_intake.py`, `tools/data_intake_scanner.py`,
`contracts/data_intake.py`) — the second of the 8 previously-unimplemented agents; precedes
`DatasetValidationAgent` in the full architecture (`New Data → Data Intake → Dataset
Structuring → Dataset Validation → ...`). Scans a **local** raw image directory (no Azure
Blob/ADLS client exists in this codebase — `raw_data_path` is local, same as
`DatasetValidator.dataset_path`). Checks: file extension against `expected_formats`,
corruption (Pillow-based when installed via the new `vision` extra — soft dependency,
`pillow_available` reported in the manifest; falls back to a non-zero-size sanity check
otherwise), SHA-256 duplicate detection. Status: `passed` / `needs_human_source_approval` (`success=True` — soft gate: missing
`source`, duplicate ratio over threshold, or any unexpected-format files) / `failed`
(`success=False` — path not found, too few files, corrupted ratio over threshold). CLI:
`agentic-mlops data-intake <raw_data_path> --dataset-name ... [--source ...]`.
Writes `dataset_manifest.json`/`.md`. Deliberately **not** wired into `run-mvp` — same
reasoning as Label QA.

### Dataset Structuring Agent (standalone, not part of MVPWorkflow)

`DatasetStructuringAgent`/`DatasetStructurer` (`agents/dataset_structuring.py`,
`tools/dataset_structurer.py`, `contracts/dataset_structuring.py`) — the third of the 8
previously-unimplemented agents; sits between Data Intake and Dataset Validation in the
full architecture. Takes raw images (+ YOLO or COCO labels — `label_format`; **VOC is not
implemented**) and writes a YOLO-compatible `images/{train,val,test}` +
`labels/{train,val,test}` + valid `data.yaml` (only splits with ratio > 0 are created —
`test_ratio=0` means no `images/test` dir and no `test:` key). COCO bboxes (absolute
pixel `[x, y, w, h]`) are converted to normalised YOLO `(class_id, xc, yc, w, h)` using
each image's `width`/`height` from the COCO JSON; unknown categories and images missing
on disk are skipped with a warning, not a hard failure.

**Split strategies**: `random` (ratio-based, seeded) and `grouped_by_source` — the spec's
"don't leak video frames across train/val/test" rule. Groups are extracted via
`group_by_regex` (one capturing group, e.g. `r"^(video\d+)_"`); a greedy
largest-remaining-deficit balancer assigns whole groups to splits so ratios are matched as
closely as group sizes allow, guaranteeing no group (hence no file) ever spans two splits.
Without `group_by_regex`, `grouped_by_source` degrades to one-file-per-group (no real
leakage protection) with a warning rather than failing outright.

CLI: `agentic-mlops structure-dataset <raw_data_path> --output-dataset-path ... --classes
a,b,c [--label-format coco --coco-annotations ...] [--split-strategy grouped_by_source
--group-by-regex ...]`. Writes `split_report.json`/`.md`. Deliberately **not** wired into
`run-mvp` — same reasoning as Label QA / Data Intake.

### Annotation / Pseudo-label Agent (standalone, not part of MVPWorkflow)

`AnnotationAgent`/`PseudoLabeler` (`agents/annotation.py`, `tools/pseudo_labeler.py`,
`contracts/annotation.py`) — the fourth of the 8 previously-unimplemented agents. Runs an
approved/pre-trained YOLO model's `.predict()` over unlabeled/partially-labeled images and
writes candidate labels to a **separate** `pseudo_labels/` directory — never merged into an
existing dataset's `labels/` automatically (the spec's safety rule: pseudo-labels are not
final without a review/audit policy). Confidence routing uses each image's *weakest*
detection (the whole image only counts as high-confidence if every box clears the bar):
`min_confidence >= auto_candidate` → `high` (still sample-audited by a human, never
auto-final) / `>= human_review` → `medium` (human review) / below → `low` (hard/expert
review). Zero detections also route to `medium` rather than being auto-accepted as an
empty label, since there's no confidence signal to trust either way. When
`existing_labels_path` is given (default `skip_existing_labels=True`), any image that
already has a human label there is skipped entirely, never re-predicted or overwritten.
`review_queue.json` lists only medium/low images. CLI: `agentic-mlops pseudo-label
<images_path> --model-path ... [--auto-candidate-threshold ...] [--existing-labels-path
...]`. Writes `pseudo_label_report.json`/`.md`. Deliberately **not** wired into `run-mvp`
— same reasoning as the other Phase 0/3 agents.

### Label QA Agent (standalone, not part of MVPWorkflow)

`LabelQAAgent`/`LabelQAChecker` (`agents/label_qa.py`, `tools/label_qa_checker.py`,
`contracts/label_qa.py`) — the first of the 8 previously-unimplemented agents from
`agentic_mlops_workflow_docs/agents/`. Checks label quality after (pseudo-)labeling;
assumes the dataset already passed `DatasetValidationAgent` (does not repeat structural
checks like malformed columns). Deterministic checks: bbox too small/large, near image
boundary, suspicious aspect ratio, missing label file, class imbalance. Optional
reference-model disagreement check (only runs if `reference_model_path` is given): loads
a YOLO model, runs `.predict()` per image, greedy-IoU-matches predictions against human
labels, flags unmatched high-confidence predictions as possible missing labels. Never
modifies labels. Status: `passed` / `review_required` (suspicious count ≥
`review_required_threshold`, default 5) / `failed` (structural — missing data.yaml, no
split directories). CLI: `agentic-mlops label-qa <dataset_path> [--reference-model ...]`.
Deliberately **not** wired into `run-mvp` — label QA is normally a one-off gate run after
a (pseudo-)labeling pass, not part of every training run.

### Dataset Versioning Agent (standalone, not part of MVPWorkflow)

`DatasetVersioningAgent`/`LocalDatasetVersionRegistry` (`agents/dataset_versioning.py`,
`integrations/dataset_registry.py`, `contracts/dataset_versioning.py`) — the fifth of the 8
previously-unimplemented agents; mirrors `ModelRegistryAgent`/`LocalModelRegistryClient`'s
local-registry pattern, but for datasets instead of trained models. The agent itself gates
on `validation_report_path`/`label_quality_report_path` when given — either report's
`status == "failed"` blocks registration (`status=blocked`), since the spec's whole point is
registering a **clean** dataset. Content hash = SHA-256 over sorted `(relative_path,
file_sha256)` pairs; if a new registration's hash matches an existing version's
`lineage.json`, that version is returned instead of copying again (`status=deduplicated`,
no new version) — satisfies the spec's "identical input hash shouldn't create redundant
versions" rule. On an actual new version the full dataset tree is copied into
`<registry_dir>/<name>/versions/<N>/dataset/` (same "registry owns an immutable copy"
philosophy as `LocalModelRegistryClient` copying `best.pt`) alongside `lineage.json`
(classes, hash, `parent_version`, `workflow_id`, `source_batches`, `approved_by`,
validation/label-QA status). **Local filesystem backend only — Azure ML Data Asset
registration is not implemented** (separately tracked in the backlog under Phase 2). CLI:
`agentic-mlops version-dataset <dataset_path> --dataset-name ... [--validation-report ...]
[--label-quality-report ...] [--parent-version ...]`. Writes `dataset_version_report.json`/
`.md`. Deliberately **not** wired into `run-mvp` — same reasoning as the other Phase 0/3
agents.

### Model Decision Agent (standalone, not part of MVPWorkflow)

`ModelDecisionAgent`/`ModelDecider` (`agents/model_decision.py`, `tools/model_decider.py`,
`contracts/model_decision.py`) — the sixth of the 8 previously-unimplemented agents. Was
originally MVP step "4. Decision" per `01_mvp_scope.md`, but the actual implementation
folded threshold checks directly into `EvaluationAgent` + `HumanApprovalAgent` instead of
giving it a standalone agent; this backfills that gap **additively**, not by duplicating
existing logic. Reads `evaluation_report.json` (`EvaluationAgent` already ran
`workflows.policies.evaluate_metrics_against_policy` — **not recomputed here**) and maps
its 7-way `EvaluationRecommendation` onto the spec's 5-way `PROMOTE`/`REJECT`/`RETRAIN`/
`NEED_MORE_DATA`/`NEED_LABEL_REVIEW`. Adds two checks that existed only as unused Pydantic
fields until now: **baseline comparison** (`PromotionPolicy.require_improvement_over_baseline`
/`baseline_improvement_min_map50` — defined since the MVP, never read anywhere) and
**runtime budget** (`EvaluationConfig.runtime.max_latency_ms`/`max_model_size_mb` — also
defined since the MVP, also never read anywhere). Either check failing downgrades a
`PROMOTE` to `RETRAIN` — never the reverse, and never further downgrades an already
non-`PROMOTE` decision. Does not benchmark inference itself — accepts an externally
measured `measured_latency_ms`. Never approves anything — `HumanApprovalAgent` remains
the sole approval gate (matches the spec's "no auto-production-promote" rule). CLI:
`agentic-mlops model-decision <evaluation_report_path> [--promotion-policy ...]
[--evaluation-config ...] [--baseline-report ...] [--measured-latency-ms ...]`. Writes
`decision_report.json`/`.md`. Deliberately **not** wired into `run-mvp`.

### MVPWorkflow factory injection

`MVPWorkflow.__init__` accepts five `_xxx_factory: Callable[[Path], Agent] | None` parameters (`_validation_factory`, `_training_factory`, `_evaluation_factory`, `_approval_factory`, `_registry_factory`), each defaulting to a lambda that also injects MLflow when a parent run is active. This enables test overrides without any mock framework — pass a lambda returning a stub instead.

### Testing conventions

- `conftest.py` exports plain helper functions (`make_valid_dataset`, `make_image`, `make_label`, `make_data_yaml`) — **not** `@pytest.fixture` decorated; tests import and call them directly with `tmp_path`
- `make_image` writes a minimal JPEG stub (magic bytes + filename bytes + EOI) sufficient for hash-based duplicate detection
- Azure/MLflow calls are replaced with `FakeAzureMLClientFactory` (+ `FakeMLClient`) / `FakeMLflowTrackingClient` / `FakeModelRegistryClient` injected via constructor
- `MVPWorkflow` tests inject `_StubAgent` / `_RaisingAgent` factories — never use real agents in workflow tests
- CLI is tested via `typer.testing.CliRunner`

### Design documentation

`agentic_mlops_workflow_docs/` contains the full agent and architecture specs:
- `agents/` — 13 markdown specs (00–12) covering all planned agents, including the 2 not yet implemented (see below)
- `docs/` — 14 architecture docs (state machine, MVP scope, data contracts, etc.)
- `prompts/` — Claude Code prompts used to bootstrap this project

### What is not yet implemented

As of 2026-07-10, all 5 MVP agents are fully implemented and tested, including every
runner variant: `fake`/`local-yolo`/`azure-ml` for training and evaluation, and
`local`/`mlflow`/`azure_ml` for model registry (see
`agentic_mlops_workflow_docs/docs/13_backlog.md` for the authoritative, actively-maintained
status of every planned item). Still missing:

- Azure ML pipeline components (multi-step AML pipeline instead of a single CommandJob per step)
- Registering the dataset itself as an Azure ML Data Asset; storing artifacts in Blob/ADLS instead of local disk
- VOC label format for Dataset Structuring (`LabelFormat` only has `yolo`/`coco`)
- Azure ML Data Asset registration for `DatasetVersioningAgent` (local filesystem backend only)
- 2 remaining agents described in `agentic_mlops_workflow_docs/agents/` — Deployment, Monitoring. `LabelQAAgent`, `DataIntakeAgent`, `DatasetStructuringAgent`, `AnnotationAgent`, `DatasetVersioningAgent`, and `ModelDecisionAgent` (see above) are the first six of the original 8 to be implemented. These correspond to Phase 4/5 in the backlog — not started.

Note: `agentic_mlops_workflow_docs/docs/` and `agentic_mlops_workflow_docs/agents/` are
design specs frozen at the project-bootstrap stage — they describe the full target
architecture (13 agents) the MVP is a subset of, and are **not** updated to track
implementation status. Trust `13_backlog.md` and this file, not the other docs in that
folder, for "is X implemented" questions.
