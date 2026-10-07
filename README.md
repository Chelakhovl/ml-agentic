# Agentic MLOps — YOLO

Sequential multi-agent MLOps pipeline for YOLO object detection, from raw data to production deployment.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/your-org/agentic-mlops-yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/agentic-mlops-yolo/actions/workflows/ci.yml)

---

## Overview

Agentic MLOps — YOLO is a production-grade, configurable MLOps pipeline for YOLO object detection
models. It implements 13 agents across the full ML lifecycle — data intake through production
deployment — with 3 compute backends (`fake` / `local-yolo` / `azure-ml`), 3 registry backends
(`local` / `mlflow` / `azure_ml`), a FastAPI web dashboard, and a complete Typer CLI. Every
stage boundary is typed via Pydantic v2 contracts; agents orchestrate and report, while all
deterministic computation lives in injected tools. The pipeline supports persistent state,
audit logs, resume-after-pause, and automated webhook notifications on completion.

---

## Quick Start

```bash
pip install -e ".[dev]"

# Dry-run the full 5-step MVP pipeline (no YOLO, no Azure, no MLflow needed)
agentic-mlops run-mvp \
  --dataset-path /path/to/yolo_dataset \
  --data-yaml /path/to/data.yaml \
  --training-config configs/training.example.yaml \
  --output-dir ./runs/workflow_001 \
  --dry-run --no-interactive
```

Expected YOLO dataset layout:

```
my_dataset/
  images/
    train/    # .jpg / .png
    val/
  labels/
    train/    # YOLO .txt — one per image
    val/
  data.yaml
```

`data.yaml` minimal example:

```yaml
path: .
train: images/train
val: images/val
names:
  0: scratch
  1: dent
  2: crack
```

---

## Full Pipeline

The Orchestrator supports any ordered subset of the 11 canonical pipeline steps. The 5-step
MVP default (`dataset_validation` → `training` → `evaluation` → `approval` → `model_registry`)
is the same chain `run-mvp` runs; use `run-workflow` with a YAML config to enable any subset.

| # | Step | Agent | Gate / Description |
|---|---|---|---|
| 1 | `data_intake` | `DataIntakeAgent` | Format, corruption, and duplicate checks on a raw image directory |
| 2 | `dataset_structuring` | `DatasetStructuringAgent` | Converts raw images + YOLO/COCO/VOC labels into a split YOLO dataset |
| 3 | `dataset_validation` | `DatasetValidationAgent` | Validates YOLO structure, bbox ranges, and cross-split duplicates |
| 4 | `dataset_versioning` | `DatasetVersioningAgent` | Hash-deduplicating local registry or Azure ML Data asset registration |
| 5 | `training_approval` | `TrainingApprovalAgent` | **H4 gate** — human approve/reject before training starts |
| 6 | `training` | `TrainingAgent` | `fake` / `local-yolo` / `azure-ml` / `azure-ml-pipeline` |
| 7 | `evaluation` | `EvaluationAgent` | Runs promotion policy; blocked if training failed |
| 8 | `model_decision` | `ModelDecisionAgent` | Maps evaluation recommendation → promote/reject/retrain/need-more-data |
| 9 | `approval` | `HumanApprovalAgent` | **H5 gate** — human model approval (interactive or non-interactive) |
| 10 | `model_registry` | `ModelRegistryAgent` | 5-gate check → registers `best.pt` locally, MLflow, or Azure ML |
| 11 | `deployment` | `DeploymentAgent` | ONNX/pt export + smoke test → local release, Docker/AKS, or Azure ML endpoint |

Run the full configurable pipeline with a single YAML config:

```bash
cp configs/orchestrator.example.yaml configs/orchestrator.yaml
# Edit steps, runners, registry backends, notifications, etc.

agentic-mlops run-workflow \
  --workflow-id wf_001 \
  --config configs/orchestrator.yaml \
  --runs-dir runs

# Resume after a human approval pause
agentic-mlops run-workflow \
  --workflow-id wf_001 \
  --config configs/orchestrator.yaml \
  --runs-dir runs --resume

# With MLflow tracking (one parent run across the whole workflow)
agentic-mlops run-workflow \
  --workflow-id wf_001 \
  --config configs/orchestrator.yaml \
  --runs-dir runs \
  --enable-mlflow --mlflow-config configs/mlflow.example.yaml
```

Each run writes `runs/<workflow_id>/state.json` (resumable snapshot) and
`runs/<workflow_id>/audit_log.jsonl` (append-only event log). Pausing at an H4/H5 gate
leaves the run open — re-run with `--resume` once the decision JSON exists on disk.

**Resume and rewind:**

```bash
# Resume from where it paused (e.g. after H4/H5 approval):
agentic-mlops run-workflow --workflow-id wf_001 --config configs/orchestrator.yaml --resume

# Rewind to a specific step and re-run from there (prior step outputs are preserved):
agentic-mlops run-workflow --workflow-id wf_001 --config configs/orchestrator.yaml \
  --resume-from-step evaluation
```

**Notifications — configure in `configs/orchestrator.yaml`:**

```yaml
notifications:
  teams_webhook_url: https://...   # Teams MessageCard with interactive approval buttons
  slack_webhook_url: https://...   # Slack attachment with approve/reject buttons
  email_to: team@example.com       # SMTP — no extra packages (stdlib smtplib)
  smtp_host: smtp.example.com
  notify_on: [step_failed, workflow_completed, workflow_failed]
```

**GitHub Actions auto-deploy on completion:**

```yaml
github_actions:
  repo: your-org/your-repo
  workflow_file: mlops-deploy.yml
  # GITHUB_ACTIONS_TOKEN env var (PAT with actions:write)
```

**Azure ML state backend (horizontal scaling):**

```bash
agentic-mlops run-workflow --workflow-id wf_001 --config configs/orchestrator.yaml \
  --state-backend azure_blob --azure-config configs/azure_ml.yaml
```

---

## Extras / Optional Dependencies

| Extra | Install | When needed |
|---|---|---|
| `dev` | pytest, ruff, typer, pydantic, etc. | Local development and testing |
| `mlflow` | `mlflow` | `--enable-mlflow`, `--backend mlflow` registry |
| `azure` | `azure-ai-ml`, `azure-identity`, `azure-monitor-query` | Any `--runner azure-ml`, `--backend azure_ml`, App Insights log ingestion |
| `vision` | `Pillow` | Real image corruption detection in `DataIntakeAgent` (falls back to non-zero-size check without it) |
| `web` | `fastapi`, `uvicorn`, `jinja2` | `agentic-mlops serve` web dashboard |
| `auth` | `python-jose[cryptography]`, `httpx` | OIDC/JWT + API-key authentication for the web dashboard |

Install multiple extras together:

```bash
pip install -e ".[dev,azure,web]"

# Everything at once:
pip install -e ".[dev,mlflow,azure,vision,web,auth]"
```

---

## CLI Reference

Run `agentic-mlops <command> --help` for all flags. See [`CLAUDE.md`](CLAUDE.md) for worked
examples and every runner/backend option.

### Data Preparation

| Command | Description |
|---|---|
| `agentic-mlops data-intake <raw_data_path> --dataset-name foo` | Scan raw images: format, corruption, SHA-256 duplicate checks; writes `dataset_manifest.json` |
| `agentic-mlops structure-dataset <raw_data_path> --output-dataset-path ./out --classes a,b` | Convert raw images + labels (YOLO/COCO/VOC) into a split YOLO dataset |
| `agentic-mlops pseudo-label <images_path> --model-path models/best.pt` | Run an approved YOLO model over unlabeled images; routes results into high/medium/low confidence buckets |
| `agentic-mlops label-qa <dataset_path>` | Check bbox geometry, class imbalance, and optional reference-model disagreement |

### Dataset Management

| Command | Description |
|---|---|
| `agentic-mlops validate-dataset <dataset_path>` | Validate YOLO structure, labels, bbox ranges, cross-split duplicates |
| `agentic-mlops version-dataset <dataset_path> --dataset-name foo` | Register a clean dataset version (local hash-dedup, or `--backend azure_ml` for Azure ML Data asset) |

### Training & Evaluation

| Command | Description |
|---|---|
| `agentic-mlops train --dataset-path ... --runner fake\|local-yolo\|azure-ml` | Train a YOLO model; `fake` writes a plan only |
| `agentic-mlops evaluate --dataset-path ... --runner fake\|local-yolo\|azure-ml` | Evaluate and apply the promotion policy |
| `agentic-mlops model-decision <evaluation_report.json>` | Map evaluation recommendation → explainable promote/reject/retrain decision |
| `agentic-mlops approve --evaluation-output <path> --action approve_model --no-interactive` | Record H5 human approval (interactive by default) |
| `agentic-mlops approve-training <dataset_quality_report.json> --action approve_training --no-interactive` | Record H4 training-start approval |
| `agentic-mlops register-model --model-name foo --backend local\|mlflow\|azure_ml` | Register an approved model (5-gate check) |

### Deployment

| Command | Description |
|---|---|
| `agentic-mlops deploy-model <model.pt> --model-name foo --export-format onnx` | Export + smoke-test + release locally; add `--target production` for H6 gate |
| `agentic-mlops deploy-model --model-name foo --backend azure_ml --azure-config configs/azure_ml.yaml` | Deploy to a real Azure ML Managed Online Endpoint |
| `agentic-mlops deploy-model <model.pt> --model-name foo --backend docker --docker-config configs/docker.yaml` | Build a Docker image and push to a registry |
| `agentic-mlops deploy-model <model.pt> --model-name foo --backend aks --docker-config ... --aks-config ...` | Deploy to AKS via `kubectl apply` |
| `agentic-mlops deploy-model ... --canary-percentage 20` | Canary rollout — send 20% traffic to new deployment, 80% to incumbent (Azure ML backend) |
| `agentic-mlops monitor <predictions.jsonl> --endpoint-name foo` | Analyze inference logs for drift, latency, error rate; writes `hard_samples_manifest.json` |
| `agentic-mlops ingest-hard-samples <manifest.json> --images-source-dir ... --dataset-name foo` | Stage hard samples and run data intake over them |
| `agentic-mlops compare-models eval_a.json eval_b.json --model-names a,b` | Rank and compare multiple evaluation reports |

### Pipelines

| Command | Description |
|---|---|
| `agentic-mlops run-mvp --dataset-path ... --dry-run` | Fixed 5-step chain: validate → train → evaluate → approve → register |
| `agentic-mlops run-workflow --workflow-id wf_001 --config configs/orchestrator.yaml` | Configurable 1–11 step pipeline with persistent state, audit log, and resume |

### Web Dashboard

| Command | Description |
|---|---|
| `agentic-mlops serve --port 8000 --runs-dir runs` | Start the FastAPI web dashboard (requires `[web]` extra) |

### Operations

| Command | Description |
|---|---|
| `agentic-mlops doctor` | System health self-check: packages, configs, dirs, optional tools (docker/az CLI) |
| `agentic-mlops status` | Quick system-wide snapshot: active workflows, registered models, datasets, alerts |
| `agentic-mlops show-state wf_001` | Drill-down view of a single workflow run state and steps |
| `agentic-mlops diff-runs wf_001 wf_002` | Side-by-side comparison of two workflow run states and metrics |
| `agentic-mlops cost-report wf_001 --runs-dir runs` | Estimate per-step compute costs for a workflow run |
| `agentic-mlops prune-runs --keep-last 10` | Delete old workflow run directories |
| `agentic-mlops tag-run wf_001 env=prod model=yolov8n` | Add, remove, or display tags on a workflow run |
| `agentic-mlops lint-config configs/orchestrator.yaml` | Validate an orchestrator YAML without running anything |
| `agentic-mlops lint-docs agentic_mlops_workflow_docs --src-dir src` | Check spec docs have living-spec headers and reference valid Python paths/classes |
| `agentic-mlops watch wf_001` | Tail a workflow's audit log live (Ctrl-C to stop; exits 0=completed, 1=failed) |

---

## Operations & Observability

Utility commands for inspecting, managing, and troubleshooting workflow runs.
All run-inspection commands default to `--runs-dir runs`.

### System health

```bash
# Check packages, config files, directories, and optional tools (docker, az CLI)
agentic-mlops doctor

# Include orchestrator + Azure config validation
agentic-mlops doctor --config configs/orchestrator.yaml --azure-config configs/azure_ml.yaml

# Skip docker/az CLI binary checks
agentic-mlops doctor --no-tools

# Quick system-wide snapshot: active workflows, registered models, datasets, alerts
agentic-mlops status
agentic-mlops status --runs-dir runs --registry-dir outputs/model_registry --limit 10
```

### Inspecting a single run

```bash
# Drill-down view of state and all step outputs
agentic-mlops show-state wf_001

# Include the last 20 lines of the audit log
agentic-mlops show-state wf_001 --audit --audit-lines 20
```

### Watching live

```bash
# Tail the audit log in real time (Ctrl-C to stop)
# Exit codes: 0 = completed, 1 = failed, 2 = not found
agentic-mlops watch wf_001
agentic-mlops watch wf_001 --runs-dir runs --interval 0.5

# Keep tailing after the workflow reaches a terminal state
agentic-mlops watch wf_001 --follow
```

### Comparing runs

```bash
# Side-by-side metrics and step-status comparison
agentic-mlops diff-runs wf_001 wf_002

# Rank and compare multiple evaluation reports; write results to a directory
agentic-mlops compare-models \
  ./runs/eval_a/evaluation_report.json \
  ./runs/eval_b/evaluation_report.json \
  --model-names model_a,model_b \
  --output-dir ./runs/comparison
```

### Tagging runs

```bash
# Add tags
agentic-mlops tag-run wf_001 env=prod model=yolov8n

# Remove a tag
agentic-mlops tag-run wf_001 --remove env

# Display current tags (no arguments beyond the run id)
agentic-mlops tag-run wf_001
```

### Cost reporting

```bash
# Estimate per-step USD compute costs for a run
agentic-mlops cost-report wf_001 --runs-dir runs

# With a custom pricing config; write a JSON cost file
agentic-mlops cost-report wf_001 --pricing-config configs/pricing.yaml --output-file cost.json
```

### Config linting

```bash
# Validate an orchestrator YAML without running anything
agentic-mlops lint-config configs/orchestrator.yaml

# Strict mode — treat warnings as errors
agentic-mlops lint-config configs/orchestrator.yaml --strict
```

### Run housekeeping

```bash
# Keep only the 10 most recent runs; delete the rest
agentic-mlops prune-runs --keep-last 10 --runs-dir runs

# Dry-run: show which runs older than 30 days with a failed status would be deleted
agentic-mlops prune-runs --older-than-days 30 --status failed --dry-run
```

---

## Deployment Backends

`deploy-model` supports four backends. The `local` (default) and `azure_ml` backends are
described in the [Deployment](#deployment) CLI section. The two container backends are:

### Docker

Build a Docker image containing the model and push it to a registry:

```bash
cp configs/docker.example.yaml configs/docker.yaml
# Edit: image_name, registry, base_image, etc.

agentic-mlops deploy-model outputs/model_registry/my-model/versions/1/model/best.pt \
  --model-name my-model \
  --backend docker \
  --docker-config configs/docker.yaml
```

### AKS

Deploy to an Azure Kubernetes Service cluster via `kubectl apply` (requires Docker build first):

```bash
cp configs/aks.example.yaml configs/aks.yaml
# Edit: namespace, replicas, resource limits, etc.

agentic-mlops deploy-model outputs/model_registry/my-model/versions/1/model/best.pt \
  --model-name my-model \
  --backend aks \
  --docker-config configs/docker.yaml \
  --aks-config configs/aks.yaml
```

Both backends are also available as `deployment_backend: docker` / `deployment_backend: aks`
in `configs/orchestrator.yaml` for fully automated pipelines via `run-workflow`.

---

## Azure ML Setup

```bash
pip install -e ".[dev,azure]"
az login   # or set service principal env vars:
#   AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET

cp configs/azure_ml.example.yaml configs/azure_ml.yaml
# Edit: subscription_id, resource_group, workspace_name, compute_name, environment_name
```

Run the full pipeline on Azure ML (training + evaluation + Azure ML registry + endpoint):

```bash
agentic-mlops run-workflow \
  --workflow-id wf_azure_001 \
  --config configs/orchestrator.yaml \
  --runs-dir runs
```

Set `training_runner: azure-ml`, `evaluation_runner: azure-ml`, `registry_backend: azure_ml`,
and `deployment_backend: azure_ml` in `configs/orchestrator.yaml`. The `azure_model_name` and
`azure_model_version` are auto-chained from the `model_registry` step into `deployment` when
using `azure_ml` for both — no need to set them twice.

**Azure ML job scripts** (`azure_jobs/`) are self-contained (no `agentic_mlops` import) since
only that directory is uploaded as the job's code snapshot:
- `train_yolo.py` — Ultralytics training; outputs `best.pt`, `last.pt`, `results.csv`
- `eval_yolo.py` — Ultralytics evaluation; outputs `metrics.json` + confusion-matrix/PR-curve plots
- `score.py` — Online Endpoint scoring script

**Optional combined pipeline job** (`azure-ml-pipeline` runner): submits a 2-step
`PipelineJob` where the eval step receives `best.pt` directly from the training step's
data flow — no intermediate download/re-upload between train and eval.

---

## Web Dashboard

```bash
pip install -e ".[dev,web]"
agentic-mlops serve --port 8000 --runs-dir runs \
  --registry-dir outputs/model_registry \
  --dataset-registry-dir outputs/dataset_registry
```

**Authentication** (three independent layers, any combination):

```bash
# HTTP Basic Auth (no extra package):
DASHBOARD_PASSWORD=secret agentic-mlops serve --user admin --port 8000

# API key (Authorization: Bearer <token> or X-API-Key: <token>):
pip install -e ".[dev,web,auth]"
agentic-mlops serve --api-key my-secret-token --port 8000
# or via env var: DASHBOARD_API_KEY=my-secret-token

# OIDC / OAuth2 JWT (RS256/ES256, JWKS fetched at startup):
agentic-mlops serve \
  --oidc-issuer https://login.microsoftonline.com/<tenant>/v2.0 \
  --oidc-client-id <app-id> --port 8000
# env vars: DASHBOARD_OIDC_ISSUER, DASHBOARD_OIDC_CLIENT_ID

# Fail-fast if no auth method configured (recommended for production):
agentic-mlops serve --require-auth --api-key my-secret-token --port 8000
```

Dashboard features:
- Workflow list with live SSE status updates
- Per-workflow step accordion: inputs, outputs, artifacts, audit log
- Inline H4/H5 approval forms — submitting writes the decision and triggers `--resume` automatically
- Teams/Slack interactive approval buttons (when webhook + signing secret configured)
- Model registry browser with promotion lineage
- Dataset registry browser with version diff and quality charts
- Model comparison and ranked leaderboard pages
- Per-workflow cost report
- Hard sample viewer with ready-to-copy `ingest-hard-samples` command
- Dark mode + mobile responsive layout
- `/health` liveness probe, `/api/me` actor reflection, `/metrics` Prometheus endpoint

---

## Annotation / Pseudo-labeling

`AnnotationAgent` (`agents/annotation.py`) runs an approved YOLO model over unlabeled images
and writes candidate labels to a `pseudo_labels/` directory — never merged into an existing
dataset automatically (per the spec's safety rule: pseudo-labels require a review pass before
becoming final).

Confidence routing uses each image's weakest detection:

- `min_confidence >= auto_candidate_threshold` → `high` (still sample-audited by a human)
- `>= human_review_threshold` → `medium` (human review queue)
- below → `low` (hard/expert review)

```bash
agentic-mlops pseudo-label /path/to/unlabeled_images \
  --model-path models/approved/best.pt \
  --auto-candidate-threshold 0.8 \
  --existing-labels-path /path/to/human_labels  # skip already-labeled images
```

After review, feed accepted labels through `structure-dataset` → `version-dataset` →
`run-workflow` for the next training iteration.

---

## Active Learning Loop

The full feedback cycle is illustrated in `examples/active_learning_loop.py`:

1. **Monitor** a live endpoint — `agentic-mlops monitor predictions.jsonl` finds low-confidence
   and zero-detection images, writing `hard_samples_manifest.json`.
2. **Ingest hard samples** — `agentic-mlops ingest-hard-samples manifest.json` stages the raw
   images and runs `DataIntakeAgent` over them.
3. **Structure and version** — `agentic-mlops structure-dataset` + `agentic-mlops version-dataset`
   produce a new, clean dataset version with full lineage.
4. **Retrain** — `agentic-mlops run-workflow` launches the next training iteration, optionally
   chaining through all 11 pipeline steps including the H4 and H5 human gates.

The web dashboard's monitoring detail page shows a ready-to-copy `ingest-hard-samples` command
whenever hard samples are present in a run's monitoring report.

---

## Testing

```bash
# All unit tests (no Azure, no YOLO, no MLflow needed)
pytest tests/unit -v                                               # ~1400 tests

# Single test file
pytest tests/unit/test_dataset_validator.py -v

# With coverage
pytest tests/unit --cov=agentic_mlops

# Real Azure ML connectivity check (opt-in; auto-skipped without --azure-config)
pip install -e ".[dev,azure]"
az login
pytest tests/integration -m azure_integration \
  --azure-config configs/azure_ml.yaml
```

**Testing conventions:**
- `conftest.py` exports plain helper functions (`make_valid_dataset`, `make_image`, `make_label`,
  `make_data_yaml`) — not `@pytest.fixture`; tests call them directly with `tmp_path`.
- Azure/MLflow calls use `FakeAzureMLClientFactory`, `FakeMLflowTrackingClient`,
  `FakeModelRegistryClient`, `FakeNotificationClient` injected via constructor — no mock framework.
- `MVPWorkflow` tests inject `_StubAgent` / `_RaisingAgent` factories.
- CLI tests use `typer.testing.CliRunner`.

---

## CI / CD

**`ci.yml`** — triggered on every push and PR:
- Lint with `ruff check` and `ruff format --check`
- Unit test matrix on Python 3.11 and 3.12
- Azure smoke gate (read-only workspace connectivity check, gated behind a secret)

**`mlops-deploy.yml`** — `workflow_dispatch` template dispatched automatically by
`GithubActionsClient` when `OrchestratorWorkflow` reaches `COMPLETED`. Runs
`agentic-mlops deploy-model` for the promoted model. Accepts `workflow_dispatch` inputs:
`model_name`, `model_version`, `registry_backend` (`local` | `azure_ml`), and
`target` (`staging` | `production`). Configure the trigger via `github_actions:` block in
`configs/orchestrator.yaml`; set `GITHUB_ACTIONS_TOKEN` (PAT with `actions:write`) as an
environment variable or in `GithubActionsConfig.token`.

**`publish.yml`** — triggered on `v*` tags; publishes to PyPI via OIDC (no long-lived
token stored in secrets).

---

## Architecture

The full agent and architecture specs live in `agentic_mlops_workflow_docs/`:
- `agents/` — 13 markdown specs (00 Orchestrator + 01–12 individual agents)
- `docs/` — 14 architecture docs (state machine, MVP scope, data contracts, etc.)
- `prompts/` — Claude Code prompts used to bootstrap the project

**ML framework abstraction:**

`tools/model_runner.py` provides a `ModelRunner` protocol and three implementations:
- `YoloModelRunner` — full train + evaluate + predict via Ultralytics (default)
- `OnnxOnlyModelRunner` — inference-only via `onnxruntime`; `train`/`evaluate` raise `NotImplementedError`
- `TorchvisionModelRunner` — reserved stub for future PyTorch/TorchVision support

`TrainingInput`, `EvaluationInput`, and `AnnotationInput` all carry a `framework: ModelFramework`
field (default `"yolo"`). Use `create_model_runner(framework)` to get the right runner.

**Design principles:**
- Agents orchestrate and report; all ML computation is in deterministic, injected tools.
- Pydantic v2 contracts (`contracts/`) type every agent I/O boundary. `ToolResult` is the
  shared output base (`success`, `message`, `artifacts`, `warnings`, `errors`, `metadata`).
- `WorkflowStateStore` (`integrations/workflow_state_store.py`) writes `state.json` +
  `audit_log.jsonl` per run — no networked state store, per the spec's MVP note.
- `WorkflowState` StrEnum (14 states) in `contracts/common.py` governs legal transitions;
  illegal transitions raise `RuntimeError` (internal consistency check).
- All Azure ML integration uses SDK v2 (`azure-ai-ml`). Real Azure connections are always
  explicit — there is no implicit fallback path from `azure_ml` to `local`.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture walkthrough with worked examples for
every runner, backend, and integration.
