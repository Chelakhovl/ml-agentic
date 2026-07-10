# Agentic MLOps — YOLO MVP

Agentic MLOps workflow for YOLO object detection.

**MVP scope:** Dataset Validation → Training → Evaluation → Human Approval → Model Registry.
Each stage can run as a dry-run (`fake`), locally (`local-yolo`), or on Azure ML (`azure-ml`) —
see [`CLAUDE.md`](CLAUDE.md) for the full architecture and every CLI command's options.

## Quick start

```bash
# Install in editable mode
pip install -e ".[dev]"

# Validate a YOLO dataset
agentic-mlops validate-dataset /path/to/my_yolo_dataset

# Override data.yaml location
agentic-mlops validate-dataset /path/to/dataset --data-yaml /path/to/data.yaml

# Save report to a custom directory
agentic-mlops validate-dataset /path/to/dataset --output-dir ./reports

# Treat warnings as errors
agentic-mlops validate-dataset /path/to/dataset --fail-on-warnings

# Run the full pipeline end-to-end in dry-run mode (no YOLO, no Azure, no MLflow needed)
agentic-mlops run-mvp \
  --dataset-path /path/to/dataset --data-yaml /path/to/data.yaml \
  --training-config configs/training.example.yaml \
  --output-dir ./runs/workflow_001 --dry-run --no-interactive
```

## Dataset format

```
my_dataset/
  images/
    train/   ← .jpg / .png images
    val/
  labels/
    train/   ← YOLO .txt labels (one per image)
    val/
  data.yaml
```

`data.yaml` example:

```yaml
path: .
train: images/train
val: images/val
names:
  0: scratch
  1: dent
  2: crack
```

Each label line: `<class_id> <x_center> <y_center> <width> <height>` (all normalised to `[0, 1]`).

## Running tests

```bash
pytest tests/unit -v
```

## Project structure

```
src/agentic_mlops/
  agents/           ← orchestration layer (5 MVP agents)
  contracts/        ← Pydantic I/O models
  tools/            ← dataset validator, YOLO trainer/evaluator, report writer
  integrations/     ← Azure ML client, MLflow tracking, model registry backends
  azure_jobs/       ← entry scripts submitted to Azure ML (train_yolo.py, eval_yolo.py)
  workflows/        ← MVPWorkflow, promotion policy
  observability/    ← structured logging
  cli/              ← Typer CLI
tests/
  unit/             ← 250 tests across all agents, tools, and integrations
  conftest.py       ← shared fixtures
configs/
  training.example.yaml
  promotion_policy.example.yaml
  evaluation.example.yaml
  mlflow.example.yaml
  azure_ml.example.yaml
```

## Commands

| Command | Description |
|---|---|
| `validate-dataset` | Validate a YOLO dataset locally — no Azure needed |
| `train` | Train a YOLO model: `fake` (dry-run plan) \| `local-yolo` (Ultralytics) \| `azure-ml` (Azure ML SDK v2) |
| `evaluate` | Evaluate a model and apply the promotion policy: `fake` \| `local-yolo` \| `azure-ml` |
| `approve` | Record a human approval decision (interactive or `--no-interactive --action ...`) |
| `register-model` | Register an approved model: `--backend local` (default) \| `mlflow` \| `azure_ml` |
| `run-mvp` | Chain all five agents end-to-end (validate → train → evaluate → approve → register) |

See `agentic-mlops <command> --help` for every flag, or [`CLAUDE.md`](CLAUDE.md) for a full
architecture walkthrough with worked examples for each runner and backend.

## Azure ML Training

Submit YOLO training jobs to Azure ML (SDK v2) with full log streaming, output download, and lineage tracking.

### Installation

```bash
pip install -e ".[azure]"
# or with dev + azure:
pip install -e ".[dev,azure]"
```

### Prerequisites

- Azure CLI: `az login` (or set service principal env vars `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`)
- An Azure ML workspace with a compute cluster
- An Azure ML environment with `ultralytics` installed (e.g. `azureml:yolo-training-env:1`)

### Configuration

```bash
cp configs/azure_ml.example.yaml configs/azure_ml.yaml
# Fill in: subscription_id, resource_group, workspace_name, compute_name
```

### Running a job

```bash
agentic-mlops train \
  --dataset-path /path/to/dataset \
  --data-yaml /path/to/data.yaml \
  --training-config configs/training.yaml \
  --runner azure-ml \
  --azure-config configs/azure_ml.yaml \
  --output-dir ./runs/azure_run_001
```

Expected output: `best.pt`, `last.pt`, `results.csv` downloaded to `--output-dir`, plus `azure_job_request.json` and `training_output.json` with Azure-specific fields (`azure_job_name`, `azure_studio_url`, etc.).

> **Cost note:** A single 1-epoch coco8 run on a Standard_NC6 GPU cluster takes approximately 15–30 minutes and costs roughly $0.50–$1.50 USD. Use `--runner fake` for dry runs.

### Azure Studio link

After submission, the job URL appears in `training_output.json` as `azure_studio_url`.

### Integration test

```bash
# Runs a real 1-epoch job — requires valid azure_ml.yaml and Azure auth
pytest -m azure_integration --azure-config configs/azure_ml.yaml
```

## Azure ML Evaluation

Run YOLO evaluation on Azure ML compute the same way, using the same `azure_ml.yaml`:

```bash
agentic-mlops evaluate \
  --dataset-path /path/to/dataset \
  --data-yaml /path/to/data.yaml \
  --training-output ./runs/azure_run_001/training_output.json \
  --runner azure-ml \
  --azure-config configs/azure_ml.yaml \
  --output-dir ./runs/azure_eval_001
```

Expected output: `metrics.json`-derived `evaluation_output.json`, confusion matrix / PR-curve
plots downloaded to `--output-dir`, plus the same promotion-policy recommendation as `local-yolo`.

## Extending

- Azure ML: see `src/agentic_mlops/integrations/azure_ml_client.py`, `src/agentic_mlops/tools/training_runner.py`, `src/agentic_mlops/tools/evaluation_runner.py`
- Azure job scripts: `src/agentic_mlops/azure_jobs/train_yolo.py`, `src/agentic_mlops/azure_jobs/eval_yolo.py`
- MLflow: see `src/agentic_mlops/integrations/mlflow_client.py`, `src/agentic_mlops/integrations/model_registry.py::MLflowModelRegistryClient`
