# MLflow Tracking

## Роль MLflow

MLflow — центральное место для:

- experiment tracking;
- parameters;
- metrics;
- artifacts;
- model lineage;
- comparison между runs.

## Что логировать

### Parameters

```text
model
epochs
imgsz
batch
optimizer
learning_rate
dataset_version
code_commit
training_mode
```

### Metrics

```text
train/box_loss
train/cls_loss
train/dfl_loss
val/box_loss
val/cls_loss
metrics/precision
metrics/recall
metrics/mAP50
metrics/mAP50-95
latency_ms
model_size_mb
```

### Artifacts

```text
best.pt
last.pt
results.csv
confusion_matrix.png
PR_curve.png
data.yaml
training_config.yaml
evaluation_report.json
decision_report.md
```

## Naming convention

Experiment:

```text
<object_detection_project_name>
```

Run name:

```text
<model_family>_<dataset_version>_<timestamp>
```

Пример:

```text
yolo11m_factory_defects_v8_20260612_103000
```

## Tags

```python
mlflow.set_tags({
    "project": "factory_defects",
    "task": "object_detection",
    "model_family": "yolo11",
    "dataset_version": "factory_defects:v8",
    "workflow_id": "wf_001",
    "git_commit": "abc123",
    "environment": "azure_ml"
})
```

## MLflow Adapter

Агенты не должны напрямую зависеть от MLflow API. Лучше сделать wrapper:

```python
class MLflowTrackingClient:
    def start_run(self, experiment_name: str, run_name: str) -> str:
        ...

    def log_params(self, params: dict) -> None:
        ...

    def log_metrics(self, metrics: dict) -> None:
        ...

    def log_artifact(self, path: str) -> None:
        ...

    def get_run_metrics(self, run_id: str) -> dict:
        ...
```

## Evaluation comparison

Evaluation Agent должен уметь сравнить candidate с baseline:

```text
candidate_run_id
baseline_run_id
```

Сравнивать:

- mAP50;
- mAP50-95;
- recall critical classes;
- precision critical classes;
- latency;
- model size.
