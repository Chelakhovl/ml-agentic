> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Training Agent

## Назначение

Training Agent запускает обучение YOLO локально или в Azure ML, логирует run в MLflow и сохраняет model artifacts.

## Что делает

```text
- читает training_config.yaml
- проверяет dataset version
- проверяет approval на compute budget
- запускает YOLO train
- логирует параметры и метрики в MLflow
- сохраняет best.pt / last.pt
- возвращает mlflow_run_id и job_id
```

## Inputs

```json
{
  "dataset_uri": "azureml:factory_defects:8",
  "data_yaml_path": "data.yaml",
  "training_config_path": "configs/training.yaml",
  "mode": "azure_train"
}
```

## Outputs

```json
{
  "azure_job_id": "azureml_job_123",
  "mlflow_run_id": "run_9283",
  "best_weights_path": "runs/wf_001/artifacts/best.pt",
  "training_artifacts_dir": "runs/wf_001/artifacts/training"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Ultralytics YOLO train | обучение модели |
| Azure ML SDK | запуск job |
| Azure ML Compute | GPU compute |
| MLflow | tracking metrics/artifacts |
| Key Vault | secrets |
| YAML config loader | training config |

## Training config example

```yaml
model: yolo11m.pt
data: data.yaml
epochs: 100
imgsz: 640
batch: 16
optimizer: auto
patience: 20
project: factory_defects
name: yolo11m_dataset_v8
```

## Human-in-the-loop

H4: Training Approval.

Человек подтверждает:

```text
- compute budget;
- GPU size;
- max training time;
- model size;
- training config.
```

## Failure cases

```text
- Azure credentials missing
- compute quota exceeded
- dataset path invalid
- YOLO training crashed
- MLflow tracking failed
```

## Acceptance Criteria

- Есть local_dry_run mode.
- Есть local_train mode.
- Есть azure_train mode через adapter.
- Все параметры логируются.
- Artifacts сохраняются.
- Ошибка Azure job не ломает audit log.
