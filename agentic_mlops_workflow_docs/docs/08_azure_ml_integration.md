> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Azure ML Integration

## Роль Azure ML

Azure ML используется для:

- запуска training jobs;
- запуска evaluation jobs;
- хранения data assets;
- model registry;
- managed online endpoints или integration с AKS;
- compute management;
- experiment tracking вместе с MLflow.

## Azure ML components

Рекомендуемые компоненты:

```text
validate_dataset_component
train_yolo_component
evaluate_model_component
register_model_component
```

## Component: validate_dataset

Inputs:

```yaml
inputs:
  dataset_path:
    type: uri_folder
  data_yaml:
    type: uri_file
```

Outputs:

```yaml
outputs:
  validation_report:
    type: uri_file
```

## Component: train_yolo

Inputs:

```yaml
inputs:
  dataset_path:
    type: uri_folder
  data_yaml:
    type: uri_file
  training_config:
    type: uri_file
```

Outputs:

```yaml
outputs:
  model_output:
    type: uri_folder
  training_artifacts:
    type: uri_folder
```

## Component: evaluate_model

Inputs:

```yaml
inputs:
  model_path:
    type: uri_file
  dataset_path:
    type: uri_folder
  data_yaml:
    type: uri_file
```

Outputs:

```yaml
outputs:
  evaluation_report:
    type: uri_file
  evaluation_artifacts:
    type: uri_folder
```

## Azure ML Adapter

В коде нужен adapter, чтобы agents не зависели напрямую от Azure SDK.

```python
class AzureMLClient:
    def submit_training_job(self, input: TrainingJobInput) -> TrainingJobOutput:
        ...

    def get_job_status(self, job_id: str) -> JobStatus:
        ...

    def register_data_asset(self, input: DataAssetInput) -> DataAssetOutput:
        ...

    def register_model(self, input: ModelRegistrationInput) -> ModelRegistrationOutput:
        ...
```

## Local fallback

Для разработки нужен режим без Azure:

```text
AZURE_MODE=disabled
```

В этом режиме:

- jobs не отправляются в Azure;
- создаются локальные fake job ids;
- artifacts сохраняются в `runs/`;
- integration tests могут проходить в CI.

## Secrets

Нельзя хранить credentials в repo.

Использовать:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_CLIENT_SECRET
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_ML_WORKSPACE
```

Для production — Managed Identity / Key Vault.
