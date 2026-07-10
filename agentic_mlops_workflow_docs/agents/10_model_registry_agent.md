# Model Registry Agent

## Назначение

Model Registry Agent регистрирует approved model, сохраняет lineage и model card.

## Что делает

```text
- проверяет human approval
- регистрирует модель в Azure ML / MLflow Model Registry
- сохраняет model version
- связывает модель с dataset version
- связывает модель с mlflow run id
- создает model_card.md
```

## Inputs

```json
{
  "approval_id": "appr_001",
  "weights_path": "best.pt",
  "mlflow_run_id": "run_9283",
  "dataset_version": "factory_defects:v8",
  "model_name": "factory_defects_detector"
}
```

## Outputs

```json
{
  "model_name": "factory_defects_detector",
  "model_version": "12",
  "registry_uri": "azureml:factory_defects_detector:12",
  "model_card_path": "model_card.md"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Azure ML Model Registry | register model |
| MLflow Model Registry | optional registry |
| model card generator | documentation |
| lineage tracker | dataset/run/code links |
| approval store | verify approval |

## Model card fields

```text
- model name/version
- task: object detection
- classes
- dataset version
- training config
- evaluation metrics
- limitations
- approval info
- intended use
- not intended use
```

## Human-in-the-loop

Модель можно регистрировать только после H5 approval.

## Acceptance Criteria

- Модель не регистрируется без approval.
- Model card создается.
- Lineage сохранен.
- Registry output machine-readable.
