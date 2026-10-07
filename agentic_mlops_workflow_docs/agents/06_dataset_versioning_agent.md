> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Dataset Versioning Agent

## Назначение

Dataset Versioning Agent регистрирует clean dataset как новую версию и сохраняет lineage.

## Что делает

```text
- фиксирует dataset version
- считает dataset hash
- сохраняет manifest
- сохраняет split info
- сохраняет class mapping
- регистрирует Azure ML Data Asset
- связывает dataset с workflow_id
```

## Inputs

```json
{
  "dataset_path": "runs/wf_001/dataset",
  "dataset_name": "factory_defects",
  "parent_version": "v7",
  "validation_report_path": "dataset_quality_report.json",
  "label_quality_report_path": "label_quality_report.json"
}
```

## Outputs

```json
{
  "dataset_name": "factory_defects",
  "dataset_version": "v8",
  "azure_ml_data_asset_uri": "azureml:factory_defects:8",
  "dataset_version_artifact": "dataset_version.json"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Azure ML Data Assets | dataset version registration |
| Azure Blob / ADLS | storage |
| hash calculator | reproducibility |
| manifest writer | lineage metadata |
| MLflow artifact logger | сохранить metadata |

## Lineage fields

```json
{
  "dataset_name": "factory_defects",
  "version": "v8",
  "parent_version": "v7",
  "workflow_id": "wf_001",
  "source_batches": ["batch_2026_06_01"],
  "classes": ["scratch", "dent", "crack"],
  "hash": "sha256:...",
  "approved_by": "ml_engineer"
}
```

## Human-in-the-loop

Обычно не нужен, если validation и label QA прошли.

Нужен, если:

```text
- меняется test set;
- меняется taxonomy;
- dataset становится official benchmark.
```

## Acceptance Criteria

- Dataset version создана.
- Lineage сохранен.
- Dataset можно передать в Training Agent.
- При одинаковом input hash не создаются лишние версии без необходимости.
