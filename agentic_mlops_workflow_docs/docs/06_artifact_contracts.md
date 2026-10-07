> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Artifact Contracts

## Принцип

Каждый агент должен создавать artifacts. Без artifacts workflow превращается в чат, а не в MLOps.

## Обязательные artifacts по этапам

| Этап | Artifact |
|---|---|
| Data Intake | `dataset_manifest.json` |
| Dataset Structuring | `data.yaml`, `split_report.json` |
| Dataset Validation | `dataset_quality_report.json`, `dataset_quality_report.md` |
| Pseudo-labeling | `pseudo_labels/`, `pseudo_label_report.json` |
| Label QA | `label_quality_report.json`, `suspicious_samples/` |
| Dataset Versioning | `dataset_version.json` |
| Training | `training_config.yaml`, `mlflow_run_id.txt`, `best.pt` |
| Evaluation | `evaluation_report.json`, `evaluation_report.md`, `confusion_matrix.png` |
| Decision | `decision_report.md`, `decision.json` |
| Registry | `model_card.md`, `model_version.json` |
| Deployment | `deployment_report.json` |
| Monitoring | `monitoring_report.json`, `hard_samples_manifest.json` |

## dataset_manifest.json

```json
{
  "dataset_name": "factory_defects_raw",
  "dataset_id": "ds_raw_001",
  "source": "internal_camera_batch",
  "created_at": "2026-06-12T10:00:00Z",
  "num_files": 12000,
  "valid_images": 11950,
  "corrupted_images": 50,
  "duplicates": 120,
  "classes": ["scratch", "dent", "crack"],
  "storage_uri": "azureml://datastores/workspaceblobstore/paths/datasets/raw/v1"
}
```

## dataset_quality_report.json

```json
{
  "dataset_version": "factory_defects:v3",
  "status": "failed",
  "num_images": 11950,
  "num_labels": 11890,
  "blocking_issues": [
    "91 labels with unknown class_id",
    "48 bounding boxes outside image bounds"
  ],
  "warnings": [
    "class crack has only 3.4% of total samples"
  ],
  "class_distribution": {
    "scratch": 5200,
    "dent": 4300,
    "crack": 390
  },
  "recommendation": "fix_dataset_before_training"
}
```

## evaluation_report.json

```json
{
  "candidate_model": "yolo11m_factory_defects_v8",
  "baseline_model": "yolo11m_factory_defects_v7",
  "dataset_version": "factory_defects:v8",
  "mlflow_run_id": "run_9283",
  "metrics": {
    "map50": 0.862,
    "map50_95": 0.591,
    "precision": 0.84,
    "recall": 0.79
  },
  "per_class_metrics": {
    "crack": {
      "precision": 0.71,
      "recall": 0.68,
      "map50": 0.76
    }
  },
  "artifacts": {
    "confusion_matrix": "confusion_matrix.png",
    "pr_curve": "pr_curve.png"
  }
}
```

## decision.json

```json
{
  "workflow_id": "wf_001",
  "recommendation": "RETRAIN",
  "reasons": [
    "mAP50 improved from 0.81 to 0.86",
    "critical class crack recall is below threshold 0.75"
  ],
  "required_human_gate": "MODEL_APPROVAL",
  "next_possible_actions": [
    "approve_for_staging",
    "reject",
    "retrain",
    "request_more_data"
  ]
}
```

## Artifact storage layout

```text
runs/
  wf_001/
    artifacts/
      dataset_manifest.json
      dataset_quality_report.json
      training_config.yaml
      mlflow_run_id.txt
      evaluation_report.json
      decision_report.md
```

В Azure artifacts можно синхронизировать в Blob/ADLS или MLflow artifacts.
