> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Orchestrator Agent

## Назначение

Orchestrator Agent — главный координатор workflow. Он не выполняет ML-логику сам, а управляет состояниями, вызывает нужных агентов, проверяет policies и создает human approval gates.

## Основная ответственность

```text
- запуск workflow
- хранение state
- routing между агентами
- проверка allowed transitions
- создание approval requests
- сбор ссылок на artifacts
- notifications
- audit log
```

## Inputs

```json
{
  "workflow_id": "wf_001",
  "trigger": "new_dataset",
  "dataset_uri": "azureml://datastores/workspaceblobstore/paths/datasets/raw/v1",
  "config_path": "configs/workflow.yaml"
}
```

## Outputs

```json
{
  "workflow_id": "wf_001",
  "current_state": "MODEL_APPROVAL_REQUIRED",
  "last_agent": "model_decision_agent",
  "artifacts": ["decision_report.md"],
  "pending_approval_id": "appr_001"
}
```

## Tools

| Tool | Назначение |
|---|---|
| State Store | хранить состояние workflow |
| Policy Engine | проверять transitions и rules |
| Approval Store | создавать approval requests |
| Notification Client | отправлять Teams/Slack/Email notifications |
| Azure ML Client | запускать Azure ML jobs через агентов |
| MLflow Client | получать run metadata |

## State transitions

Orchestrator должен проверять, что переходы легальны:

```text
DATASET_VALIDATION_RUNNING → DATASET_READY_FOR_VERSIONING
DATASET_VALIDATION_RUNNING → DATASET_NEEDS_CLEANUP
MODEL_DECISION_RUNNING → MODEL_APPROVAL_REQUIRED
MODEL_DECISION_RUNNING → TRAINING_APPROVAL_REQUIRED
```

## Human-in-the-loop

Orchestrator создает HITL-запросы:

```text
H1 Data Source Approval
H2 Dataset Cleanup Approval
H3 Label Review
H4 Training Approval
H5 Model Approval
H6 Production Release Approval
```

## Failure handling

Если агент падает:

```text
- сохранить error в audit log
- сохранить stack trace в artifacts, если безопасно
- перевести workflow в FAILED state
- отправить notification
- дать retry option
```

## MVP реализация

В MVP Orchestrator может быть простым Python class + JSON state file:

```text
runs/<workflow_id>/state.json
runs/<workflow_id>/audit_log.jsonl
```
