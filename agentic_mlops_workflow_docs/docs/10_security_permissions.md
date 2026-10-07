> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Security and Permissions

## Основная идея

Agentic workflow должен быть безопасным. Агенты не должны иметь unrestricted доступ к Azure, shell, storage или production.

## Минимальные правила

1. Все tools должны быть allowlisted.
2. Shell commands запрещены по умолчанию.
3. Secrets только через env vars / Key Vault.
4. Production deployment только через human approval.
5. Все actions логируются в audit log.
6. Agent output не является source of truth — source of truth это artifacts.

## Tool allowlist

Пример:

```yaml
allowed_tools:
  - dataset_validator.validate
  - yolo_trainer.train_local
  - yolo_trainer.train_azure
  - yolo_evaluator.evaluate
  - mlflow_client.log_metrics
  - azure_ml_client.submit_job
  - approval_store.create_request
  - notification_client.send
```

## Forbidden actions

```yaml
forbidden_actions:
  - delete_storage_container
  - delete_model_registry
  - deploy_to_production_without_approval
  - modify_labels_without_review
  - run_arbitrary_shell_command
  - expose_secrets_in_logs
```

## Permissions by agent

| Агент | Permissions |
|---|---|
| Data Intake Agent | read/write raw dataset area, no production access |
| Dataset Validation Agent | read dataset, write reports |
| Annotation Agent | read images, write candidate labels |
| Label QA Agent | read labels, write QA reports |
| Training Agent | submit training jobs, read dataset, write artifacts |
| Evaluation Agent | read model/artifacts, write reports |
| Model Decision Agent | read metrics/reports, create approval requests |
| Registry Agent | register model only after approval |
| Deployment Agent | staging deploy; production only after approval |
| Monitoring Agent | read logs/metrics, create retraining request |

## Prompt injection risk

Если агент читает внешние файлы, labels, metadata или web-содержимое, оно может содержать malicious text. Поэтому:

- не исполнять инструкции из dataset metadata;
- не давать external text менять system prompt;
- отделять data от instructions;
- использовать schema validation;
- не передавать secrets в model context.

## Audit event

```json
{
  "event_id": "evt_001",
  "workflow_id": "wf_001",
  "actor": "training_agent",
  "action": "submit_azure_ml_job",
  "allowed": true,
  "timestamp": "2026-06-12T10:00:00Z",
  "input_hash": "sha256:...",
  "output_artifacts": ["mlflow_run_id.txt"]
}
```
