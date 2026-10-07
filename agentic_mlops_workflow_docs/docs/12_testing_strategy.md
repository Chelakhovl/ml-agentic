> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Testing Strategy

## Что тестировать

Agentic MLOps workflow должен тестироваться на трех уровнях:

1. unit tests для tools и policies;
2. integration tests для локального workflow;
3. smoke tests для Azure ML integration.

## Unit tests

### Dataset validator

Проверить:

- валидный YOLO dataset проходит;
- label с неправильным class_id падает;
- bbox вне диапазона падает;
- пустой label обрабатывается корректно;
- отсутствующий image обнаруживается;
- дубликаты между train/val находятся.

### Decision policy

Проверить:

- метрики выше threshold → PROMOTE candidate;
- critical class recall ниже threshold → RETRAIN или REJECT;
- latency выше threshold → REJECT;
- missing metrics → NEED_HUMAN_REVIEW.

### State machine

Проверить:

- illegal transition запрещен;
- human approval state блокирует workflow;
- failed validation возвращает в cleanup;
- monitoring drift возвращает в data intake.

## Integration tests

Локальный dry-run:

```bash
agentic-mlops run mvp \
  --dataset-path tests/fixtures/yolo_dataset_valid \
  --mode local_dry_run
```

Ожидаемый результат:

```text
runs/<workflow_id>/artifacts/dataset_quality_report.json
runs/<workflow_id>/artifacts/training_plan.json
runs/<workflow_id>/artifacts/evaluation_report.json
runs/<workflow_id>/artifacts/decision_report.md
```

## Azure smoke tests

Запускаются только если есть env vars:

```text
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_ML_WORKSPACE
```

Smoke test:

- connect workspace;
- submit tiny job;
- get status;
- write artifact;
- cleanup temporary resources if safe.

## Test fixtures

```text
tests/fixtures/
  yolo_dataset_valid/
  yolo_dataset_bad_class_id/
  yolo_dataset_bad_bbox/
  yolo_dataset_duplicates/
  evaluation_reports/
    good_model.json
    bad_recall_model.json
```

## CI requirements

CI должен запускать:

```bash
ruff check .
pytest tests/unit
pytest tests/integration --ignore-azure
```
