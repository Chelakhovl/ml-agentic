> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# MVP Scope

## MVP цель

Сделать минимальный рабочий Agentic MLOps workflow:

```text
Dataset Validation Agent
 → Training Agent
 → Evaluation Agent
 → Model Decision Agent
 → Human Approval
```

MVP должен работать локально в dry-run режиме и иметь возможность запускать реальные Azure ML jobs при наличии credentials.

## MVP функциональность

### 1. Dataset Validation

Вход:

```text
path/to/yolo_dataset
```

Выход:

```text
dataset_quality_report.json
```

Проверки:

- существует `data.yaml`;
- существуют `images/train`, `images/val`, `labels/train`, `labels/val`;
- каждый label-файл валиден;
- class_id входит в диапазон classes;
- bbox values находятся в диапазоне `0..1`;
- bbox width/height больше 0;
- нет пустых critical labels;
- нет дубликатов между train/val/test;
- есть class distribution report.

### 2. Training

Вход:

```text
dataset path
training_config.yaml
```

Выход:

```text
Azure ML job id
MLflow run id
best.pt
training artifacts
```

MVP режимы:

```text
local_dry_run: не запускает Azure, только печатает команду
local_train: запускает YOLO локально
azure_train: запускает Azure ML job
```

### 3. Evaluation

Вход:

```text
best.pt
data.yaml
test dataset
baseline metrics optional
```

Выход:

```text
evaluation_report.json
evaluation_report.md
confusion_matrix.png если доступно
```

Метрики:

- mAP50;
- mAP50-95;
- precision;
- recall;
- per-class metrics;
- latency если доступно;
- model size.

### 4. Decision

Вход:

```text
evaluation_report.json
promotion_policy.yaml
baseline_metrics.json optional
```

Выход:

```text
decision_report.md
recommendation: PROMOTE | REJECT | RETRAIN | NEED_MORE_DATA | NEED_LABEL_REVIEW
```

### 5. Human Approval

MVP может быть CLI-based:

```bash
agentic-mlops approve --workflow-id <id> --decision promote
agentic-mlops reject --workflow-id <id> --reason "Recall for crack is too low"
```

## MVP Acceptance Criteria

- Есть CLI entrypoint.
- Есть Pydantic-контракты для входов/выходов.
- Есть unit tests для dataset validator.
- Есть dry-run режим без Azure credentials.
- Есть structured logs.
- Все artifacts сохраняются в `runs/<workflow_id>/artifacts/`.
- Агент не вызывает shell напрямую без allowlist.
- Secrets не хардкодятся.
- README содержит пример запуска.

## MVP не должен делать

- production deploy;
- real-time monitoring;
- advanced label UI;
- automatic data scraping;
- automatic production approval.
