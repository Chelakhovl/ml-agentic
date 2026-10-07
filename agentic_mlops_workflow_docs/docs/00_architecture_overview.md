> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Architecture Overview

## Цель

Построить воспроизводимый Agentic MLOps workflow для object detection моделей на базе YOLO + Azure ML + MLflow.

Система должна автоматизировать:

1. прием новых изображений;
2. структурирование YOLO-датасета;
3. валидацию датасета и labels;
4. pseudo-labeling через pre-trained или approved model;
5. human label review;
6. версионирование датасета;
7. запуск обучения в Azure ML;
8. логирование в MLflow;
9. evaluation и сравнение с baseline;
10. human model approval;
11. регистрацию модели;
12. deployment;
13. monitoring и retraining loop.

## Архитектурный принцип

```text
Agents = orchestration / reasoning / routing / reporting
Tools = deterministic execution
Humans = approval / correction / final decision
```

Агент не должен “сам” обучать модель. Он должен вызвать инструмент:

```text
Training Agent → Azure ML job → YOLO train script → MLflow run
```

Агент не должен “сам” валидировать labels через LLM. Он должен вызвать deterministic validator:

```text
Dataset Validation Agent → Python validator → dataset_quality_report.json
```

## Слои системы

```text
Layer 1: User / Human Review UI
Layer 2: Agent Orchestrator
Layer 3: Agent Workers
Layer 4: Tool Layer
Layer 5: Azure ML / MLflow / Storage
Layer 6: Observability / Audit / Security
```

## Основной workflow

```text
New Data
 → Data Intake Agent
 → Dataset Structuring Agent
 → Dataset Validation Agent
 → Annotation / Pseudo-label Agent
 → Label QA Agent
 → Dataset Versioning Agent
 → Training Agent
 → Evaluation Agent
 → Model Decision Agent
 → Human Approval
 → Model Registry Agent
 → Deployment Agent
 → Monitoring Agent
 → back to Data Intake if drift/hard samples
```

## Ключевые artifacts

```text
dataset_manifest.json
data.yaml
dataset_quality_report.json
pseudo_labels/
label_quality_report.json
dataset_version.json
training_config.yaml
mlflow_run_id.txt
evaluation_report.json
decision_report.md
model_card.md
deployment_report.json
monitoring_report.json
```

## Non-goals на первом этапе

Не делать в MVP:

- full production deployment;
- автоматический auto-promote без человека;
- сложный UI;
- fully autonomous data collection from the internet;
- сложный multi-agent chat между всеми агентами;
- автоматическое изменение labels без human review.

## Target outcome

В результате система должна позволить ML engineer сделать так:

```bash
agentic-mlops run workflow \
  --dataset-path azureml://datastores/workspaceblobstore/paths/datasets/factory_defects/v1 \
  --config configs/workflow.yaml
```

И получить:

```text
- dataset validation report
- Azure ML training job
- MLflow run
- evaluation report
- recommendation: promote / reject / retrain / need more data
- human approval request
```
