# Agentic MLOps Workflow для Object Detection

Цель проекта — построить **agentic orchestration layer** поверх Azure ML, MLflow и Ultralytics YOLO для автоматизации MLOps-процесса object detection: от приема данных и проверки датасета до обучения, evaluation, model approval, deployment и monitoring.

Главный принцип:

> Агенты не заменяют MLOps pipeline. Агенты управляют воспроизводимыми tools, jobs и pipelines, собирают evidence, маршрутизируют workflow и вызывают human-in-the-loop в ключевых decision points.

## Что внутри

```text
agentic_mlops_workflow_docs/
  README.md
  assets/
    agentic_mlops_workflow_diagram.png
  agents/
    00_orchestrator_agent.md
    01_data_intake_agent.md
    02_dataset_structuring_agent.md
    03_dataset_validation_agent.md
    04_annotation_pseudolabel_agent.md
    05_label_qa_agent.md
    06_dataset_versioning_agent.md
    07_training_agent.md
    08_evaluation_agent.md
    09_model_decision_agent.md
    10_model_registry_agent.md
    11_deployment_agent.md
    12_monitoring_agent.md
  docs/
    00_architecture_overview.md
    01_mvp_scope.md
    02_repository_structure.md
    03_workflow_state_machine.md
    04_human_in_the_loop.md
    05_tool_contracts.md
    06_artifact_contracts.md
    07_data_contracts.md
    08_azure_ml_integration.md
    09_mlflow_tracking.md
    10_security_permissions.md
    11_observability.md
    12_testing_strategy.md
    13_backlog.md
  prompts/
    CLAUDE_CODE_PROMPT.md
    CLAUDE_CODE_MVP_PROMPT.md
```

## Рекомендуемый старт

Начинать нужно не со всех 12 агентов сразу, а с MVP:

```text
Dataset Validation Agent
  → Training Agent
  → Evaluation Agent
  → Human Approval
```

После MVP можно добавить:

```text
Annotation / Pseudo-label Agent
Label QA Agent
Dataset Versioning Agent
Model Registry Agent
Deployment Agent
Monitoring Agent
```

## Базовый стек

| Зона | Рекомендуемые инструменты |
|---|---|
| Object Detection | Ultralytics YOLO |
| Training infra | Azure ML Compute / Azure ML Jobs / Azure ML Pipelines |
| Experiment tracking | MLflow |
| Dataset storage | Azure Blob Storage / ADLS |
| Dataset versioning | Azure ML Data Assets |
| Agents / workflow | Microsoft Agent Framework, LangGraph, Semantic Kernel или custom state machine |
| App/API layer | Python, FastAPI, Typer CLI |
| Contracts | Pydantic |
| Logs/monitoring | Azure Monitor, Application Insights, structured JSON logs |
| Secrets | Azure Key Vault / environment variables |

## Самое важное правило

LLM/Agent layer должен иметь доступ только к **allowlisted tools**. Нельзя давать агентам бесконтрольный доступ к Azure, storage, shell или production deployment.
