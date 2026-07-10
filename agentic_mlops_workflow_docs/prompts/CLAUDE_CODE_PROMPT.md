# Claude Code Prompt — старт реализации Agentic MLOps инфраструктуры

Ты — senior Python/MLOps engineer. Нужно создать production-oriented, но MVP-friendly репозиторий для **Agentic MLOps Workflow for Object Detection**, где агенты управляют процессом обучения YOLO моделей через Azure ML, MLflow и Human-in-the-loop gates.

## Контекст проекта

Мы обучаем object detection модели на Ultralytics YOLO. Сейчас процесс ручной:

1. собираем изображения;
2. раскладываем dataset в train/val/test;
3. проверяем labels;
4. делаем annotation или pseudo-labeling;
5. запускаем training в Azure ML;
6. смотрим MLflow/evaluation metrics;
7. вручную решаем, принимать модель или нет.

Нужно начать реализацию инфраструктуры, где агенты автоматизируют orchestration, но heavy logic выполняется deterministic tools и Azure ML pipeline jobs.

Главный принцип:

```text
Agents = orchestration / routing / decisions / reports
Tools = deterministic execution
Humans = approval / correction / final decisions
```

Не делай “магический LLM-проект”. Сделай инженерный Python repo с typed contracts, tools, agents, state machine, dry-run mode и тестами.

---

## MVP scope

Сначала реализуй MVP:

```text
Dataset Validation Agent
 → Training Agent
 → Evaluation Agent
 → Model Decision Agent
 → Human Approval через CLI
```

Не реализуй сразу full deployment, monitoring, CVAT/Azure Labeling UI. Создай для них интерфейсы/stubs и TODO.

---

## Требуемая структура репозитория

Создай такую структуру:

```text
agentic-mlops-yolo/
  README.md
  pyproject.toml
  .env.example
  .gitignore

  configs/
    workflow.example.yaml
    training.example.yaml
    promotion_policy.example.yaml
    azure_ml.example.yaml

  src/
    agentic_mlops/
      __init__.py

      cli/
        __init__.py
        main.py

      agents/
        __init__.py
        base.py
        orchestrator.py
        dataset_validation.py
        training.py
        evaluation.py
        model_decision.py
        data_intake.py
        dataset_structuring.py
        annotation_pseudolabel.py
        label_qa.py
        dataset_versioning.py
        model_registry.py
        deployment.py
        monitoring.py

      contracts/
        __init__.py
        common.py
        datasets.py
        training.py
        evaluation.py
        decisions.py
        approvals.py
        artifacts.py

      tools/
        __init__.py
        dataset_validator.py
        yolo_trainer.py
        yolo_evaluator.py
        decision_policy.py
        report_writer.py
        approval_store.py
        notification_client.py

      integrations/
        __init__.py
        azure_ml_client.py
        mlflow_client.py
        storage_client.py
        keyvault_client.py

      workflows/
        __init__.py
        state_machine.py
        workflow_runner.py
        policies.py

      observability/
        __init__.py
        logging.py

      security/
        __init__.py
        allowlist.py

  tests/
    unit/
      test_dataset_validator.py
      test_decision_policy.py
      test_state_machine.py
    fixtures/
      yolo_dataset_valid/
      yolo_dataset_bad_bbox/
      yolo_dataset_bad_class_id/
```

---

## Технологии

Используй:

```text
Python 3.11+
Pydantic v2
Typer для CLI
PyYAML
rich для красивого CLI output
pytest
ruff
structlog или стандартный logging в JSON формате
```

Опционально, но не обязательно сразу:

```text
mlflow
azure-ai-ml
ultralytics
opencv-python
Pillow
```

Важно: если этих библиотек нет, проект всё равно должен иметь dry-run mode и проходить unit tests.

---

## CLI requirements

Сделай CLI:

```bash
agentic-mlops validate-dataset --dataset-path ./data/yolo_dataset
agentic-mlops run-mvp --dataset-path ./data/yolo_dataset --mode local_dry_run
agentic-mlops approvals list
agentic-mlops approvals approve <approval_id> --comment "Approved"
agentic-mlops approvals reject <approval_id> --reason "Recall too low"
```

`run-mvp` должен делать:

```text
1. DatasetValidationAgent
2. TrainingAgent
3. EvaluationAgent
4. ModelDecisionAgent
5. create Human Approval request if needed
```

Artifacts сохранять в:

```text
runs/<workflow_id>/artifacts/
```

State сохранять в:

```text
runs/<workflow_id>/state.json
runs/<workflow_id>/audit_log.jsonl
```

---

## Contracts

Создай Pydantic models.

### Common

```python
class ArtifactRef(BaseModel):
    name: str
    path: str
    type: str

class AgentResult(BaseModel):
    success: bool
    message: str
    next_state: str | None = None
    artifacts: list[ArtifactRef] = []
    warnings: list[str] = []
    errors: list[str] = []
```

### Dataset validation

```python
class DatasetValidationInput(BaseModel):
    dataset_path: str
    data_yaml_path: str | None = None

class DatasetValidationReport(BaseModel):
    status: Literal["passed", "failed", "warning"]
    num_images: int
    num_labels: int
    class_distribution: dict[str, int]
    blocking_issues: list[str]
    warnings: list[str]
```

### Training

```python
class TrainingInput(BaseModel):
    dataset_path: str
    data_yaml_path: str
    training_config_path: str
    mode: Literal["local_dry_run", "local_train", "azure_train"] = "local_dry_run"

class TrainingOutput(BaseModel):
    mlflow_run_id: str | None = None
    azure_job_id: str | None = None
    best_weights_path: str | None = None
    artifacts_dir: str
```

### Evaluation

```python
class EvaluationReport(BaseModel):
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    per_class_metrics: dict[str, dict] = {}
```

### Decision

```python
class DecisionOutput(BaseModel):
    recommendation: Literal["PROMOTE", "REJECT", "RETRAIN", "NEED_MORE_DATA", "NEED_LABEL_REVIEW"]
    reasons: list[str]
    requires_human_approval: bool = True
```

---

## Dataset Validator details

Реализуй реальный validator для YOLO dataset.

Проверки:

```text
- data.yaml exists;
- images/train and images/val exist;
- labels/train and labels/val exist;
- image files can be opened by Pillow if Pillow installed;
- label txt lines have 5 columns;
- class_id is integer;
- bbox values are floats;
- x_center, y_center, width, height are in 0..1;
- width and height > 0;
- class_id exists in data.yaml names;
- count class distribution;
- detect duplicate file names/hashes if possible.
```

Если Pillow не установлен, сделай fallback: проверять только file extension и labels.

---

## Training Agent details

Сделай 3 режима:

### local_dry_run

Не запускает YOLO. Создает `training_plan.json` и fake outputs:

```json
{
  "mode": "local_dry_run",
  "command": "yolo detect train model=yolo11m.pt data=data.yaml epochs=100 imgsz=640",
  "mlflow_run_id": "dry_run_<id>",
  "best_weights_path": null
}
```

### local_train

Если `ultralytics` установлен, запусти YOLO train. Если нет — верни понятную ошибку.

### azure_train

Через `AzureMLClient` adapter. На первом этапе можно сделать stub, который проверяет config и возвращает fake Azure job id. В коде оставь TODO для реального `azure-ai-ml` submit.

---

## Evaluation Agent details

Для MVP сделай:

- если есть real YOLO artifacts/metrics — парсить;
- если local_dry_run — создавать fake evaluation report из config или deterministic mock;
- сохранять `evaluation_report.json` и `evaluation_report.md`.

---

## Model Decision Agent details

Сделай `promotion_policy.example.yaml`:

```yaml
min_map50: 0.85
min_map50_95: 0.55
max_latency_ms: 100
critical_classes:
  crack:
    min_recall: 0.75
require_human_approval: true
```

Decision logic:

```text
- if required metric missing → NEED_LABEL_REVIEW or RETRAIN with reason
- if map50/map50_95 below threshold → RETRAIN
- if critical class recall below threshold → NEED_MORE_DATA or NEED_LABEL_REVIEW
- if all checks pass → PROMOTE, but still requires human approval
```

Создай `decision_report.md`.

---

## Human Approval MVP

Сделай local JSON approval store:

```text
runs/approvals.json
```

Approval object:

```json
{
  "approval_id": "appr_001",
  "workflow_id": "wf_001",
  "gate": "MODEL_APPROVAL",
  "status": "pending",
  "summary": "Candidate model is ready for review",
  "options": ["approve", "reject", "retrain", "request_more_data"],
  "artifacts": ["evaluation_report.json", "decision_report.md"]
}
```

---

## State machine

Создай simple state machine:

```text
NEW_DATASET
DATASET_VALIDATION_RUNNING
DATASET_VALIDATION_FAILED
DATASET_READY_FOR_TRAINING
TRAINING_RUNNING
TRAINING_FAILED
EVALUATION_RUNNING
EVALUATION_FAILED
MODEL_DECISION_RUNNING
MODEL_APPROVAL_REQUIRED
MODEL_APPROVED
MODEL_REJECTED
COMPLETED
FAILED
```

Illegal transitions должны бросать exception.

---

## Security requirements

- Не хардкодить secrets.
- Не запускать arbitrary shell commands.
- Все команды YOLO строить через allowlisted builder.
- Azure credentials только через env vars.
- Production deployment не реализовывать в MVP.
- Approval required для model promotion.

---

## README requirements

README должен содержать:

1. Что это за проект.
2. Архитектура MVP.
3. Установка.
4. Пример запуска validator.
5. Пример запуска MVP dry-run.
6. Где смотреть artifacts.
7. Как approve/reject.
8. Как добавить Azure ML integration later.

---

## Tests

Создай unit tests:

```text
test_dataset_validator_valid_dataset
test_dataset_validator_bad_bbox
test_dataset_validator_bad_class_id
test_decision_policy_promote
test_decision_policy_retrain_when_map_low
test_decision_policy_requires_human_approval
test_state_machine_blocks_illegal_transition
```

Создай маленькие fixtures YOLO datasets.

---

## Expected result

После выполнения должно работать:

```bash
pip install -e .[dev]
pytest
agentic-mlops validate-dataset --dataset-path tests/fixtures/yolo_dataset_valid
agentic-mlops run-mvp --dataset-path tests/fixtures/yolo_dataset_valid --mode local_dry_run
agentic-mlops approvals list
```

---

## Implementation style

Пиши чистый, простой, typed Python код. Не усложняй. Сначала сделай работающий MVP skeleton с реальным dataset validator, dry-run training, fake evaluation и real decision policy. Затем оставь понятные TODO для Azure ML, MLflow и YOLO real execution.

Используй маленькие классы, dependency injection и адаптеры. Каждый agent должен быть thin orchestration wrapper над tools.
