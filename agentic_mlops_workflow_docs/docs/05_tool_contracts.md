# Tool Contracts

## Зачем нужны tool contracts

Агенты должны вызывать tools через строгие контракты. Это снижает хаос и делает поведение тестируемым.

Плохо:

```text
Agent → arbitrary shell command
```

Хорошо:

```text
Agent → typed tool method → validated input → deterministic output
```

## Base tool response

```python
class ToolResult(BaseModel):
    success: bool
    message: str
    artifacts: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
```

## Dataset Validator Tool

### Input

```python
class DatasetValidationToolInput(BaseModel):
    dataset_path: str
    data_yaml_path: str | None = None
    expected_format: Literal["yolo"] = "yolo"
    fail_on_warnings: bool = False
```

### Output

```python
class DatasetValidationToolOutput(ToolResult):
    status: Literal["passed", "failed", "warning"]
    num_images: int
    num_labels: int
    class_distribution: dict[str, int]
    blocking_issues: list[str]
    warnings: list[str]
    report_path: str
```

## YOLO Trainer Tool

### Input

```python
class YoloTrainingToolInput(BaseModel):
    dataset_path: str
    data_yaml_path: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    experiment_name: str
    run_name: str
    mode: Literal["local_dry_run", "local_train", "azure_train"]
```

### Output

```python
class YoloTrainingToolOutput(ToolResult):
    mlflow_run_id: str | None
    azure_job_id: str | None
    best_weights_path: str | None
    last_weights_path: str | None
    training_artifacts_dir: str
```

## YOLO Evaluator Tool

### Input

```python
class YoloEvaluationToolInput(BaseModel):
    weights_path: str
    data_yaml_path: str
    split: Literal["val", "test"] = "test"
    baseline_report_path: str | None = None
```

### Output

```python
class YoloEvaluationToolOutput(ToolResult):
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None
    per_class_metrics: dict[str, dict]
    report_json_path: str
    report_md_path: str
```

## Decision Policy Tool

### Input

```python
class DecisionPolicyInput(BaseModel):
    evaluation_report_path: str
    promotion_policy_path: str
    baseline_report_path: str | None = None
```

### Output

```python
class DecisionPolicyOutput(ToolResult):
    recommendation: Literal[
        "PROMOTE",
        "REJECT",
        "RETRAIN",
        "NEED_MORE_DATA",
        "NEED_LABEL_REVIEW"
    ]
    reasons: list[str]
    passed_checks: list[str]
    failed_checks: list[str]
    decision_report_path: str
```

## Notification Tool

### Input

```python
class NotificationInput(BaseModel):
    channel: Literal["console", "teams", "slack", "email"]
    title: str
    message: str
    artifacts: list[str] = []
```

### Output

```python
class NotificationOutput(ToolResult):
    notification_id: str | None
```

## Tool safety

Все tools должны быть:

- allowlisted;
- логируемыми;
- идемпотентными там, где возможно;
- ограниченными по правам;
- тестируемыми через unit tests;
- без прямого доступа к secrets в коде.
