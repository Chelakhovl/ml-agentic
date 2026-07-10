# Dataset Validation Agent

## Назначение

Dataset Validation Agent проверяет, готов ли YOLO-датасет к обучению.

## Что делает

```text
- проверяет folders и data.yaml
- проверяет image-label matching
- валидирует YOLO label format
- проверяет class_id
- проверяет bbox range
- ищет дубликаты между splits
- считает class distribution
- создает quality report
- решает: можно training или нужен cleanup
```

## Inputs

```json
{
  "dataset_path": "runs/wf_001/dataset",
  "data_yaml_path": "runs/wf_001/dataset/data.yaml",
  "validation_policy_path": "configs/dataset_validation_policy.yaml"
}
```

## Outputs

```json
{
  "status": "failed",
  "report_path": "runs/wf_001/artifacts/dataset_quality_report.json",
  "blocking_issues": 3,
  "warnings": 5,
  "next_state": "DATASET_NEEDS_CLEANUP"
}
```

## Tools

| Tool | Для чего |
|---|---|
| custom Python validator | основные проверки |
| OpenCV / Pillow | проверка изображений |
| pandas | статистика |
| FiftyOne | visual QA опционально |
| Report writer | JSON/Markdown report |

## Blocking checks

```text
- data.yaml missing
- label format invalid
- class_id unknown
- bbox outside 0..1
- bbox width/height <= 0
- duplicated images across splits
- corrupted images
```

## Warnings

```text
- class imbalance
- too few examples for critical classes
- many empty labels
- unusual image sizes
- very small objects
```

## Human-in-the-loop

H2: Dataset Review / Cleanup.

Человек решает:

```text
- исправлять labels;
- удалить плохие samples;
- принять warnings;
- вернуть на structuring;
- продолжать training.
```

## Acceptance Criteria

- Работает на локальном YOLO dataset.
- Создает JSON и Markdown report.
- Возвращает machine-readable status.
- Unit tests покрывают bad class_id, bad bbox, missing labels, duplicates.
