# Evaluation Agent

## Назначение

Evaluation Agent оценивает candidate model, сравнивает с baseline и создает evaluation report.

## Что делает

```text
- запускает YOLO val на val/test split
- собирает mAP50, mAP50-95, precision, recall
- строит confusion matrix / PR curve если доступно
- считает per-class metrics
- сравнивает с baseline
- формирует human-readable report
```

## Inputs

```json
{
  "weights_path": "runs/wf_001/artifacts/best.pt",
  "data_yaml_path": "data.yaml",
  "split": "test",
  "baseline_run_id": "run_9001"
}
```

## Outputs

```json
{
  "evaluation_report_json": "evaluation_report.json",
  "evaluation_report_md": "evaluation_report.md",
  "map50": 0.862,
  "map50_95": 0.591,
  "precision": 0.84,
  "recall": 0.79
}
```

## Tools

| Tool | Для чего |
|---|---|
| Ultralytics YOLO val | validation/evaluation |
| MLflow client | получить candidate/baseline metrics |
| pandas | metric tables |
| matplotlib | charts |
| Report writer | Markdown/JSON report |
| error analysis scripts | FP/FN samples |

## Evaluation report должен отвечать

```text
- Модель лучше baseline или хуже?
- Какие classes просели?
- Есть ли overfitting?
- Какие false positives/false negatives критичны?
- Можно ли модель отправлять на approval?
```

## Human-in-the-loop

H5: Model Approval.

Evaluation Agent сам не принимает финальное решение, но передает evidence в Model Decision Agent и человеку.

## Acceptance Criteria

- Evaluation report создается в JSON и Markdown.
- Метрики machine-readable.
- Baseline comparison поддерживается.
- Per-class metrics поддерживаются.
- Missing metrics не приводят к silent pass.
