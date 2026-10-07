> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Label QA Agent

## Назначение

Label QA Agent проверяет качество разметки после manual labeling или pseudo-labeling.

## Что делает

```text
- ищет suspicious bbox
- ищет possible missing objects
- сравнивает human labels и model predictions
- считает class imbalance
- создает label_quality_report.json
- формирует review queue для человека
```

## Inputs

```json
{
  "dataset_path": "runs/wf_001/dataset",
  "labels_path": "runs/wf_001/dataset/labels",
  "reference_model_path": "models/approved/best.pt"
}
```

## Outputs

```json
{
  "label_quality_score": 0.87,
  "suspicious_samples_path": "runs/wf_001/artifacts/suspicious_samples",
  "report_path": "runs/wf_001/artifacts/label_quality_report.json",
  "next_state": "LABEL_REVIEW_REQUIRED"
}
```

## Tools

| Tool | Для чего |
|---|---|
| custom QA scripts | label checks |
| FiftyOne | visual QA |
| IoU comparison | сравнение boxes |
| previous approved YOLO | missing object detection |
| Report writer | QA report |

## Checks

```text
- bbox too small
- bbox too large
- bbox near image boundary
- suspicious aspect ratio
- missing label file
- class distribution anomaly
- disagreement with reference model
```

## Human-in-the-loop

H3: Label Review.

Если suspicious samples > threshold, workflow должен остановиться и отправить labels на review.

## Acceptance Criteria

- Создает suspicious samples list.
- Создает label_quality_report.json.
- Возвращает status: passed / review_required / failed.
- Не изменяет labels сам без отдельного tool/action.
