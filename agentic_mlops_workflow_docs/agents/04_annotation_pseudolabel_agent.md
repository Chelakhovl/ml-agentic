# Annotation / Pseudo-label Agent

## Назначение

Annotation / Pseudo-label Agent ускоряет разметку, используя approved/pre-trained YOLO model для предварительной разметки.

## Что делает

```text
- берет unlabeled или partially labeled images
- запускает YOLO predict
- сохраняет candidate labels
- маршрутизирует predictions по confidence
- отправляет uncertain cases в human review
- создает pseudo_label_report.json
```

## Inputs

```json
{
  "images_path": "runs/wf_001/dataset/images/unlabeled",
  "model_path": "models/approved/best.pt",
  "confidence_thresholds": {
    "auto_candidate": 0.90,
    "human_review": 0.50
  }
}
```

## Outputs

```json
{
  "pseudo_labels_path": "runs/wf_001/artifacts/pseudo_labels",
  "review_queue_path": "runs/wf_001/artifacts/review_queue.json",
  "high_confidence_count": 3200,
  "medium_confidence_count": 800,
  "low_confidence_count": 240
}
```

## Tools

| Tool | Для чего |
|---|---|
| Ultralytics YOLO predict | pre-labeling |
| Approved model registry | выбрать модель для pseudo-labeling |
| CVAT / Label Studio / Azure ML Data Labeling | human review UI |
| confidence router | маршрутизация predictions |
| Report writer | pseudo_label_report.json |

## Confidence routing

```text
confidence >= 0.90
  → candidate label
  → sample audit by human

0.50 <= confidence < 0.90
  → human review

confidence < 0.50
  → hard sample / expert review
```

## Human-in-the-loop

H3: Label Review.

Человек:

```text
- исправляет bbox;
- добавляет missing objects;
- исправляет class;
- подтверждает uncertain labels;
- проверяет sample high-confidence labels.
```

## Safety rule

Pseudo-labels не должны автоматически становиться final labels без политики review/audit.

## Acceptance Criteria

- Создает labels в YOLO формате.
- Создает review queue.
- Сохраняет confidence metadata.
- Не перезаписывает human labels без approval.
