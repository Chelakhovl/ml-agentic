> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Data Contracts

## YOLO dataset structure

Ожидаемая структура:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

## data.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: scratch
  1: dent
  2: crack
  3: missing_part
```

## YOLO label format

Каждая строка label-файла:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Все координаты должны быть нормализованы в диапазоне `0..1`.

Пример:

```text
0 0.512 0.433 0.120 0.088
2 0.234 0.521 0.050 0.040
```

## Dataset validation rules

### Blocking issues

Должны блокировать training:

- отсутствует `data.yaml`;
- нет `images/train` или `images/val`;
- нет labels directory;
- label-файл имеет неправильное количество колонок;
- `class_id` не существует в `names`;
- bbox координаты не в диапазоне `0..1`;
- bbox width/height <= 0;
- image файл битый;
- дубликаты между train и val/test;
- test set изменился без approval.

### Warnings

Не всегда блокируют training:

- сильный class imbalance;
- мало примеров critical class;
- слишком маленькие objects;
- слишком большие objects;
- много empty label files;
- unusual image resolution distribution.

## Split rules

Рекомендуемые пропорции:

```text
train: 70-80%
val: 10-20%
test: 10-15%
```

Важно:

- test set должен быть максимально стабильным;
- нельзя допускать дубликаты между splits;
- если есть видео-кадры, split должен учитывать source/video id, чтобы frames из одного видео не попадали в train и val одновременно.

## Dataset lineage

Каждая версия датасета должна знать:

```json
{
  "dataset_version": "v8",
  "parent_dataset_version": "v7",
  "source_batches": ["camera_batch_2026_06_01"],
  "labeling_project": "labeling_001",
  "created_by": "dataset_versioning_agent",
  "approved_by": "ml_engineer",
  "hash": "sha256:..."
}
```
