> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Data Intake Agent

## Назначение

Data Intake Agent принимает новые изображения, проверяет базовое качество файлов и создает `dataset_manifest.json`.

## Когда вызывается

```text
- появился новый raw dataset
- monitoring нашел hard samples
- Model Decision Agent запросил more data
```

## Что делает

```text
- проверяет путь к данным
- считает количество файлов
- проверяет форматы изображений
- ищет битые изображения
- считает hashes
- ищет дубликаты
- собирает metadata
- сохраняет manifest
```

## Inputs

```json
{
  "raw_data_uri": "azureml://datastores/workspaceblobstore/paths/raw/factory/v1",
  "dataset_name": "factory_defects_raw",
  "expected_formats": ["jpg", "jpeg", "png"],
  "source": "internal_camera_batch"
}
```

## Outputs

```json
{
  "manifest_path": "runs/wf_001/artifacts/dataset_manifest.json",
  "num_files": 12000,
  "valid_images": 11950,
  "corrupted_images": 50,
  "duplicates": 120,
  "status": "needs_human_source_approval"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Azure Blob / ADLS client | читать raw images |
| Pillow / OpenCV | открыть и проверить image |
| hashlib / imagehash | hashes и duplicates |
| EXIF / metadata extractor | source metadata |
| Report writer | создать manifest |

## Human-in-the-loop

H1: Human Data Owner approval.

Человек подтверждает:

```text
- можно ли использовать source;
- нет ли license/privacy проблем;
- правильная ли taxonomy classes;
- можно ли продолжать pipeline.
```

## Failure cases

```text
- storage path не найден
- слишком много corrupted images
- source не указан
- dataset слишком маленький
- duplicate ratio выше threshold
```

## Acceptance Criteria

- Создается `dataset_manifest.json`.
- Битые изображения перечислены отдельно.
- Дубликаты перечислены отдельно.
- Source metadata сохранена.
- Если source неизвестен, workflow требует human approval.
