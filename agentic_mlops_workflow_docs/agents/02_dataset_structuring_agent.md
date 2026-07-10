# Dataset Structuring Agent

## Назначение

Dataset Structuring Agent приводит raw/curated данные к YOLO-совместимой структуре.

## Что делает

```text
- создает folders images/train, images/val, images/test
- создает folders labels/train, labels/val, labels/test
- делает split train/val/test
- генерирует data.yaml
- конвертирует labels, если исходный формат COCO/VOC
- сохраняет split_report.json
```

## Inputs

```json
{
  "dataset_manifest_path": "dataset_manifest.json",
  "raw_data_uri": "azureml://.../raw/v1",
  "label_format": "yolo",
  "split_strategy": "grouped_by_source",
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "classes": ["scratch", "dent", "crack"]
}
```

## Outputs

```json
{
  "structured_dataset_path": "runs/wf_001/dataset",
  "data_yaml_path": "runs/wf_001/dataset/data.yaml",
  "split_report_path": "runs/wf_001/artifacts/split_report.json"
}
```

## Tools

| Tool | Для чего |
|---|---|
| pathlib / shutil | file operations |
| scikit-learn train_test_split | split |
| custom grouped splitter | не допускать video/frame leakage |
| COCO→YOLO converter | format conversion |
| YAML generator | data.yaml |
| Azure ML Data Assets | later versioning |

## Важное правило

Если данные пришли из видео, нельзя случайно разделять соседние frames между train и val/test.

Нужно учитывать:

```text
source_video_id
camera_id
time_window
scene_id
```

## Human-in-the-loop

Человек нужен, если:

```text
- меняется список классов;
- меняется test set;
- split strategy спорная;
- появились новые classes.
```

## Acceptance Criteria

- Структура YOLO создана.
- `data.yaml` валиден.
- Split report создан.
- Нет пересечения файлов между train/val/test.
