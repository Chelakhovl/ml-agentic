# Claude Code Prompt — короткий MVP prompt

Создай Python 3.11+ репозиторий `agentic-mlops-yolo` для MVP Agentic MLOps workflow под YOLO object detection.

MVP workflow:

```text
Dataset Validation Agent → Training Agent → Evaluation Agent → Model Decision Agent → Human Approval
```

Требования:

1. Используй `Typer`, `Pydantic v2`, `PyYAML`, `pytest`, `ruff`.
2. Создай CLI:
   ```bash
   agentic-mlops validate-dataset --dataset-path <path>
   agentic-mlops run-mvp --dataset-path <path> --mode local_dry_run
   agentic-mlops approvals list
   agentic-mlops approvals approve <approval_id> --comment "Approved"
   agentic-mlops approvals reject <approval_id> --reason "Reason"
   ```
3. Реализуй реальный YOLO dataset validator:
   - `data.yaml` exists;
   - `images/train`, `images/val`, `labels/train`, `labels/val` exist;
   - label line has 5 columns;
   - `class_id` valid;
   - bbox values in `0..1`;
   - width/height > 0;
   - class distribution report.
4. Training Agent в `local_dry_run` не запускает YOLO, а создает `training_plan.json`.
5. Evaluation Agent в dry-run создает deterministic fake `evaluation_report.json` и `evaluation_report.md`.
6. Model Decision Agent читает `promotion_policy.example.yaml` и возвращает `PROMOTE | REJECT | RETRAIN | NEED_MORE_DATA | NEED_LABEL_REVIEW`.
7. Human approval сохраняется в `runs/approvals.json`.
8. Все artifacts сохраняются в `runs/<workflow_id>/artifacts/`.
9. State сохраняется в `runs/<workflow_id>/state.json`.
10. Напиши unit tests для validator, decision policy и state machine.
11. Не реализуй production deployment. Только stubs/TODO для Azure ML, MLflow и Ultralytics real train.

Создай понятный README с примерами запуска.
