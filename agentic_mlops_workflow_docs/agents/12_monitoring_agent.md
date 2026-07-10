# Monitoring Agent

## Назначение

Monitoring Agent следит за production/staging моделью, собирает drift/hard samples и запускает retraining loop.

## Что мониторит

```text
- latency
- error rate
- confidence distribution
- low-confidence samples
- class distribution drift
- image quality drift
- feedback / corrections
- hard samples
```

## Inputs

```json
{
  "endpoint_name": "factory-defects-prod",
  "model_version": "12",
  "monitoring_window": "24h"
}
```

## Outputs

```json
{
  "monitoring_report_path": "monitoring_report.json",
  "drift_detected": true,
  "hard_samples_manifest": "hard_samples_manifest.json",
  "recommended_action": "CREATE_NEW_DATASET_VERSION"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Azure Monitor | endpoint metrics |
| Application Insights | logs/traces |
| custom drift scripts | drift detection |
| hard sample miner | low-confidence sample collection |
| Azure Blob/ADLS | store samples |
| Notification client | alerts |

## Retraining triggers

```yaml
triggers:
  low_confidence_ratio:
    threshold: 0.25
    action: need_more_data

  p95_latency_ms:
    threshold: 200
    action: notify_ops

  critical_class_drop:
    threshold: 0.15
    action: model_review

  drift_score:
    threshold: 0.30
    action: create_retraining_request
```

## Human-in-the-loop

Человек нужен, если:

```text
- drift критичный;
- много low-confidence cases;
- появились новые classes;
- нужен новый retraining cycle;
- нужно проверить production errors.
```

## Acceptance Criteria

- Создает monitoring report.
- Может создать hard samples manifest.
- Может вернуть workflow в Data Intake.
- Не запускает retraining без approval policy.
