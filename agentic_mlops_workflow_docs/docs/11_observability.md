> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Observability

## Что нужно наблюдать

Observability нужна для двух вещей:

1. понимать, что происходит с workflow;
2. понимать, почему модель стала лучше или хуже.

## Structured logs

Каждый агент должен писать structured JSON logs.

```json
{
  "timestamp": "2026-06-12T10:00:00Z",
  "level": "INFO",
  "workflow_id": "wf_001",
  "agent": "dataset_validation_agent",
  "event": "validation_completed",
  "status": "failed",
  "blocking_issues_count": 3,
  "artifacts": ["dataset_quality_report.json"]
}
```

## Metrics для workflow

```text
workflow_duration_seconds
agent_duration_seconds
training_job_duration_seconds
approval_wait_time_seconds
num_failed_validations
num_retraining_cycles
num_human_reviews
```

## Metrics для ML

```text
map50
map50_95
precision
recall
critical_class_recall
false_positive_count
false_negative_count
latency_ms
model_size_mb
```

## Monitoring Agent signals

Monitoring Agent должен собирать:

```text
- endpoint latency
- error rate
- prediction confidence distribution
- class distribution drift
- low-confidence samples
- user feedback
- hard samples
```

## Alerts

Примеры alert rules:

```yaml
alerts:
  low_confidence_spike:
    condition: "low_confidence_ratio > 0.25"
    action: "create_hard_sample_dataset"

  latency_degradation:
    condition: "p95_latency_ms > 200"
    action: "notify_mlops_team"

  drift_detected:
    condition: "class_distribution_drift > threshold"
    action: "request_retraining_review"
```

## Traceability

Для каждой production model должно быть понятно:

```text
Какие данные использовали?
Какая версия labels?
Какой git commit?
Какие параметры обучения?
Какие метрики?
Кто approved?
Когда deployed?
```
