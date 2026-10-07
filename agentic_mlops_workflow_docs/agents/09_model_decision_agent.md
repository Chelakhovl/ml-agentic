> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Model Decision Agent

## Назначение

Model Decision Agent сравнивает evaluation report с promotion policy и готовит recommendation.

## Возможные рекомендации

```text
PROMOTE
REJECT
RETRAIN
NEED_MORE_DATA
NEED_LABEL_REVIEW
```

## Что делает

```text
- читает evaluation_report.json
- читает promotion_policy.yaml
- сравнивает candidate с baseline
- проверяет critical class thresholds
- проверяет latency/model size
- создает decision_report.md
- создает human approval request
```

## Inputs

```json
{
  "evaluation_report_path": "evaluation_report.json",
  "promotion_policy_path": "configs/promotion_policy.yaml",
  "baseline_report_path": "baseline_report.json"
}
```

## Outputs

```json
{
  "recommendation": "RETRAIN",
  "reasons": [
    "mAP50 improved",
    "critical class crack recall below threshold"
  ],
  "decision_report_path": "decision_report.md",
  "required_approval_gate": "MODEL_APPROVAL"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Policy engine | threshold checks |
| MLflow comparison | compare runs |
| Report writer | decision report |
| Approval store | create approval request |
| Notification client | notify reviewers |

## Promotion policy example

```yaml
min_map50: 0.85
min_map50_95: 0.55
max_latency_ms: 40
critical_classes:
  crack:
    min_recall: 0.75
  missing_part:
    min_recall: 0.80
require_human_approval: true
```

## Decision routing

```text
PROMOTE → Human Model Approval
RETRAIN → Training Approval / Training Agent
NEED_MORE_DATA → Data Intake Agent
NEED_LABEL_REVIEW → Label QA / Human Label Review
REJECT → End or back to planning
```

## Human-in-the-loop

H5 обязателен для promotion.

Даже если policy passed, агент должен создать approval request, а не автоматически продвигать модель в production.

## Acceptance Criteria

- Все decisions объяснимы.
- Reasons сохраняются.
- Нет auto-production-promote.
- Policy покрыта unit tests.
