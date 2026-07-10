# Human-in-the-loop Design

## Принцип

Human-in-the-loop нужен не на каждом шаге, а только там, где есть риск, стоимость или бизнес-решение.

```text
Automate routine checks.
Ask human for decisions.
Log every approval.
```

## HITL gates

| Gate | Где | Кто | Что решает |
|---|---|---|---|
| H1 | Data Intake | Data Owner / PM | Можно ли использовать data source, license, taxonomy |
| H2 | Dataset Validation | ML Engineer / Data Engineer | Чистить dataset или продолжать |
| H3 | Label Review | Labeler / Domain Expert | Проверка uncertain labels и bbox |
| H4 | Training Approval | ML Engineer | Approve compute budget и config |
| H5 | Model Approval | ML Lead / Product Owner | Approve / reject / retrain / request data |
| H6 | Production Release Approval | Tech Lead / Owner | Approve release и rollback plan |

## Approval object

```json
{
  "approval_id": "appr_001",
  "workflow_id": "wf_001",
  "gate": "MODEL_APPROVAL",
  "requested_by": "model_decision_agent",
  "status": "pending",
  "summary": "Candidate model improved mAP50 but crack recall is close to threshold.",
  "options": ["approve", "reject", "retrain", "request_more_data"],
  "artifacts": [
    "evaluation_report.json",
    "decision_report.md",
    "confusion_matrix.png"
  ]
}
```

## Approval commands для MVP

```bash
agentic-mlops approvals list
agentic-mlops approvals show appr_001
agentic-mlops approvals approve appr_001 --comment "Approved for staging only"
agentic-mlops approvals reject appr_001 --reason "Recall for crack is too low"
```

## Approval UI later

После MVP можно сделать:

- FastAPI endpoint для approvals;
- simple web UI;
- Teams/Slack interactive cards;
- Azure DevOps manual approval integration;
- Jira ticket workflow.

## Что агент должен показывать человеку

Для каждого approval должны быть:

```text
- краткий summary
- recommendation
- риски
- метрики
- ссылка на artifacts
- варианты решения
- последствия каждого решения
```

## Что нельзя делать

Нельзя:

- auto-approve production deployment;
- скрывать failed checks;
- перезаписывать approval history;
- принимать approval без user identity;
- принимать approval только на основании LLM summary без raw artifacts.
