> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Deployment Agent

## Назначение

Deployment Agent готовит модель к serving и выкатывает в staging/production после approval.

## Что делает

```text
- берет registered model
- экспортирует в нужный формат
- собирает inference container
- запускает smoke tests
- deploy to staging
- создает production release approval
- deploy to production после approval
```

## Inputs

```json
{
  "model_registry_uri": "azureml:factory_defects_detector:12",
  "target": "staging",
  "deployment_config_path": "configs/deployment.yaml"
}
```

## Outputs

```json
{
  "endpoint_name": "factory-defects-staging",
  "deployment_status": "deployed_to_staging",
  "deployment_report_path": "deployment_report.json"
}
```

## Tools

| Tool | Для чего |
|---|---|
| Ultralytics export | ONNX/TensorRT/OpenVINO export |
| Docker | container image |
| Azure ML Online Endpoint | serving |
| AKS | optional serving |
| CI/CD | GitHub Actions / Azure DevOps |
| pytest / smoke tests | endpoint check |

## Human-in-the-loop

H6: Production Release Approval.

Человек подтверждает:

```text
- release plan;
- rollback plan;
- production endpoint;
- traffic switch;
- monitoring readiness.
```

## Safety rules

```text
- staging может быть semi-automatic;
- production только после approval;
- rollback plan обязателен;
- smoke tests обязательны.
```

## Acceptance Criteria

- Staging deployment supported.
- Production требует approval.
- Smoke tests выполняются.
- Deployment report создается.
