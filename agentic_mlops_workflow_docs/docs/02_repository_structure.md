> **Living Spec** — This document is kept in sync with the implementation.
> Authoritative sources: [`CLAUDE.md`](../../CLAUDE.md) and [`docs/13_backlog.md`](../docs/13_backlog.md).
> When implementation diverges from this spec, `CLAUDE.md` is the ground truth.

# Repository Structure

Рекомендуемая структура репозитория для реализации.

```text
agentic-mlops-yolo/
  README.md
  pyproject.toml
  .env.example
  .gitignore

  configs/
    workflow.example.yaml
    training.example.yaml
    promotion_policy.example.yaml
    azure_ml.example.yaml

  src/
    agentic_mlops/
      __init__.py

      cli/
        __init__.py
        main.py

      agents/
        __init__.py
        base.py
        orchestrator.py
        data_intake.py
        dataset_structuring.py
        dataset_validation.py
        annotation_pseudolabel.py
        label_qa.py
        dataset_versioning.py
        training.py
        evaluation.py
        model_decision.py
        model_registry.py
        deployment.py
        monitoring.py

      contracts/
        __init__.py
        common.py
        datasets.py
        labels.py
        training.py
        evaluation.py
        decisions.py
        approvals.py
        artifacts.py

      tools/
        __init__.py
        dataset_validator.py
        yolo_trainer.py
        yolo_evaluator.py
        report_writer.py
        approval_store.py
        notification_client.py

      integrations/
        __init__.py
        azure_ml_client.py
        mlflow_client.py
        storage_client.py
        keyvault_client.py
        devops_client.py

      workflows/
        __init__.py
        state_machine.py
        workflow_runner.py
        policies.py

      observability/
        __init__.py
        logging.py
        tracing.py
        metrics.py

      security/
        __init__.py
        permissions.py
        allowlist.py

  pipelines/
    azureml/
      components/
        validate_dataset/
          component.yaml
          validate_dataset.py
        train_yolo/
          component.yaml
          train_yolo.py
        evaluate_model/
          component.yaml
          evaluate_model.py
      pipeline.py

  tests/
    unit/
      test_dataset_validator.py
      test_decision_policy.py
      test_state_machine.py
    integration/
      test_local_workflow.py

  docs/
    architecture.md
    agents.md
    hitl.md
    operations.md
```

## Почему так

### `agents/`

Агенты отвечают за orchestration и decision routing. Они не должны содержать тяжелую ML-логику.

### `tools/`

Tools выполняют конкретную работу:

```text
validate dataset
train YOLO
evaluate model
write report
send notification
```

### `contracts/`

Pydantic-модели для входов/выходов каждого шага. Это нужно, чтобы агенты общались через строгие структуры, а не через произвольный текст.

### `integrations/`

Внешние системы:

```text
Azure ML
MLflow
Azure Blob/ADLS
Key Vault
Azure DevOps/Jira
```

### `workflows/`

State machine, transition rules, workflow runner, policy checks.

### `pipelines/`

Azure ML components и pipeline definitions.

## Принцип реализации

Почти каждый агент должен иметь форму:

```python
class DatasetValidationAgent(BaseAgent):
    def run(self, input: DatasetValidationInput) -> DatasetValidationOutput:
        report = self.tools.dataset_validator.validate(input.dataset_path)
        decision = self.policy.evaluate(report)
        return DatasetValidationOutput(report=report, decision=decision)
```

А не такую:

```python
# плохо
class DatasetValidationAgent:
    def run(self, text: str):
        # LLM сам решает что делать
        ...
```
