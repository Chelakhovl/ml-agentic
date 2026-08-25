"""CLI entry-point for the Agentic MLOps workflow."""

from __future__ import annotations

import typer

from agentic_mlops.observability.logging import configure_logging

from .commands import data, deployment, orchestration, registry, training, utils

app = typer.Typer(
    name="agentic-mlops",
    help="Agentic MLOps workflow for YOLO object detection.",
    no_args_is_help=True,
)


@app.callback()
def _setup(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-formatted log lines"),
) -> None:
    configure_logging(level=log_level, json_format=json_logs)


# Data commands
app.command("validate-dataset")(data.validate_dataset)
app.command("data-intake")(data.data_intake)
app.command("structure-dataset")(data.structure_dataset)
app.command("pseudo-label")(data.pseudo_label)
app.command("label-qa")(data.label_qa)
app.command("version-dataset")(data.version_dataset)

# Training commands
app.command("train")(training.train)
app.command("evaluate")(training.evaluate)
app.command("approve")(training.approve)
app.command("approve-training")(training.approve_training)
app.command("register-model")(training.register_model)
app.command("run-mvp")(training.run_mvp)

# Deployment commands
app.command("deploy-model")(deployment.deploy_model)
app.command("monitor")(deployment.monitor)
app.command("ingest-hard-samples")(deployment.ingest_hard_samples)

# Orchestration commands
app.command("run-workflow")(orchestration.run_workflow)
app.command("serve")(orchestration.serve)
app.command("watch")(orchestration.watch_workflow)

# Registry commands
app.command("compare-models")(registry.compare_models)

# Utility commands
app.command("model-decision")(utils.model_decision)
app.command("doctor")(utils.doctor)
app.command("status")(utils.status)
app.command("show-state")(utils.show_state)
app.command("diff-runs")(utils.diff_runs)
app.command("prune-runs")(utils.prune_runs)
app.command("tag-run")(utils.tag_run)
app.command("lint-config")(utils.lint_config)
app.command("cost-report")(utils.cost_report)

if __name__ == "__main__":
    app()
