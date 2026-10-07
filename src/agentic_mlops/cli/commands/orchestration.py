"""Orchestration-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import (
    OutputFormat,
    _make_artifact_store,
    _print_orchestrator_result,
    _resolve_mlflow,
    console,
)


def run_workflow(
    workflow_id: str = typer.Option(..., "--workflow-id", help="Unique ID for this workflow run"),
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to an orchestrator config YAML (see configs/orchestrator.example.yaml)",
    ),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Root directory for state.json/audit_log.jsonl/artifacts"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume a previously started workflow_id, skipping completed steps"
    ),
    resume_from_step: str = typer.Option(
        None,
        "--resume-from-step",
        help=(
            "Rewind a saved workflow to the named step and re-run from there. "
            "All prior step outputs are kept; the named step and everything after "
            "it are re-run. Implies --resume. "
            "Valid step names: data_intake, dataset_structuring, dataset_validation, "
            "dataset_versioning, training_approval, training, evaluation, "
            "model_decision, approval, model_registry, deployment."
        ),
    ),
    trigger: str = typer.Option("manual", "--trigger", help="What triggered this run"),
    lock_timeout_seconds: float = typer.Option(
        30.0,
        "--lock-timeout-seconds",
        help=(
            "Seconds to wait when acquiring the workflow run lock. "
            "0 = fail immediately if another process holds the lock."
        ),
    ),
    state_backend: str = typer.Option(
        "local",
        "--state-backend",
        help=(
            "Where to persist state.json and audit_log.jsonl: "
            "'local' (default, filesystem) or 'azure_blob' (Azure Blob Storage; "
            "requires --azure-config in orchestrator.yaml and "
            "state_blob_container set there or via AZURE_STORAGE_CONNECTION_STRING)."
        ),
    ),
    mlflow_config: str = typer.Option(None, "--mlflow-config", help="Path to MLflow config YAML"),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--disable-mlflow",
        help="Enable MLflow tracking as one parent run across all steps (default: disabled)",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", "-f", help="Output format: text or json"
    ),
) -> None:
    """Run the full, configurable Orchestrator pipeline (any subset of PIPELINE_STEPS).

    Unlike `run-mvp` (fixed 5-step chain), this reads step selection and every
    step's config from a single YAML file, persists state.json/audit_log.jsonl
    under --runs-dir/<workflow-id>/, and supports --resume after a pause (e.g.
    at the human approval gate) or a failure. --resume-from-step <name> rewinds
    a saved workflow to re-run from a specific step without replaying earlier ones
    (prior outputs are preserved so downstream steps still have their context).
    With --enable-mlflow, the whole run is tracked as one parent MLflow run.
    """
    from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

    try:
        inp = OrchestratorInput.from_yaml(
            config,
            workflow_id=workflow_id,
            runs_dir=runs_dir,
            resume=resume,
            resume_from_step=resume_from_step or None,
            trigger=trigger,
            lock_timeout_seconds=lock_timeout_seconds,
            state_backend=state_backend,
        )
    except Exception as exc:
        console.print(f"[red]Cannot load orchestrator config '{config}': {exc}[/red]")
        raise typer.Exit(code=1)

    mlflow_client, mlflow_cfg = _resolve_mlflow(mlflow_config, enable_mlflow)
    artifact_store = _make_artifact_store(inp.azure_config_path)

    result = OrchestratorWorkflow(
        mlflow_client=mlflow_client if mlflow_cfg.enabled else None,
        mlflow_config=mlflow_cfg if mlflow_cfg.enabled else None,
        artifact_store=artifact_store,
    ).run(inp)

    if format == OutputFormat.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_orchestrator_result(result)

    if not result.success:
        raise typer.Exit(code=1)


def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Directory containing workflow run state files"
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry", "--registry-dir", help="Local model registry root directory"
    ),
    datasets_dir: str = typer.Option(
        "outputs/dataset_registry",
        "--dataset-registry-dir",
        help="Local dataset registry root directory",
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development mode)"),
    password: str = typer.Option(
        "",
        "--password",
        envvar="DASHBOARD_PASSWORD",
        help="HTTP Basic Auth password. Omit (or leave DASHBOARD_PASSWORD unset) to disable auth.",
    ),
    user: str = typer.Option(
        "admin",
        "--user",
        envvar="DASHBOARD_USER",
        help="HTTP Basic Auth username (default: admin).",
    ),
    slack_signing_secret: str = typer.Option(
        "",
        "--slack-signing-secret",
        envvar="SLACK_SIGNING_SECRET",
        help=(
            "Slack signing secret for validating interactive button callbacks "
            "at POST /webhooks/approve/slack. Get it from your Slack app's "
            "'Basic Information' page. Also readable from the "
            "SLACK_SIGNING_SECRET environment variable."
        ),
    ),
    teams_signing_secret: str = typer.Option(
        "",
        "--teams-signing-secret",
        envvar="TEAMS_SIGNING_SECRET",
        help=(
            "Teams connector security token (base64-encoded symmetric key) for "
            "validating interactive HttpPOST button callbacks at "
            "POST /webhooks/approve/teams. Get it from your Teams connector "
            "configuration page. Also readable from the TEAMS_SIGNING_SECRET "
            "environment variable."
        ),
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        envvar="DASHBOARD_API_KEY",
        help=(
            "Bearer / X-API-Key token required on every request. "
            "Supply via Authorization: Bearer <token> or X-API-Key: <token>. "
            "Also readable from the DASHBOARD_API_KEY environment variable. "
            "Disabled (no token required) when empty."
        ),
    ),
    oidc_issuer: str = typer.Option(
        "",
        "--oidc-issuer",
        envvar="DASHBOARD_OIDC_ISSUER",
        help=(
            "OIDC issuer URL for JWT validation (e.g. "
            "https://login.microsoftonline.com/<tenant>/v2.0). "
            "Requires the 'auth' extra (pip install -e '.[auth]') and "
            "--oidc-client-id. Also readable from DASHBOARD_OIDC_ISSUER."
        ),
    ),
    oidc_client_id: str = typer.Option(
        "",
        "--oidc-client-id",
        envvar="DASHBOARD_OIDC_CLIENT_ID",
        help=(
            "Expected 'aud' claim in OIDC JWTs. "
            "Required when --oidc-issuer is set. "
            "Also readable from DASHBOARD_OIDC_CLIENT_ID."
        ),
    ),
    require_auth: bool = typer.Option(
        False,
        "--require-auth/--no-require-auth",
        help=(
            "Fail fast on startup if no auth method is configured. "
            "Use in production to prevent accidentally running unauthenticated."
        ),
    ),
) -> None:
    """Start the MLOps web dashboard (requires the 'web' extra: pip install -e '.[web]')."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]The 'web' extra is required. Install with:[/red]\n" "  pip install -e '.[web]'"
        )
        raise typer.Exit(code=1) from None

    from agentic_mlops.web.app import create_app

    try:
        web_app = create_app(
            runs_dir=runs_dir,
            registry_dir=registry_dir,
            datasets_dir=datasets_dir,
            dashboard_user=user,
            dashboard_password=password,
            slack_signing_secret=slack_signing_secret,
            teams_signing_secret=teams_signing_secret,
            api_key=api_key,
            oidc_issuer=oidc_issuer,
            oidc_client_id=oidc_client_id,
            require_auth=require_auth,
        )
    except (RuntimeError, ImportError) as exc:
        console.print(f"[red]Dashboard startup failed: {exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"[bold green]MLOps Dashboard[/bold green]  http://{host}:{port}")
    console.print(f"  Runs dir     : [bold]{runs_dir}[/bold]")
    console.print(f"  Registry dir : [bold]{registry_dir}[/bold]")
    console.print(f"  Datasets dir : [bold]{datasets_dir}[/bold]")
    _auth_methods = []
    if password:
        _auth_methods.append(f"Basic Auth (user: {user})")
    if api_key:
        _auth_methods.append("API key")
    if oidc_issuer and oidc_client_id:
        _auth_methods.append(f"OIDC ({oidc_issuer})")
    if _auth_methods:
        console.print(f"  Auth         : {', '.join(_auth_methods)}")
    else:
        console.print(
            "  Auth         : [yellow]disabled[/yellow]"
            " — set DASHBOARD_PASSWORD or DASHBOARD_API_KEY to enable"
        )
    uvicorn.run(web_app, host=host, port=port, reload=reload)


def watch_workflow(
    workflow_id: str = typer.Argument(..., help="Workflow ID to watch"),
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Root directory for workflow run state files",
    ),
    interval: float = typer.Option(
        1.0,
        "--interval",
        "-i",
        help="Poll interval in seconds",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Keep watching even after the workflow reaches a terminal state",
    ),
) -> None:
    """Tail a workflow's audit log live.

    Prints all existing audit events, then polls for new ones every INTERVAL
    seconds.  Stops automatically when the workflow reaches a terminal state
    (completed / failed / blocked) unless --follow is set.

    Exit codes: 0 = workflow completed; 1 = workflow failed/blocked; 2 = not found.
    """
    import json as _json
    import time as _time

    _TERMINAL = {"completed", "failed", "blocked"}
    _ICONS = {
        "workflow_started": "▶",
        "workflow_resumed": "↩",
        "workflow_finished": "■",
        "workflow_already_completed": "✓",
        "step_started": "→",
        "step_finished": "✓",
        "step_exception": "✕",
        "workflow_failed_precheck": "✕",
    }

    state_path = runs_dir / workflow_id / "state.json"
    audit_path = runs_dir / workflow_id / "audit_log.jsonl"

    if not state_path.exists():
        console.print(f"[red]Workflow '{workflow_id}' not found in {runs_dir}[/red]")
        raise typer.Exit(code=2)

    def _fmt(entry: dict) -> str:
        ev = entry.get("event", "")
        icon = _ICONS.get(ev, "·")
        ts_raw = entry.get("timestamp", "")
        try:
            ts = ts_raw[:19].replace("T", " ")
        except Exception:
            ts = ts_raw
        parts = [f"[dim]{ts}[/dim]  {icon} [bold]{ev}[/bold]"]
        if entry.get("step"):
            parts.append(f"  step=[cyan]{entry['step']}[/cyan]")
        if entry.get("status"):
            parts.append(f"  status={entry['status']}")
        if entry.get("error"):
            parts.append(f"  [red]error: {entry['error']}[/red]")
        return "".join(parts)

    seen = 0
    console.print(f"\n[bold]Watching[/bold] [cyan]{workflow_id}[/cyan]  (Ctrl-C to stop)\n")

    try:
        while True:
            if audit_path.exists():
                with open(audit_path, encoding="utf-8") as fh:
                    lines = fh.readlines()
                for raw in lines[seen:]:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = _json.loads(raw)
                    except Exception:
                        continue
                    console.print(_fmt(entry))
                seen = len(lines)

            try:
                with open(state_path, encoding="utf-8") as fh:
                    state = _json.load(fh)
                status = state.get("status", "unknown")
            except Exception:
                status = "unknown"

            if status in _TERMINAL and not follow:
                console.print(f"\n[bold]Workflow {status}.[/bold]")
                raise typer.Exit(code=0 if status == "completed" else 1)

            _time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
