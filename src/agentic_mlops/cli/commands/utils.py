"""Utility CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import _print_model_decision_result, console


def model_decision(
    evaluation_report: str = typer.Argument(..., help="Path to evaluation_report.json"),
    promotion_policy: str = typer.Option(
        None,
        "--promotion-policy",
        help="Path to promotion_policy.yaml (for baseline-improvement requirement)",
    ),
    evaluation_config: str = typer.Option(
        None,
        "--evaluation-config",
        help="Path to evaluation config YAML (for max_latency_ms / max_model_size_mb)",
    ),
    baseline_report: str = typer.Option(
        None,
        "--baseline-report",
        help="Path to a baseline evaluation_report.json to compare against",
    ),
    auto_baseline: bool = typer.Option(
        False,
        "--auto-baseline/--no-auto-baseline",
        help=(
            "Auto-resolve the best registered model as baseline from --registry-dir. "
            "Ignored when --baseline-report is also set."
        ),
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry",
        "--registry-dir",
        help="Local model registry root (used when --auto-baseline is set)",
    ),
    measured_latency_ms: float = typer.Option(
        None, "--measured-latency-ms", help="Externally-measured inference latency, in ms"
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Where to save the decision report (default: <evaluation_report dir>/decision_out)",
    ),
) -> None:
    """Turn an evaluation report into an explainable promote/reject/retrain decision."""
    from agentic_mlops.agents.model_decision import ModelDecisionAgent  # noqa: PLC0415
    from agentic_mlops.contracts.model_decision import ModelDecisionInput  # noqa: PLC0415

    artifacts_dir = (
        Path(output_dir) if output_dir else Path(evaluation_report).parent / "decision_out"
    )

    # Auto-resolve baseline from registry when requested and no explicit path given
    resolved_baseline = baseline_report
    if not resolved_baseline and auto_baseline:
        from agentic_mlops.tools.baseline_resolver import BaselineResolver  # noqa: PLC0415

        resolver = BaselineResolver()
        resolved = resolver.resolve(Path(registry_dir))
        if resolved is not None:
            synthetic = resolver.write_synthetic_report(resolved, output_dir=artifacts_dir)
            resolved_baseline = str(synthetic)
            console.print(
                f"[dim]Auto-baseline: {resolved.model_name} v{resolved.version} "
                f"(mAP50={resolved.map50:.4f})[/dim]"
            )
        else:
            console.print("[yellow]--auto-baseline: no registered models found; skipping.[/yellow]")

    agent = ModelDecisionAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        ModelDecisionInput(
            evaluation_report_path=evaluation_report,
            promotion_policy_path=promotion_policy,
            evaluation_config_path=evaluation_config,
            baseline_report_path=resolved_baseline,
            measured_latency_ms=measured_latency_ms,
        )
    )

    _print_model_decision_result(result)

    if not result.success:
        raise typer.Exit(code=1)


def doctor(
    config: str = typer.Option(
        None,
        "--config",
        help="Orchestrator config YAML to validate (can be passed multiple times)",
        show_default=False,
    ),
    azure_config: str = typer.Option(
        None,
        "--azure-config",
        help="Azure ML config YAML to validate",
    ),
    mlflow_config: str = typer.Option(
        None,
        "--mlflow-config",
        help="MLflow config YAML to validate",
    ),
    runs_dir: str = typer.Option(
        "runs",
        "--runs-dir",
        help="Workflow runs directory to check for write access",
    ),
    registry_dir: str = typer.Option(
        "outputs/model_registry",
        "--registry-dir",
        help="Model registry directory to check for write access",
    ),
    dataset_registry_dir: str = typer.Option(
        "outputs/dataset_registry",
        "--dataset-registry-dir",
        help="Dataset registry directory to check for write access",
    ),
    no_tools: bool = typer.Option(
        False,
        "--no-tools",
        help="Skip external CLI tool checks (docker, az)",
    ),
) -> None:
    """Run a local self-check — verify dependencies, configs, and directory access.

    Checks required and optional Python packages, validates any YAML config files
    you pass, confirms output directories are writable, and probes for optional
    external tools (docker, az CLI).  No network calls are made.
    """
    from agentic_mlops.contracts.doctor import CheckStatus  # noqa: PLC0415
    from agentic_mlops.tools.doctor import SystemDoctor  # noqa: PLC0415

    config_paths: list[tuple[Path, str]] = []
    if config:
        config_paths.append((Path(config), "orchestrator config"))
    if azure_config:
        config_paths.append((Path(azure_config), "azure_ml config"))
    if mlflow_config:
        config_paths.append((Path(mlflow_config), "mlflow config"))

    directory_paths: list[tuple[Path, str]] = [
        (Path(runs_dir), "runs_dir"),
        (Path(registry_dir), "registry_dir"),
        (Path(dataset_registry_dir), "dataset_registry_dir"),
    ]

    report = SystemDoctor().check(
        config_paths=config_paths,
        directory_paths=directory_paths,
        check_docker=not no_tools,
        check_az=not no_tools,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    _STATUS_ICON = {
        CheckStatus.OK: "[green]✓[/green]",
        CheckStatus.WARNING: "[yellow]⚠[/yellow]",
        CheckStatus.ERROR: "[red]✗[/red]",
    }
    _CATEGORY_HEADER = {
        "packages": "Python Packages",
        "configs": "Config Files",
        "directories": "Directories",
        "tools": "External Tools",
    }

    console.print("\n[bold]agentic-mlops doctor[/bold]\n")

    current_cat = ""
    for chk in report.checks:
        if chk.category != current_cat:
            current_cat = chk.category
            console.print(f"  [bold]{_CATEGORY_HEADER.get(current_cat, current_cat)}[/bold]")
        icon = _STATUS_ICON[chk.status]
        console.print(f"    {icon}  {chk.message}")

    console.print()
    ok_s = f"[green]{report.num_ok} ok[/green]"
    warn_s = f"[yellow]{report.num_warnings} warnings[/yellow]"
    err_s = f"[red]{report.num_errors} errors[/red]"
    console.print(f"  Summary: {ok_s}  {warn_s}  {err_s}")

    if report.overall_status == CheckStatus.OK:
        console.print("  [green bold]All checks passed.[/green bold]\n")
    elif report.overall_status == CheckStatus.WARNING:
        console.print(
            "  [yellow]Some optional components are missing — see warnings above.[/yellow]\n"
        )
        raise typer.Exit(code=0)
    else:
        console.print("  [red bold]One or more required checks failed.[/red bold]\n")
        raise typer.Exit(code=1)


def status(
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    registry_dir: str = typer.Option(
        "outputs/model_registry", "--registry-dir", help="Model registry root"
    ),
    dataset_registry_dir: str = typer.Option(
        "outputs/dataset_registry", "--dataset-registry-dir", help="Dataset registry root"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Max recent workflows to show"),
) -> None:
    """Show a quick health snapshot: workflows, models, datasets, monitoring alerts."""
    from agentic_mlops.web import reader  # noqa: PLC0415

    _rd = Path(runs_dir)
    _mod = Path(registry_dir)
    _ds = Path(dataset_registry_dir)

    _WF_ICON = {
        "completed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "running": "[cyan]⟳[/cyan]",
        "pending_approval": "[yellow]⏳[/yellow]",
        "blocked": "[red]⊘[/red]",
    }
    _ACTION_COLOR = {
        "no_action": "green",
        "notify_ops": "yellow",
        "need_more_data": "yellow",
        "model_review": "red",
        "create_retraining_request": "red",
    }

    def _short_dt(ts: str | None) -> str:
        if not ts:
            return "—"
        return ts[:16].replace("T", " ")

    console.print()

    # ── Workflows ──────────────────────────────────────────────────────────────
    workflows = reader.list_workflows(_rd)
    pending = [w for w in workflows if w.get("status") == "pending_approval"]
    recent = workflows[:limit]

    console.print(
        f"  [bold]Workflows[/bold]  "
        f"[dim]({len(workflows)} total"
        + (f", [yellow]{len(pending)} awaiting approval[/yellow]" if pending else "")
        + ")[/dim]"
    )
    if recent:
        for w in recent:
            st = w.get("status", "?")
            icon = _WF_ICON.get(st, "·")
            wid = w.get("workflow_id", "?")
            state = w.get("current_state", "")
            started = _short_dt(w.get("started_at"))
            pend = w.get("pending_approval_id")
            pending_id = f"  [dim]→ {pend}[/dim]" if pend else ""
            console.print(f"    {icon}  {wid:<24} {st:<20} {state:<30} {started}{pending_id}")
    else:
        console.print("    [dim]No workflows found.[/dim]")

    console.print()

    # ── Models ─────────────────────────────────────────────────────────────────
    models = reader.list_models(_mod)
    console.print(f"  [bold]Models[/bold]  [dim]({len(models)} registered)[/dim]")
    if models:
        for m in models[:3]:
            name = m.get("model_name", "?")
            ver = m.get("version", "?")
            lineage = m.get("lineage") or {}
            map50 = lineage.get("map50")
            map50_str = f"mAP50={map50:.4f}" if isinstance(map50, int | float) else ""
            approved_by = lineage.get("approved_by") or m.get("approved_by", "")
            reg_at = _short_dt(m.get("registered_at"))
            by_str = f"by {approved_by}" if approved_by else ""
            meta = "  ".join(filter(None, [map50_str, by_str, reg_at]))
            console.print(f"    [green]✓[/green]  {name}  v{ver}  [dim]{meta}[/dim]")
    else:
        console.print("    [dim]No registered models.[/dim]")

    console.print()

    # ── Datasets ───────────────────────────────────────────────────────────────
    datasets = reader.list_datasets(_ds)
    console.print(f"  [bold]Datasets[/bold]  [dim]({len(datasets)} registered)[/dim]")
    if datasets:
        for d in datasets[:3]:
            name = d.get("dataset_name", "?")
            ver = d.get("version", "?")
            reg_at = _short_dt(d.get("registered_at"))
            console.print(f"    [green]✓[/green]  {name}  v{ver}  [dim]{reg_at}[/dim]")
    else:
        console.print("    [dim]No registered datasets.[/dim]")

    console.print()

    # ── Monitoring ─────────────────────────────────────────────────────────────
    mon_reports = reader.list_monitoring_reports(_rd)
    active_alerts = [
        r for r in mon_reports if r.get("recommended_action", "no_action") != "no_action"
    ]
    console.print(
        f"  [bold]Monitoring[/bold]  [dim]({len(mon_reports)} report(s)"
        + (f", [red]{len(active_alerts)} alert(s)[/red]" if active_alerts else "")
        + ")[/dim]"
    )
    if active_alerts:
        for r in active_alerts[:3]:
            action = r.get("recommended_action", "")
            color = _ACTION_COLOR.get(action, "white")
            endpoint = r.get("endpoint_name", "?")
            wid = r.get("workflow_id", "?")
            ts = _short_dt(r.get("generated_at"))
            console.print(
                f"    [{color}]![/{color}]  {endpoint}  [{color}]{action}[/{color}]"
                f"  [dim]{wid}  {ts}[/dim]"
            )
    elif mon_reports:
        latest = mon_reports[0]
        ts = _short_dt(latest.get("generated_at"))
        console.print(
            f"    [green]✓[/green]  [dim]Latest: "
            f"{latest.get('endpoint_name', '?')}  {ts}  no alerts[/dim]"
        )
    else:
        console.print("    [dim]No monitoring reports found.[/dim]")

    console.print()

    # ── Summary ────────────────────────────────────────────────────────────────
    issues = len([w for w in workflows if w.get("status") == "failed"]) + len(active_alerts)
    if pending:
        console.print(
            f"  [yellow bold]{len(pending)} workflow(s) waiting for human approval.[/yellow bold]  "
            "Run [dim]agentic-mlops run-workflow ... --resume[/dim] after setting the action."
        )
    if issues == 0 and not pending:
        console.print("  [green bold]All systems nominal.[/green bold]")
    console.print()


def show_state(
    workflow_id: str = typer.Argument(
        None,
        help="Workflow ID to inspect. Omit to list all available workflow IDs.",
    ),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    audit: bool = typer.Option(
        False,
        "--audit",
        help="Also print the last audit log events.",
    ),
    audit_lines: int = typer.Option(
        20,
        "--audit-lines",
        help="Number of recent audit events to show with --audit.",
    ),
) -> None:
    """Show the detailed state of a workflow run (step table, artifacts, approval info).

    Without WORKFLOW_ID, lists all workflow IDs found under --runs-dir.
    """
    from agentic_mlops.tools.state_inspector import WorkflowStateInspector  # noqa: PLC0415

    inspector = WorkflowStateInspector()

    if workflow_id is None:
        ids = inspector.list_workflows(runs_dir)
        if not ids:
            console.print(f"[dim]No workflows found in {runs_dir}.[/dim]")
            return
        console.print(f"\n  [bold]Workflows in {runs_dir}[/bold]\n")
        for wid in ids:
            snap = inspector.inspect(runs_dir, wid, max_audit_events=0)
            if snap is None:
                continue
            _STATUS_COLOR = {
                "completed": "green",
                "failed": "red",
                "running": "cyan",
                "pending_approval": "yellow",
                "blocked": "red",
            }
            color = _STATUS_COLOR.get(snap.status, "white")
            started = (snap.started_at or "")[:16].replace("T", " ")
            console.print(
                f"    [{color}]{snap.status:<20}[/{color}]  " f"{wid:<30}  [dim]{started}[/dim]"
            )
        console.print()
        return

    snap = inspector.inspect(runs_dir, workflow_id, max_audit_events=audit_lines)
    if snap is None:
        console.print(
            f"[red]No state file found for workflow '{workflow_id}' " f"in {runs_dir}.[/red]"
        )
        raise typer.Exit(code=1)

    _STATUS_COLOR = {
        "completed": "green",
        "failed": "red",
        "running": "cyan",
        "pending_approval": "yellow",
        "blocked": "red",
    }
    _STEP_ICON = {
        "completed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "skipped": "[dim]–[/dim]",
        "running": "[cyan]⟳[/cyan]",
        "pending": "[dim]·[/dim]",
    }

    color = _STATUS_COLOR.get(snap.status, "white")
    started = (snap.started_at or "—")[:19].replace("T", " ")
    updated = (snap.updated_at or "—")[:19].replace("T", " ")

    console.print()
    console.print(f"  [bold]Workflow:[/bold]  {snap.workflow_id}")
    console.print(
        f"  [bold]Status:[/bold]   [{color}]{snap.status}[/{color}]"
        f"  [dim]state={snap.current_state}[/dim]"
    )
    console.print(f"  [bold]Started:[/bold]  {started}    [bold]Updated:[/bold]  {updated}")
    if snap.mlflow_run_id:
        console.print(f"  [bold]MLflow:[/bold]   [dim]{snap.mlflow_run_id}[/dim]")
    if snap.pending_approval_id:
        console.print(f"  [bold yellow]Pending approval:[/bold yellow]  {snap.pending_approval_id}")
        console.print(
            "  [dim]→ Set the action in your orchestrator config and re-run with "
            "--resume (or use --resume-from-step)[/dim]"
        )

    console.print()
    console.print(
        f"  [bold]Steps[/bold]  [dim]("
        f"{snap.num_completed} completed"
        + (f", {snap.num_failed} failed" if snap.num_failed else "")
        + (f", {snap.num_skipped} skipped" if snap.num_skipped else "")
        + f" of {len(snap.step_summaries)})[/dim]"
    )
    console.print()

    for ss in snap.step_summaries:
        icon = _STEP_ICON.get(ss.status, "·")
        status_str = f"[dim]{ss.status}[/dim]" if ss.status == "pending" else ss.status
        console.print(f"    {icon}  {ss.name:<28}  {status_str}")
        for art in ss.artifacts[:2]:
            console.print(f"           [dim]{art}[/dim]")

    if audit and snap.audit_events:
        console.print()
        n = len(snap.audit_events)
        console.print(f"  [bold]Audit log[/bold]  [dim](last {n} events)[/dim]")
        console.print()
        for ev in snap.audit_events:
            ts = (ev.get("timestamp") or "")[:19].replace("T", " ")
            event_name = ev.get("event", "?")
            step = ev.get("step", "")
            detail = f"  [dim]{step}[/dim]" if step else ""
            console.print(f"    [dim]{ts}[/dim]  {event_name}{detail}")

    console.print(f"\n  [dim]State file: {snap.state_path}[/dim]")
    console.print()


def diff_runs(
    wf_id_a: str = typer.Argument(..., help="First (baseline) workflow ID"),
    wf_id_b: str = typer.Argument(..., help="Second (target) workflow ID"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
) -> None:
    """Compare two workflow runs: step outcomes and evaluation metrics."""
    from agentic_mlops.tools.run_diff import WorkflowRunDiffer

    diff = WorkflowRunDiffer().diff(runs_dir, wf_id_a, wf_id_b)

    if diff is None:
        missing = []
        from pathlib import Path as _P

        if not (_P(runs_dir) / wf_id_a / "state.json").exists():
            missing.append(wf_id_a)
        if not (_P(runs_dir) / wf_id_b / "state.json").exists():
            missing.append(wf_id_b)
        console.print(f"[red]Workflow(s) not found:[/red] {', '.join(missing)}")
        raise typer.Exit(1)

    _STATUS_COLOUR = {
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
        "running": "yellow",
        "pending": "dim",
    }

    def _sc(s: str | None) -> str:
        if s is None:
            return "[dim]—[/dim]"
        c = _STATUS_COLOUR.get(s, "white")
        return f"[{c}]{s}[/{c}]"

    console.print(f"\n[bold]diff-runs:[/bold] [cyan]{wf_id_a}[/cyan]  →  [cyan]{wf_id_b}[/cyan]")
    console.print("─" * 60)

    # Status
    sa_col = _sc(diff.status_a)
    sb_col = _sc(diff.status_b)
    changed_mark = "  [yellow]←[/yellow]" if diff.status_a != diff.status_b else ""
    console.print(f"  Status:  {sa_col}  →  {sb_col}{changed_mark}")
    if diff.started_at_a or diff.started_at_b:
        ta = (diff.started_at_a or "")[:19].replace("T", " ")
        tb = (diff.started_at_b or "")[:19].replace("T", " ")
        console.print(f"  Started: [dim]{ta}[/dim]  →  [dim]{tb}[/dim]")

    # Steps
    if diff.step_diffs or diff.steps_only_in_a or diff.steps_only_in_b:
        console.print("\n[bold]Steps[/bold]")
        for sd in diff.step_diffs:
            arrow = f"{_sc(sd.status_a)}  →  {_sc(sd.status_b)}"
            marker = "  [yellow]←[/yellow]" if sd.changed else ""
            console.print(f"  {sd.name:<28} {arrow}{marker}")
        for s in diff.steps_only_in_a:
            console.print(f"  [dim]{s:<28} only in {wf_id_a}[/dim]")
        for s in diff.steps_only_in_b:
            console.print(f"  [dim]{s:<28} only in {wf_id_b}[/dim]")

    # Metrics
    if diff.metric_diffs:
        console.print("\n[bold]Metrics[/bold]")
        current_step = ""
        for md in diff.metric_diffs:
            step, key = md.metric.split(".", 1)
            if step != current_step:
                console.print(f"  [bold dim]{step}[/bold dim]")
                current_step = step

            def _fmt(v: float | None) -> str:
                return f"{v:.4f}" if v is not None else "[dim]—[/dim]"

            val_str = f"{_fmt(md.value_a)}  →  {_fmt(md.value_b)}"
            if md.delta is not None:
                sign = "+" if md.delta >= 0 else ""
                arrow = "↑" if md.improved else ("↓" if md.improved is False else "")
                colour = "green" if md.improved else ("red" if md.improved is False else "dim")
                delta_str = f"  [{colour}]({sign}{md.delta:.4f} {arrow})[/{colour}]"
            else:
                delta_str = ""
            console.print(f"    {key:<20} {val_str}{delta_str}")

    if not diff.has_changes and not diff.metric_diffs:
        console.print("\n  [dim]No differences.[/dim]")
    console.print()


def prune_runs(
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
    keep_last: int = typer.Option(
        None,
        "--keep-last",
        help="Keep only the N most-recent runs; delete the rest.",
    ),
    older_than_days: float = typer.Option(
        None,
        "--older-than-days",
        help="Delete runs whose started_at is older than N days.",
    ),
    statuses: list[str] = typer.Option(
        None,
        "--status",
        help="Only delete runs with this status (repeatable). E.g. --status completed --status failed.",  # noqa: E501
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Show what would be deleted without actually deleting.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Delete old workflow run directories by age or count."""
    from agentic_mlops.tools.run_pruner import WorkflowPruner

    if keep_last is None and older_than_days is None:
        console.print("[red]Error:[/red] Specify --keep-last or --older-than-days.")
        raise typer.Exit(1)

    try:
        pruner = WorkflowPruner(
            keep_last=keep_last,
            older_than_days=older_than_days,
            statuses=list(statuses) if statuses else None,
            dry_run=True,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    preview = pruner.prune(runs_dir)

    if not preview.deleted:
        console.print("\n  [dim]Nothing to prune.[/dim]\n")
        return

    console.print(f"\n[bold]prune-runs[/bold]  {'(dry run)' if dry_run else ''}")
    console.print("─" * 50)
    for wf in preview.deleted:
        console.print(f"  [red]delete[/red]  {wf}")
    for wf in preview.kept:
        console.print(f"  [green]keep  [/green]  {wf}")
    if preview.skipped:
        for wf in preview.skipped:
            console.print(f"  [yellow]skip  [/yellow]  {wf}  (unreadable state)")
    console.print(f"\n  {len(preview.deleted)} run(s) would be deleted.")

    if dry_run:
        console.print()
        return

    if not yes:
        confirm = typer.confirm(f"\nDelete {len(preview.deleted)} run(s)?", default=False)
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    real_pruner = WorkflowPruner(
        keep_last=keep_last,
        older_than_days=older_than_days,
        statuses=list(statuses) if statuses else None,
        dry_run=False,
    )
    result = real_pruner.prune(runs_dir)
    console.print(f"\n  [green]Deleted {result.deleted_count} run(s).[/green]\n")


def tag_run(
    workflow_id: str = typer.Argument(..., help="Workflow ID to tag"),
    tags: list[str] = typer.Argument(
        None,
        help="Tags as key=value pairs (e.g. env=prod version=2). " "Omit to show current tags.",
    ),
    remove: list[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Tag key(s) to remove (repeatable).",
    ),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Workflow runs directory"),
) -> None:
    """Add, remove, or display tags on a workflow run.

    Set tags:   agentic-mlops tag-run wf_001 env=prod dataset=v3
    Remove tag: agentic-mlops tag-run wf_001 --remove env
    Show tags:  agentic-mlops tag-run wf_001
    """
    from agentic_mlops.tools.run_tagger import WorkflowRunTagger

    tagger = WorkflowRunTagger()

    try:
        if remove:
            current = tagger.remove(runs_dir, workflow_id, list(remove))
        elif tags:
            parsed: dict[str, str] = {}
            for item in tags:
                if "=" not in item:
                    console.print(f"[red]Error:[/red] Tag '{item}' must be in key=value format.")
                    raise typer.Exit(1)
                k, _, v = item.partition("=")
                parsed[k.strip()] = v.strip()
            current = tagger.add(runs_dir, workflow_id, parsed)
        else:
            current = tagger.get(runs_dir, workflow_id)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Tags for[/bold] [cyan]{workflow_id}[/cyan]")
    console.print("─" * 40)
    if current:
        for k, v in sorted(current.items()):
            console.print(f"  [green]{k}[/green] = {v}")
    else:
        console.print("  [dim](no tags)[/dim]")
    console.print()


def lint_config(
    config: str = typer.Argument(
        ...,
        help="Orchestrator config YAML to validate",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 even when the only issues are warnings (no errors).",
    ),
) -> None:
    """Validate an orchestrator YAML config without running anything.

    Checks step names, required fields, file path existence, step dependency order,
    and Azure backend / runner consistency.  No agents are started and no side effects
    are produced.

    Exit codes: 0 = ok (or warnings-only without --strict); 1 = errors present
    (or warnings with --strict).
    """
    from agentic_mlops.contracts.doctor import CheckStatus  # noqa: PLC0415
    from agentic_mlops.tools.config_linter import ConfigLinter  # noqa: PLC0415

    report = ConfigLinter().lint(config)

    _STATUS_ICON = {
        CheckStatus.OK: "[green]✓[/green]",
        CheckStatus.WARNING: "[yellow]⚠[/yellow]",
        CheckStatus.ERROR: "[red]✗[/red]",
    }

    console.print(f"\n[bold]agentic-mlops lint-config[/bold]  {config}\n")

    current_cat = ""
    for chk in report.checks:
        if chk.category != current_cat:
            current_cat = chk.category
            console.print(f"  [bold]{chk.category}[/bold]")
        icon = _STATUS_ICON[chk.status]
        console.print(f"    {icon}  {chk.message}")
        if chk.detail:
            console.print(f"         [dim]{chk.detail}[/dim]")

    console.print()
    ok_s = f"[green]{report.num_ok} ok[/green]"
    warn_s = f"[yellow]{report.num_warnings} warnings[/yellow]"
    err_s = f"[red]{report.num_errors} errors[/red]"
    console.print(f"  Summary: {ok_s}  {warn_s}  {err_s}")

    if report.overall_status == CheckStatus.OK:
        console.print("  [green bold]Config is valid.[/green bold]\n")
    elif report.overall_status == CheckStatus.WARNING:
        console.print("  [yellow]Config is valid with warnings — review above.[/yellow]\n")
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("  [red bold]Config has errors — fix them before running.[/red bold]\n")
        raise typer.Exit(code=1)


def cost_report(
    workflow_id: str = typer.Argument(..., help="Workflow ID to generate a cost report for"),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir", help="Directory containing workflow run state files"
    ),
    pricing_config: str = typer.Option(
        None,
        "--pricing-config",
        help=(
            "YAML file with ComputePricing config (rates: {instance_type: usd/hr}). "
            "Uses built-in Azure ML default rates when omitted."
        ),
    ),
    output_file: str = typer.Option(
        None,
        "--output-file",
        help="Write cost_summary.json to this path instead of the workflow artifacts dir.",
    ),
) -> None:
    """Estimate the compute cost of a completed workflow run.

    Reads training/evaluation timing from the workflow's state.json and outputs a
    cost summary.  Rates default to built-in Azure ML pricing; override with
    --pricing-config.
    """
    import json as _json

    from agentic_mlops.contracts.cost import ComputePricing
    from agentic_mlops.tools.cost_tracker import CostTracker

    workflow_dir = Path(runs_dir) / workflow_id
    state_path = workflow_dir / "state.json"
    if not state_path.exists():
        console.print(f"[red]State file not found:[/red] {state_path}")
        raise typer.Exit(code=1)

    state = _json.loads(state_path.read_text(encoding="utf-8"))
    step_outputs: dict = state.get("step_outputs", {})

    pricing = ComputePricing()
    if pricing_config:
        try:
            pricing = ComputePricing.from_yaml(pricing_config)
        except Exception as exc:
            console.print(f"[red]Cannot load pricing config:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    tracker = CostTracker(pricing)

    # Record training cost
    training = step_outputs.get("training")
    if training and not training.get("skipped"):
        tracker.record(
            step="training",
            runner=state.get("training_runner") or "unknown",
            compute_type=training.get("compute_type") or "unknown",
            started_at=training.get("started_at"),
            completed_at=training.get("completed_at"),
        )

    # Record evaluation cost
    evaluation = step_outputs.get("evaluation")
    if evaluation and not evaluation.get("skipped"):
        tracker.record(
            step="evaluation",
            runner=state.get("evaluation_runner") or "unknown",
            compute_type=evaluation.get("compute_type") or "unknown",
            started_at=evaluation.get("started_at"),
            completed_at=evaluation.get("completed_at"),
        )

    summary = tracker.summary(workflow_id)

    # Write JSON
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = workflow_dir / "artifacts" / "cost_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(summary.model_dump(), indent=2), encoding="utf-8")

    # Print table
    console.print(f"\n[bold]Cost Report — Workflow:[/bold] {workflow_id}")
    console.print(f"  Currency : {summary.currency}")
    console.print(f"  Steps    : {len(summary.entries)}")
    for entry in summary.entries:
        dur = f"{entry.duration_seconds:.0f}s" if entry.duration_seconds is not None else "—"
        console.print(
            f"  {entry.step:20s}  runner={entry.runner:18s}  "
            f"compute={entry.compute_type:22s}  "
            f"duration={dur:8s}  "
            f"cost={entry.estimated_cost:.4f} {entry.currency}"
        )
    console.print(
        f"\n  [bold]Total estimated cost:[/bold] {summary.total_cost:.4f} {summary.currency}"
    )
    console.print(f"\n  Report written to: {out_path}")
