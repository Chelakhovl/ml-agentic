"""Registry-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from .._shared import console


def compare_models(
    report_paths: list[str] = typer.Argument(
        ..., help="Paths to evaluation_report.json files to compare (two or more)"
    ),
    model_names: str = typer.Option(
        None,
        "--model-names",
        help="Comma-separated model labels (default: path stems)",
    ),
    promotion_policy: str = typer.Option(
        None,
        "--promotion-policy",
        help="Path to promotion_policy.yaml — marks which models pass the policy",
    ),
    weights: str = typer.Option(
        None,
        "--weights",
        help=("Comma-separated metric=value weights, e.g. " "'map50=0.5,recall=0.3,precision=0.2'"),
    ),
    latency_ms: str = typer.Option(
        None,
        "--latency-ms",
        help="Comma-separated measured latency in ms (index-aligned with report paths)",
    ),
    model_size_mb: str = typer.Option(
        None,
        "--model-size-mb",
        help="Comma-separated measured model size in MB (index-aligned with report paths)",
    ),
    output_dir: str = typer.Option(
        "outputs/comparison",
        "--output-dir",
        help="Where to write comparison_report.json/md",
    ),
) -> None:
    """Rank multiple trained models by a weighted composite score.

    Accepts two or more evaluation_report.json files produced by
    ``agentic-mlops evaluate``. Prints a ranked table and writes
    comparison_report.json and comparison_report.md.
    """
    from agentic_mlops.agents.model_comparison import ModelComparisonAgent  # noqa: PLC0415
    from agentic_mlops.contracts.model_comparison import (  # noqa: PLC0415
        ComparisonWeights,
        ModelComparisonInput,
    )

    if len(report_paths) < 2:
        console.print("[red]Provide at least two report paths to compare.[/red]")
        raise typer.Exit(code=1)

    parsed_names: list[str] | None = (
        [n.strip() for n in model_names.split(",")] if model_names else None
    )

    parsed_weights: ComparisonWeights | None = None
    if weights:
        try:
            wdict = dict(kv.split("=") for kv in weights.split(","))
            parsed_weights = ComparisonWeights(**{k: float(v) for k, v in wdict.items()})
        except Exception as exc:
            console.print(f"[red]Invalid --weights format: {exc}[/red]")
            raise typer.Exit(code=1)

    parsed_latency: list[float | None] | None = None
    if latency_ms:
        parsed_latency = [float(x.strip()) if x.strip() else None for x in latency_ms.split(",")]

    parsed_size: list[float | None] | None = None
    if model_size_mb:
        parsed_size = [float(x.strip()) if x.strip() else None for x in model_size_mb.split(",")]

    inp = ModelComparisonInput(
        report_paths=report_paths,
        model_names=parsed_names,
        promotion_policy_path=promotion_policy,
        weights=parsed_weights,
        measured_latency_ms=parsed_latency,
        measured_model_size_mb=parsed_size,
        output_dir=output_dir,
    )

    agent = ModelComparisonAgent(artifacts_dir=Path(output_dir))
    result = agent.run(inp)

    if result.success:
        console.print(f"\n[bold]Model Comparison — {result.total_models} models[/bold]")
        console.print(f"Winner: [green]{result.winner or '(none)'}[/green]")
        if result.policy_passing is not None:
            console.print(f"Policy passing: {result.policy_passing}/{result.total_models}")
        console.print("")
        for r in result.rankings:
            policy_mark = " ✓" if r.passes_policy else (" ✗" if r.passes_policy is False else "")
            map50 = f"{r.map50:.3f}" if r.map50 is not None else "—"
            console.print(
                f"  #{r.rank} {r.model_name}{policy_mark}  "
                f"mAP50={map50}  score={r.composite_score:.3f}"
            )
        if result.comparison_report_path:
            console.print(f"\nReport: {result.comparison_report_path}")
    else:
        console.print(f"[red]Comparison failed: {result.message}[/red]")
        for err in result.errors:
            console.print(f"  [red]• {err}[/red]")

    if not result.success:
        raise typer.Exit(code=1)
