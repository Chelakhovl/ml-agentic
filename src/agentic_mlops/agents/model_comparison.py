"""Model Comparison Agent — compares multiple trained models and ranks them.

Standalone agent; not part of MVPWorkflow or OrchestratorWorkflow (model
comparison is typically a one-off analysis step, not a per-run pipeline step).
CLI: ``agentic-mlops compare-models <report1> <report2> ... [options]``
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.model_comparison import (
    ModelComparisonInput,
    ModelComparisonOutput,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.model_comparator import ModelComparator


class ModelComparisonAgent(BaseAgent):
    """Loads a set of evaluation reports, ranks the models, and writes a report.

    Output artifacts:
        artifacts_dir/comparison_report.json
        artifacts_dir/comparison_report.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        comparator: ModelComparator | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._comparator = comparator or ModelComparator()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: ModelComparisonInput) -> ModelComparisonOutput:
        self._log_start(n_models=len(input.report_paths))

        output = self._comparator.compare(input)

        if output.success:
            json_path, md_path = self._write_reports(input, output)
            output.comparison_report_path = str(json_path)
            output.md_report_path = str(md_path)
            for p in (str(json_path), str(md_path)):
                if p not in output.artifacts:
                    output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id and output.success:
            self._log_to_mlflow(output)

        status = "completed" if output.success else "failed"
        self._log_done(status, winner=output.winner, total=str(output.total_models))
        return output

    # ── report writing ────────────────────────────────────────────────────────

    def _write_reports(
        self, inp: ModelComparisonInput, out: ModelComparisonOutput
    ) -> tuple[Path, Path]:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "total_models": out.total_models,
            "winner": out.winner,
            "winner_report_path": out.winner_report_path,
            "policy_passing": out.policy_passing,
            "weights": (
                (inp.weights or {}).model_dump()
                if hasattr(inp.weights or {}, "model_dump")
                else None
            ),
            "rankings": [r.model_dump() for r in out.rankings],
        }
        json_path = self.artifacts_dir / "comparison_report.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        md_path = self.artifacts_dir / "comparison_report.md"
        md_path.write_text(_md_report(payload, out), encoding="utf-8")
        return json_path, md_path

    # ── MLflow ────────────────────────────────────────────────────────────────

    def _log_to_mlflow(self, out: ModelComparisonOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        if out.rankings:
            winner = out.rankings[0]
            metrics: dict[str, float] = {"comparison.top_composite_score": winner.composite_score}
            if winner.map50 is not None:
                metrics["comparison.winner_map50"] = winner.map50
            if winner.recall is not None:
                metrics["comparison.winner_recall"] = winner.recall
            client.log_metrics(rid, metrics)

        client.log_params(
            rid,
            {
                "comparison.total_models": str(out.total_models),
                "comparison.winner": out.winner or "",
            },
        )
        client.log_tags(rid, {"workflow_step": "model_comparison"})


def _md_report(payload: dict, out: ModelComparisonOutput) -> str:
    lines = [
        "# Model Comparison Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Total models: {payload['total_models']}",
    ]
    if out.winner:
        lines += [f"**Winner: {out.winner}**"]
    if payload["policy_passing"] is not None:
        passing = payload["policy_passing"]
        total = payload["total_models"]
        lines += [f"Models passing promotion policy: {passing}/{total}"]
    lines += ["", "## Rankings", ""]
    lines += [
        "| Rank | Model | mAP50 | Precision | Recall | mAP50-95 | Latency (ms) | Size (MB) | Score | Policy |"  # noqa: E501
    ]
    lines += [
        "|------|-------|-------|-----------|--------|----------|--------------|-----------|-------|--------|"  # noqa: E501
    ]
    for r in out.rankings:

        def _fmt(v: object) -> str:
            return f"{v:.3f}" if isinstance(v, float) else "—"

        policy = "✓" if r.passes_policy else ("✗" if r.passes_policy is False else "—")
        lines.append(
            f"| {r.rank} | {r.model_name} | {_fmt(r.map50)} | {_fmt(r.precision)} | "
            f"{_fmt(r.recall)} | {_fmt(r.map50_95)} | {_fmt(r.latency_ms)} | "
            f"{_fmt(r.model_size_mb)} | {r.composite_score:.3f} | {policy} |"
        )
    lines += [""]
    for r in out.rankings:
        if r.notes:
            lines += [f"**{r.model_name}**: " + "; ".join(r.notes)]
    return "\n".join(lines) + "\n"
