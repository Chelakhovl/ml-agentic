"""Automatic baseline comparison tool.

Reads two evaluation_report.json files (current eval + saved baseline) and
computes per-metric deltas.  A positive delta means the new model improved.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.contracts.baseline import BaselineComparisonResult
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_METRICS_KEYS = ("map50", "map50_95", "precision", "recall")


def _read_metrics(path: Path) -> dict[str, float]:
    """Extract the top-level metrics dict from an evaluation_report.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("metrics") or {}
    return {k: float(raw.get(k, 0.0)) for k in _METRICS_KEYS}


class BaselineComparator:
    """Compare a new evaluation result against a saved baseline report.

    Args:
        min_improvement_map50: Minimum Δ mAP50 required to mark ``improved=True``.
            Defaults to 0.0 (any improvement counts).
    """

    def __init__(self, min_improvement_map50: float = 0.0) -> None:
        self._threshold = min_improvement_map50

    def compare(
        self,
        eval_report_path: str | Path,
        baseline_path: str | Path,
        output_dir: Path | None = None,
    ) -> BaselineComparisonResult:
        """Compare *eval_report_path* against *baseline_path*.

        Both paths must point to files that contain a ``metrics`` dict with
        ``map50``, ``precision``, and ``recall`` fields (the shape written by
        ``EvaluationAgent``).

        If *output_dir* is given, writes ``baseline_comparison.json`` there.
        """
        eval_path = Path(eval_report_path)
        base_path = Path(baseline_path)

        try:
            new_m = _read_metrics(eval_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.error("BaselineComparator: cannot read eval report", extra={"error": str(exc)})
            raise

        try:
            base_m = _read_metrics(base_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "BaselineComparator: cannot read baseline report", extra={"error": str(exc)}
            )
            raise

        delta_map50 = round(new_m["map50"] - base_m["map50"], 6)
        delta_map50_95 = round(new_m["map50_95"] - base_m["map50_95"], 6)
        delta_precision = round(new_m["precision"] - base_m["precision"], 6)
        delta_recall = round(new_m["recall"] - base_m["recall"], 6)

        result = BaselineComparisonResult(
            new_map50=new_m["map50"],
            new_map50_95=new_m["map50_95"],
            new_precision=new_m["precision"],
            new_recall=new_m["recall"],
            baseline_map50=base_m["map50"],
            baseline_map50_95=base_m["map50_95"],
            baseline_precision=base_m["precision"],
            baseline_recall=base_m["recall"],
            delta_map50=delta_map50,
            delta_map50_95=delta_map50_95,
            delta_precision=delta_precision,
            delta_recall=delta_recall,
            improved=delta_map50 >= self._threshold,
            min_improvement_map50=self._threshold,
            baseline_path=str(base_path),
            eval_report_path=str(eval_path),
            generated_at=datetime.now(tz=UTC).isoformat(),
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "baseline_comparison.json"
            out_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            logger.info(
                "Baseline comparison written",
                extra={"path": str(out_path), "delta_map50": delta_map50},
            )

        return result


class FakeBaselineComparator(BaselineComparator):
    """Test double — records calls; returns a configurable result."""

    def __init__(
        self,
        result: BaselineComparisonResult | None = None,
        min_improvement_map50: float = 0.0,
    ) -> None:
        super().__init__(min_improvement_map50)
        self._result = result or BaselineComparisonResult(
            new_map50=0.9,
            baseline_map50=0.85,
            delta_map50=0.05,
            improved=True,
        )
        self.calls: list[tuple[str, str]] = []

    def compare(
        self,
        eval_report_path: str | Path,
        baseline_path: str | Path,
        output_dir: Path | None = None,
    ) -> BaselineComparisonResult:
        self.calls.append((str(eval_report_path), str(baseline_path)))
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "baseline_comparison.json"
            out_path.write_text(
                json.dumps(self._result.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
        return self._result
