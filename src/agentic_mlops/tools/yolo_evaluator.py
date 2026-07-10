"""YOLO model evaluator — dispatches to runner implementations.

local_dry_run : FakeEvaluationRunner (deterministic fake metrics).
local_eval    : LocalYOLOEvaluationRunner (real Ultralytics .val()).
azure_eval    : AzureMLEvaluationRunner (requires an injected instance — see azure_runner).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentic_mlops.contracts.evaluation import (
    EvaluationInput,
    EvaluationMetrics,
    EvaluationMode,
    EvaluationOutput,
)
from agentic_mlops.tools.evaluation_runner import FakeEvaluationRunner, LocalYOLOEvaluationRunner

if TYPE_CHECKING:
    from agentic_mlops.tools.evaluation_runner import AzureMLEvaluationRunner


class YoloEvaluator:
    """Runs YOLO evaluation or a deterministic dry-run.

    Args:
        dry_run_override_metrics: Inject custom metrics in dry-run mode (tests).
        azure_runner: Inject an AzureMLEvaluationRunner (built from AzureMLConfig)
            to enable azure_eval mode.
    """

    def __init__(
        self,
        dry_run_override_metrics: EvaluationMetrics | None = None,
        azure_runner: AzureMLEvaluationRunner | None = None,
    ) -> None:
        self._override = dry_run_override_metrics
        self._azure_runner = azure_runner

    def run(self, inp: EvaluationInput, artifacts_dir: Path) -> EvaluationOutput:
        if inp.mode == EvaluationMode.LOCAL_DRY_RUN:
            return FakeEvaluationRunner(override_metrics=self._override).run(inp, artifacts_dir)
        if inp.mode == EvaluationMode.LOCAL_EVAL:
            return LocalYOLOEvaluationRunner().run(inp, artifacts_dir)
        if inp.mode == EvaluationMode.AZURE_EVAL:
            if self._azure_runner is None:
                raise RuntimeError(
                    "Azure ML evaluation requires an AzureMLEvaluationRunner — "
                    "pass --azure-config on the CLI or inject azure_runner."
                )
            return self._azure_runner.run(inp, artifacts_dir)
        raise ValueError(f"Unknown evaluation mode: {inp.mode}")
