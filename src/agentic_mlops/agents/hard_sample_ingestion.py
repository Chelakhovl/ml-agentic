"""HardSampleIngestionAgent — closes the monitoring → data-intake feedback loop.

Reads a hard_samples_manifest.json produced by MonitoringAgent, stages the
referenced images into a temporary directory, then runs DataIntakeAgent over
that directory so the images enter the normal data-quality pipeline.

Standalone agent; deliberately not wired into run-mvp or OrchestratorWorkflow
because monitoring itself is a recurring, post-deploy concern that runs
independently from the forward pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..contracts.data_intake import DataIntakeInput
from ..contracts.hard_sample_ingestion import (
    HardSampleIngestionInput,
    HardSampleIngestionOutput,
)
from ..tools.hard_sample_ingester import HardSampleIngester
from .base import BaseAgent
from .data_intake import DataIntakeAgent

if TYPE_CHECKING:
    from ..integrations.mlflow_client import MLflowTrackingClientBase


class HardSampleIngestionAgent(BaseAgent):
    """Stage hard samples from a monitoring manifest and run DataIntakeAgent over them."""

    def __init__(
        self,
        artifacts_dir: Path,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._ingester = HardSampleIngester()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    # ------------------------------------------------------------------
    def run(self, input: HardSampleIngestionInput) -> HardSampleIngestionOutput:  # noqa: A002
        self._log_start(
            manifest=input.hard_samples_manifest_path,
            source_dir=input.images_source_dir,
            dataset_name=input.dataset_name,
        )

        manifest_path = Path(input.hard_samples_manifest_path)
        if not manifest_path.exists():
            return self._failed(f"hard_samples_manifest_path not found: {manifest_path}")

        source_dir = Path(input.images_source_dir).resolve()
        if not source_dir.is_dir():
            return self._failed(f"images_source_dir not found or not a directory: {source_dir}")

        staging_dir = self.artifacts_dir / "hard_samples_staging"
        num_total, num_found, num_not_found, stage_warnings = self._ingester.stage(
            manifest_path=manifest_path,
            source_dir=source_dir,
            staging_dir=staging_dir,
        )

        if num_found == 0:
            msg = (
                f"No hard sample images could be located in {source_dir} "
                f"({num_total} entries in manifest)."
            )
            self._log_done(status="failed", num_found=0)
            return HardSampleIngestionOutput(
                success=False,
                message=msg,
                num_hard_samples_in_manifest=num_total,
                num_images_found=0,
                num_images_not_found=num_not_found,
                staging_dir=str(staging_dir),
                warnings=stage_warnings,
            )

        # Run DataIntakeAgent on the staging directory
        intake_artifacts_dir = self.artifacts_dir / "data_intake"
        intake_agent = DataIntakeAgent(
            artifacts_dir=intake_artifacts_dir,
            mlflow_client=self._mlflow,
            mlflow_run_id=self._mlflow_run_id,
        )
        intake_output = intake_agent.run(
            DataIntakeInput(
                raw_data_path=str(staging_dir),
                dataset_name=input.dataset_name,
                source=input.source or "hard_sample_mining",
                expected_formats=input.expected_formats,
                min_files=input.min_files,
                corrupted_ratio_threshold=input.corrupted_ratio_threshold,
                duplicate_ratio_threshold=input.duplicate_ratio_threshold,
            )
        )

        all_warnings = stage_warnings + intake_output.warnings
        all_artifacts = [str(staging_dir)] + intake_output.artifacts

        self._log_done(
            status=str(intake_output.status)
            if hasattr(intake_output, "status")
            else ("passed" if intake_output.success else "failed"),
            num_staged=num_found,
            num_not_found=num_not_found,
        )

        return HardSampleIngestionOutput(
            success=intake_output.success,
            message=(
                f"Staged {num_found}/{num_total} hard samples; "
                f"DataIntake: {intake_output.message}"
            ),
            artifacts=all_artifacts,
            warnings=all_warnings,
            errors=intake_output.errors,
            num_hard_samples_in_manifest=num_total,
            num_images_found=num_found,
            num_images_not_found=num_not_found,
            staging_dir=str(staging_dir),
            intake_status=str(intake_output.status) if hasattr(intake_output, "status") else None,
            intake_manifest_path=intake_output.manifest_path
            if hasattr(intake_output, "manifest_path")
            else None,
        )

    # ------------------------------------------------------------------
    def _failed(self, message: str) -> HardSampleIngestionOutput:
        self._log_done(status="failed")
        return HardSampleIngestionOutput(success=False, message=message)
