"""Training Approval Agent — H4 gate: human sign-off before a training run starts.

Reads a dataset_quality_report.json (written by DatasetValidationAgent), presents
a summary, collects a human decision, and writes structured approval artifacts.
Mirrors HumanApprovalAgent's interactive/non-interactive shape (H5 Model
Approval), scoped down to a training-time question: proceed to train on this
validated dataset, or not.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.training_approval import (
    ACTION_TO_STATUS,
    TrainingApprovalAction,
    TrainingApprovalInput,
    TrainingApprovalOutput,
    TrainingApprovalStatus,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.report_writer import ReportWriter

_MENU_CHOICES: list[tuple[str, str, TrainingApprovalAction]] = [
    ("1", "Approve training", TrainingApprovalAction.APPROVE_TRAINING),
    ("2", "Reject training", TrainingApprovalAction.REJECT_TRAINING),
    ("3", "Cancel", TrainingApprovalAction.CANCEL),
]


def _load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _error_output(msg: str, dataset_report_path: str) -> TrainingApprovalOutput:
    return TrainingApprovalOutput(
        success=False, message=msg, errors=[msg], dataset_report_path=dataset_report_path
    )


class TrainingApprovalAgent(BaseAgent):
    """Reads a dataset quality report, collects a human decision to proceed to training.

    Output artifacts:
        artifacts_dir/training_approval_decision.json
        artifacts_dir/training_approval_decision.md
    """

    def __init__(
        self,
        artifacts_dir: Path,
        _input_fn: Callable[[str], str] = input,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._input_fn = _input_fn
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: TrainingApprovalInput) -> TrainingApprovalOutput:
        self._log_start(dataset_report_path=input.dataset_report_path)

        try:
            report = _load_report(input.dataset_report_path)
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Cannot read dataset report: {exc}"
            self.logger.error(msg)
            return _error_output(msg, input.dataset_report_path)

        status: str = report.get("status", "failed")
        num_images: int = report.get("num_images", 0)
        is_failed = status == "failed"
        is_risky = status == "warning"

        if input.interactive:
            action, approver, comment = self._interactive_prompt(
                report, status, is_failed, is_risky
            )
        else:
            result = self._non_interactive_validate(input, status, is_failed, is_risky)
            if isinstance(result, TrainingApprovalOutput):
                return result
            action, approver, comment = result

        # Final safety guard: never approve a failed dataset validation.
        if action == TrainingApprovalAction.APPROVE_TRAINING and is_failed:
            msg = "Cannot approve training on a failed dataset validation. Choose reject or cancel."
            return _error_output(msg, input.dataset_report_path)

        status_out = ACTION_TO_STATUS[action]
        timestamp = datetime.now(tz=UTC).isoformat()
        output = TrainingApprovalOutput(
            success=status_out != TrainingApprovalStatus.CANCELLED,
            message=f"Training approval recorded: {status_out}",
            status=status_out,
            action=action,
            approver=approver,
            timestamp=timestamp,
            dataset_report_path=input.dataset_report_path,
            comment=comment,
        )

        json_path, md_path = self._report_writer.write_training_approval_decision(
            output, self.artifacts_dir
        )
        output.generated_artifacts = [str(json_path), str(md_path)]
        output.artifacts = list(output.generated_artifacts)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output, num_images)

        self._log_done(status=str(status_out), action=str(action), approver=str(approver))
        return output

    def _log_to_mlflow(
        self, inp: TrainingApprovalInput, output: TrainingApprovalOutput, num_images: int
    ) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {"training_approval.interactive": str(inp.interactive)}
        if output.approver:
            params["training_approval.approver"] = output.approver
        if output.action:
            params["training_approval.action"] = str(output.action)
        client.log_params(rid, params)

        client.log_metrics(rid, {"training_approval.num_images": float(num_images)})
        client.log_tags(rid, {
            "workflow_step": "training_approval",
            "training_approval_status": str(output.status),
        })
        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)

    # ── non-interactive ────────────────────────────────────────────────────────

    def _non_interactive_validate(
        self, inp: TrainingApprovalInput, status: str, is_failed: bool, is_risky: bool
    ) -> tuple[TrainingApprovalAction, str | None, str | None] | TrainingApprovalOutput:
        if inp.action is None:
            return _error_output(
                "Non-interactive mode requires --action.", inp.dataset_report_path
            )

        action = inp.action

        if action == TrainingApprovalAction.APPROVE_TRAINING and is_failed:
            return _error_output(
                "Cannot approve training on a failed dataset validation. "
                "Choose reject or cancel.",
                inp.dataset_report_path,
            )

        if action == TrainingApprovalAction.APPROVE_TRAINING and is_risky and not inp.force:
            return _error_output(
                f"Cannot approve training on a dataset with status '{status}' in "
                "non-interactive mode without --force.",
                inp.dataset_report_path,
            )

        return action, inp.approver, inp.comment

    # ── interactive ────────────────────────────────────────────────────────────

    def _interactive_prompt(
        self, report: dict, status: str, is_failed: bool, is_risky: bool
    ) -> tuple[TrainingApprovalAction, str | None, str | None]:
        _print_training_approval_summary(report, status)

        action: TrainingApprovalAction | None = None
        while action is None:
            print("\nAvailable actions:")
            for key, label, act in _MENU_CHOICES:
                blocked = is_failed and act == TrainingApprovalAction.APPROVE_TRAINING
                suffix = " [BLOCKED — dataset validation failed]" if blocked else ""
                print(f"  {key}. {label}{suffix}")

            choice = self._input_fn("\nEnter choice (1-3): ").strip()
            matched = next((act for k, _, act in _MENU_CHOICES if k == choice), None)
            if matched is None:
                print("Invalid choice. Please enter 1-3.")
                continue

            if matched == TrainingApprovalAction.APPROVE_TRAINING and is_failed:
                print(
                    "Cannot approve training on a failed dataset validation. "
                    "Choose another action."
                )
                continue

            if matched == TrainingApprovalAction.APPROVE_TRAINING and is_risky:
                confirm = self._input_fn(
                    f"\nWARNING: Dataset validation status is '{status}'. "
                    "Type 'CONFIRM' to approve anyway: "
                ).strip()
                if confirm != "CONFIRM":
                    print("Approval not confirmed. Please choose again.")
                    continue

            action = matched

        approver_raw = self._input_fn("Approver name (leave blank to skip): ").strip()
        approver = approver_raw or None

        comment_raw = self._input_fn("Comment (leave blank to skip): ").strip()
        comment = comment_raw or None

        return action, approver, comment


def _print_training_approval_summary(report: dict, status: str) -> None:
    print("\n" + "=" * 60)
    print("TRAINING APPROVAL GATE (H4) — Dataset Validation Summary")
    print("=" * 60)
    print(f"  Dataset status   : {status}")
    print(f"  Images           : {report.get('num_images', 0)}")
    print(f"  Label files      : {report.get('num_labels', 0)}")
    blocking = report.get("blocking_issues", [])
    warnings = report.get("warnings", [])
    if blocking:
        print("\n  Blocking Issues:")
        for issue in blocking:
            print(f"    - {issue}")
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    print("=" * 60)
