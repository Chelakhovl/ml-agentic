"""Human Approval Agent — MVP Agent 4.

Reads an evaluation report, presents a decision summary to the human,
collects their decision, and writes structured approval artifacts.
No Azure ML, MLflow, or deployment calls are made.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.approvals import (
    ACTION_TO_STATUS,
    ApprovalAction,
    ApprovalInput,
    ApprovalOutput,
    ApprovalRequest,
    ApprovalStatus,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.report_writer import ReportWriter

# Recommendations that are not promote_candidate — approving requires extra caution
_RISKY_RECOMMENDATIONS: set[str] = {"retrain", "need_label_review"}

_MENU_CHOICES: list[tuple[str, str, ApprovalAction]] = [
    ("1", "Approve model", ApprovalAction.APPROVE_MODEL),
    ("2", "Reject model", ApprovalAction.REJECT_MODEL),
    ("3", "Request retraining", ApprovalAction.REQUEST_RETRAINING),
    ("4", "Request more data", ApprovalAction.REQUEST_MORE_DATA),
    ("5", "Request label review", ApprovalAction.REQUEST_LABEL_REVIEW),
    ("6", "Cancel", ApprovalAction.CANCEL),
]

_NEXT_ACTIONS: dict[ApprovalAction, list[str]] = {
    ApprovalAction.APPROVE_MODEL: [
        "Deploy model to staging",
        "Notify ML engineering team",
    ],
    ApprovalAction.REJECT_MODEL: [
        "Archive candidate model",
        "Review model quality criteria",
    ],
    ApprovalAction.REQUEST_RETRAINING: [
        "Retrain with updated config or more data",
    ],
    ApprovalAction.REQUEST_MORE_DATA: [
        "Collect more labelled examples",
        "Retrain after data augmentation",
    ],
    ApprovalAction.REQUEST_LABEL_REVIEW: [
        "Review and correct label quality",
        "Focus on critical classes",
    ],
    ApprovalAction.CANCEL: ["No action taken — workflow cancelled"],
}


def _load_eval_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _error_output(
    msg: str,
    evaluation_output_path: str,
    approval_request: ApprovalRequest | None = None,
) -> ApprovalOutput:
    return ApprovalOutput(
        success=False,
        message=msg,
        errors=[msg],
        evaluation_output_path=evaluation_output_path,
        approval_request=approval_request,
    )


class HumanApprovalAgent(BaseAgent):
    """Reads evaluation output, collects human decision, writes approval_decision artifacts.

    Output artifacts:
        artifacts_dir/approval_decision.json
        artifacts_dir/approval_decision.md
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

    def run(self, input: ApprovalInput) -> ApprovalOutput:
        self._log_start(evaluation_output_path=input.evaluation_output_path)

        # 1. Load evaluation report
        try:
            eval_report = _load_eval_report(input.evaluation_output_path)
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Cannot read evaluation report: {exc}"
            self.logger.error(msg)
            return _error_output(msg, input.evaluation_output_path)

        recommendation: str | None = eval_report.get("recommendation")
        metrics: dict = eval_report.get("metrics", {})
        failed_checks: list[str] = eval_report.get("failed_checks", [])
        eval_message: str = eval_report.get("message", "")

        # 2. Build ApprovalRequest
        eval_status = recommendation if recommendation else "failed"
        reasons = failed_checks or ([eval_message] if eval_message else ["Evaluation blocked"])
        approval_req = ApprovalRequest(
            candidate_model=input.candidate_model,
            dataset_version_or_path=input.dataset_version_or_path,
            evaluation_status=eval_status,
            recommendation=recommendation,
            key_metrics={
                k: float(metrics.get(k, 0.0)) for k in ("map50", "map50_95", "precision", "recall")
            },
            threshold_summary=eval_report.get("passed_checks", [])[:5],
            reasons=reasons,
        )

        # 3. Collect action
        if input.interactive:
            action, approver, comment = self._interactive_prompt(approval_req, recommendation)
        else:
            result = self._non_interactive_validate(input, recommendation, approval_req)
            if isinstance(result, ApprovalOutput):
                return result
            action, approver, comment = result

        # 4. Final safety guard: never approve a failed/blocked evaluation
        if action == ApprovalAction.APPROVE_MODEL and recommendation is None:
            msg = (
                "Cannot approve a blocked/failed evaluation. " "Choose reject, retrain, or cancel."
            )
            return _error_output(msg, input.evaluation_output_path, approval_req)

        # 5. Finalise request
        approval_req.next_actions = _NEXT_ACTIONS.get(action, [])

        # 6. Build output
        status = ACTION_TO_STATUS[action]
        timestamp = datetime.now(tz=UTC).isoformat()
        output = ApprovalOutput(
            success=status != ApprovalStatus.CANCELLED,
            message=f"Human approval recorded: {status}",
            status=status,
            action=action,
            approver=approver,
            timestamp=timestamp,
            evaluation_output_path=input.evaluation_output_path,
            approval_request=approval_req,
            comment=comment,
        )

        # 7. Write artifacts
        json_path, md_path = self._report_writer.write_approval_decision(output, self.artifacts_dir)
        output.generated_artifacts = [str(json_path), str(md_path)]
        output.artifacts = list(output.generated_artifacts)

        if self._mlflow and self._mlflow_run_id:
            self._log_to_mlflow(input, output)

        self._log_done(status=str(status), action=str(action), approver=str(approver))
        return output

    def _log_to_mlflow(self, inp: ApprovalInput, output: ApprovalOutput) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "approval.interactive": str(inp.interactive),
        }
        if output.approver:
            params["approval.approver"] = output.approver
        if output.action:
            params["approval.action"] = str(output.action)
        client.log_params(rid, params)

        action = output.action
        metrics: dict[str, float] = {
            "approval.approved": 1.0 if action == ApprovalAction.APPROVE_MODEL else 0.0,
            "approval.needs_retraining": (
                1.0 if action == ApprovalAction.REQUEST_RETRAINING else 0.0
            ),
            "approval.needs_more_data": (
                1.0 if action == ApprovalAction.REQUEST_MORE_DATA else 0.0
            ),
            "approval.needs_label_review": (
                1.0 if action == ApprovalAction.REQUEST_LABEL_REVIEW else 0.0
            ),
        }
        client.log_metrics(rid, metrics)

        client.log_tags(
            rid,
            {
                "workflow_step": "human_approval",
                "approval_status": str(output.status),
                "approval_action": str(output.action or "none"),
            },
        )

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)

    # ── non-interactive ────────────────────────────────────────────────────────

    def _non_interactive_validate(
        self,
        inp: ApprovalInput,
        recommendation: str | None,
        approval_req: ApprovalRequest,
    ) -> tuple[ApprovalAction, str | None, str | None] | ApprovalOutput:
        if inp.action is None:
            return _error_output(
                "Non-interactive mode requires --action.",
                inp.evaluation_output_path,
                approval_req,
            )

        action = inp.action

        # Cannot approve a failed evaluation — not even with --force
        if action == ApprovalAction.APPROVE_MODEL and recommendation is None:
            return _error_output(
                "Cannot approve a blocked/failed evaluation. " "Choose reject, retrain, or cancel.",
                inp.evaluation_output_path,
                approval_req,
            )

        # Risky approval (non-promote recommendation) requires --force
        if action == ApprovalAction.APPROVE_MODEL and recommendation in _RISKY_RECOMMENDATIONS:
            if not inp.force:
                return _error_output(
                    f"Cannot approve a non-promoted candidate in non-interactive mode "
                    f"without --force. Recommendation was '{recommendation}'.",
                    inp.evaluation_output_path,
                    approval_req,
                )

        return action, inp.approver, inp.comment

    # ── interactive ────────────────────────────────────────────────────────────

    def _interactive_prompt(
        self,
        req: ApprovalRequest,
        recommendation: str | None,
    ) -> tuple[ApprovalAction, str | None, str | None]:
        _print_approval_summary(req)

        is_failed = recommendation is None
        is_risky = recommendation in _RISKY_RECOMMENDATIONS

        action: ApprovalAction | None = None
        while action is None:
            print("\nAvailable actions:")
            for key, label, act in _MENU_CHOICES:
                blocked = is_failed and act == ApprovalAction.APPROVE_MODEL
                suffix = " [BLOCKED — evaluation failed/blocked]" if blocked else ""
                print(f"  {key}. {label}{suffix}")

            choice = self._input_fn("\nEnter choice (1-6): ").strip()
            matched = next((act for k, _, act in _MENU_CHOICES if k == choice), None)
            if matched is None:
                print("Invalid choice. Please enter 1-6.")
                continue

            if matched == ApprovalAction.APPROVE_MODEL and is_failed:
                print("Cannot approve a blocked/failed evaluation. Choose another action.")
                continue

            if matched == ApprovalAction.APPROVE_MODEL and is_risky:
                confirm = self._input_fn(
                    f"\nWARNING: Evaluation recommendation is '{recommendation}'. "
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


def _print_approval_summary(req: ApprovalRequest) -> None:
    print("\n" + "=" * 60)
    print("HUMAN APPROVAL GATE — Evaluation Summary")
    print("=" * 60)
    print(f"  Candidate model  : {req.candidate_model}")
    print(f"  Dataset          : {req.dataset_version_or_path}")
    print(f"  Recommendation   : {req.recommendation or 'N/A (blocked/failed)'}")
    print(f"  Evaluation status: {req.evaluation_status}")
    if req.key_metrics:
        print("\n  Key Metrics:")
        for k, v in req.key_metrics.items():
            print(f"    {k:<12}: {v:.4f}")
    if req.reasons:
        print("\n  Reasons / Issues:")
        for r in req.reasons:
            print(f"    - {r}")
    print("=" * 60)
