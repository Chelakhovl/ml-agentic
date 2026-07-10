"""Model Registry Agent — MVP Agent 5.

Runs after Human Approval. Reads training_output.json, evaluation_output.json,
and approval_decision.json from disk. Builds ModelLineage, then delegates
to a ModelRegistryClientBase to store best.pt + lineage + model card.

Gates (all must pass to register):
  1. Approval status must be "approved"
  2. Approval action must be "approve_model"
  3. Evaluation must have succeeded (success=True)
  4. Training job_status must be "completed"
  5. best_weights_path in training_output.json must exist on disk
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.approvals import ApprovalAction, ApprovalStatus
from agentic_mlops.contracts.model_registry import (
    ModelLineage,
    ModelRegistrationInput,
    ModelRegistrationOutput,
    RegistrationStatus,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.integrations.model_registry import (
    ModelRegistryClientBase,
    create_registry_client,
)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _blocked(msg: str, model_name: str = "") -> ModelRegistrationOutput:
    return ModelRegistrationOutput(
        success=False,
        message=msg,
        status=RegistrationStatus.BLOCKED,
        model_name=model_name,
        block_reason=msg,
        errors=[msg],
    )


def _failed(msg: str, model_name: str = "") -> ModelRegistrationOutput:
    return ModelRegistrationOutput(
        success=False,
        message=msg,
        status=RegistrationStatus.FAILED,
        model_name=model_name,
        errors=[msg],
    )


class ModelRegistryAgent(BaseAgent):
    """Reads pipeline outputs from disk and registers the approved model.

    Output artifacts (in artifacts_dir):
        registration_output.json
    Registry artifacts (in registry_dir/<model_name>/versions/<N>/):
        model/best.pt
        lineage.json
        model_card.md
        registration_output.json
    """

    def __init__(
        self,
        artifacts_dir: Path,
        registry_client: ModelRegistryClientBase | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        # None = resolve from ModelRegistrationInput.backend at run() time;
        # explicit value = test override / forced backend, takes precedence.
        self._registry_client = registry_client
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, inp: ModelRegistrationInput) -> ModelRegistrationOutput:
        self._log_start(
            model_name=inp.model_name,
            backend=str(inp.backend),
        )

        # ── Load pipeline outputs ─────────────────────────────────────────────
        try:
            train_data = _load_json(inp.training_output_path)
        except (OSError, json.JSONDecodeError) as exc:
            return _failed(
                f"Cannot read training_output.json: {exc}", inp.model_name
            )

        try:
            eval_data = _load_json(inp.evaluation_output_path)
        except (OSError, json.JSONDecodeError) as exc:
            return _failed(
                f"Cannot read evaluation_output.json: {exc}", inp.model_name
            )

        try:
            approval_data = _load_json(inp.approval_decision_path)
        except (OSError, json.JSONDecodeError) as exc:
            return _failed(
                f"Cannot read approval_decision.json: {exc}", inp.model_name
            )

        # ── Gate 1: approval status ───────────────────────────────────────────
        approval_status = approval_data.get("status", "")
        if approval_status != ApprovalStatus.APPROVED:
            return _blocked(
                f"Approval status is '{approval_status}', not 'approved'. "
                "Registration requires approved status.",
                inp.model_name,
            )

        # ── Gate 2: approval action ───────────────────────────────────────────
        approval_action = approval_data.get("action", "")
        if approval_action != ApprovalAction.APPROVE_MODEL:
            return _blocked(
                f"Approval action is '{approval_action}', not 'approve_model'. "
                "Registration requires approve_model action.",
                inp.model_name,
            )

        # ── Gate 3: evaluation success ────────────────────────────────────────
        if not eval_data.get("success", False):
            return _blocked(
                "Evaluation did not succeed. Registration requires a passing evaluation.",
                inp.model_name,
            )

        # ── Gate 4: training completed ────────────────────────────────────────
        training_job_status = train_data.get("job_status", "")
        if training_job_status != "completed":
            return _blocked(
                f"Training job_status is '{training_job_status}', not 'completed'.",
                inp.model_name,
            )

        # ── Gate 5: best.pt exists ────────────────────────────────────────────
        best_weights_str = train_data.get("best_weights_path")
        if not best_weights_str:
            # Fall back to sibling of training_output.json
            candidate = Path(inp.training_output_path).parent / "best.pt"
            if candidate.exists():
                best_weights_str = str(candidate)
            else:
                return _blocked(
                    "best_weights_path not set in training_output.json and best.pt not found.",
                    inp.model_name,
                )
        else:
            if not Path(best_weights_str).exists():
                return _blocked(
                    f"best.pt not found at '{best_weights_str}'.",
                    inp.model_name,
                )

        # ── Build ModelLineage ────────────────────────────────────────────────
        eval_metrics: dict = eval_data.get("metrics", {})

        lineage = ModelLineage(
            training_job_id=train_data.get("job_id"),
            training_runner=train_data.get("runner"),
            training_model=None,
            training_config=None,
            data_yaml=None,
            dataset_path=None,
            map50=eval_metrics.get("map50"),
            map50_95=eval_metrics.get("map50_95"),
            precision=eval_metrics.get("precision"),
            recall=eval_metrics.get("recall"),
            evaluation_recommendation=eval_data.get("recommendation"),
            approved_by=approval_data.get("approver"),
            approval_action=approval_action,
            approval_timestamp=approval_data.get("timestamp"),
            approval_comment=approval_data.get("comment"),
            mlflow_run_id=inp.mlflow_run_id,
            mlflow_experiment_name=inp.mlflow_experiment_name,
            mlflow_tracking_uri=inp.mlflow_tracking_uri,
            training_output_path=inp.training_output_path,
            evaluation_output_path=inp.evaluation_output_path,
            approval_decision_path=inp.approval_decision_path,
        )

        # ── Delegate to registry client ───────────────────────────────────────
        client = self._registry_client or create_registry_client(inp.backend)
        output = client.register(inp, lineage, self.artifacts_dir)

        # ── MLflow logging ────────────────────────────────────────────────────
        if self._mlflow and self._mlflow_run_id and output.success:
            self._log_to_mlflow(inp, output, lineage)

        self._log_done(
            status=str(output.status),
            model_name=inp.model_name,
            version=str(output.version),
        )
        return output

    def _log_to_mlflow(
        self,
        inp: ModelRegistrationInput,
        output: ModelRegistrationOutput,
        lineage: ModelLineage,
    ) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "registry.model_name": inp.model_name,
            "registry.backend": str(inp.backend),
            "registry.version": str(output.version),
        }
        client.log_params(rid, params)

        metrics: dict[str, float] = {}
        if lineage.map50 is not None:
            metrics["registry.map50"] = lineage.map50
        if lineage.map50_95 is not None:
            metrics["registry.map50_95"] = lineage.map50_95
        if lineage.precision is not None:
            metrics["registry.precision"] = lineage.precision
        if lineage.recall is not None:
            metrics["registry.recall"] = lineage.recall
        if metrics:
            client.log_metrics(rid, metrics)

        client.log_tags(rid, {
            "workflow_step": "model_registry",
            "registry_status": str(output.status),
            "registered_model_name": inp.model_name,
            "registered_model_version": str(output.version),
        })

        for artifact in output.registration_artifacts:
            client.log_artifact(rid, artifact.path)
